"""Decision client for the 0G Compute Router.

Every outcome is either a Decision that 0G actually returned, carrying its request id, or
BrainUnavailable. There is deliberately no default answer: the caller must stop the robot.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass

import requests

ROUTER_URL = os.environ.get("ZG_ROUTER_URL", "https://router-api.0g.ai/v1")
CONNECT_TIMEOUT_S = 5.0
ANSWER_TIMEOUT_S = float(os.environ.get("ZG_TIMEOUT_S", "30"))
GO_TO_ZONE = "go_to_zone"
FINISH = "finish"
DECISION_MODELS = {"laya", "vela-2.0-9b"}
DECISION_INSTRUCTIONS = (
    "Pick the option that best achieves the goal: prefer options with more balls and shorter "
    "travel, and follow any instruction in the goal about the net."
)

SYSTEM_PROMPT = """You are the mission brain of a tennis-ball collecting rover on a 30 m x 15 m court.
A low-level driving policy handles motion. You choose where it works next.

The court is a 6 x 4 grid of zones. zone = row * 6 + col. Column 0 is x = -15 m, column 5 is
x = +15 m, row 0 is y = -7.5 m, row 3 is y = +7.5 m. The net runs along x = 0.2 m, between
columns 2 and 3, and can only be passed through the gaps at its two ends (|y| > 6.35 m).
Crossing it costs a long detour. Balls are only known once the rover has seen them, so zones
that are unvisited may hold balls you cannot see yet.

Weigh the mission brief, the time left, the known balls, unexplored zones and past results.
Reply with one JSON object and nothing else:
{"action": "go_to_zone", "zone": <0-23>, "reason": "<one short sentence>"}
or, when continuing is not worthwhile:
{"action": "finish", "reason": "<one short sentence>"}"""


class BrainUnavailable(Exception):
    """0G gave no usable decision. The robot must stay stopped."""


@dataclass(frozen=True)
class Decision:
    action: str
    zone: int | None
    reason: str
    request_id: str
    model: str
    provider: str | None
    chat_id: str | None
    tee_verified: bool | None
    latency_s: float
    probabilities: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_answer(text: str) -> tuple[str, int | None, str]:
    """Read and validate the model's answer. Raises BrainUnavailable if it is unusable."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match is None:
        raise BrainUnavailable(f"no JSON in answer: {text!r:.200}")
    try:
        answer = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise BrainUnavailable(f"bad JSON in answer: {exc}") from exc
    action = answer.get("action")
    reason = str(answer.get("reason", "")).strip()
    if action == FINISH:
        return FINISH, None, reason
    zone = answer.get("zone")
    if action != GO_TO_ZONE or isinstance(zone, bool) or not isinstance(zone, int) or not 0 <= zone <= 23:
        raise BrainUnavailable(f"invalid decision: {answer!r:.200}")
    return GO_TO_ZONE, zone, reason


def describe_option(option: dict) -> str:
    text = f"{option['balls']} known balls, {option['metres']} m away, "
    text += "across the net" if option["crosses_net"] else "same side of the net"
    return text + (", unexplored" if option["unexplored"] else "")


def decision_request(model: str, state: dict) -> dict:
    """Body for POST /decisions. State stays short because the model truncates long input."""
    goal = f"Goal: {state['mission_brief']} {state['seconds_left']} seconds left."
    return {
        "model": model,
        "state": goal,
        "verify_tee": True,
        "questions": {"next_zone": {
            "type": "choice",
            "instructions": DECISION_INSTRUCTIONS,
            "criteria": {f"zone_{o['zone']}": describe_option(o) for o in state["options"]},
        }},
    }


def offered(state: dict, zone: int | None) -> bool:
    options = state.get("options")
    return zone is None or options is None or any(o["zone"] == zone for o in options)


class ZeroGBrain:
    def __init__(self, api_key: str, models: list[str], base_url: str = ROUTER_URL,
                 answer_timeout_s: float = ANSWER_TIMEOUT_S):
        if not models:
            raise ValueError("at least one 0G model is required")
        self.api_key = api_key
        self.models = models
        self.base_url = base_url.rstrip("/")
        self.answer_timeout_s = answer_timeout_s

    def decide(self, state: dict) -> Decision:
        """Ask 0G for a decision, moving to the next 0G model if one fails."""
        failures = []
        for model in self.models:
            try:
                return self._ask(model, state)
            except BrainUnavailable as exc:
                failures.append(f"{model}: {exc}")
        raise BrainUnavailable("; ".join(failures))

    def _ask(self, model: str, state: dict) -> Decision:
        if model in DECISION_MODELS:
            return self._ask_decision_model(model, state)
        return self._ask_chat_model(model, state)

    def _post(self, path: str, body: dict) -> tuple[requests.Response, dict, float]:
        started = time.monotonic()
        try:
            response = requests.post(
                f"{self.base_url}/{path}",
                headers={"Authorization": f"Bearer {self.api_key}", "X-0G-Provider-Trust-Mode": "private"},
                json=body,
                timeout=(CONNECT_TIMEOUT_S, self.answer_timeout_s),
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise BrainUnavailable(str(exc)) from exc
        return response, data, round(time.monotonic() - started, 3)

    @staticmethod
    def _proof(response: requests.Response, data: dict) -> tuple[dict, str]:
        """Return the trace and request id, refusing anything without valid proof."""
        trace = data.get("x_0g_trace") or {}
        header_id = response.headers.get("X-Request-Id", "").split(",")[-1].strip()
        request_id = trace.get("request_id") or header_id
        if not request_id:
            raise BrainUnavailable("response has no request id")
        if trace.get("tee_verified") is False:
            raise BrainUnavailable(f"TEE signature failed for request {request_id}")
        return trace, request_id

    def _ask_decision_model(self, model: str, state: dict) -> Decision:
        response, data, latency_s = self._post("decisions", decision_request(model, state))
        trace, request_id = self._proof(response, data)
        try:
            answer = data["answers"]["next_zone"]
            zone = int(answer["choice"].removeprefix("zone_"))
            confidence = float(answer["answer_confidence"])
            probabilities = {k: float(v) for k, v in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise BrainUnavailable(f"malformed answer for request {request_id}") from exc
        if not offered(state, zone):
            raise BrainUnavailable(f"zone {zone} was not offered, request {request_id}")
        return Decision(
            action=GO_TO_ZONE, zone=zone, reason=f"{confidence:.0%} probability", request_id=request_id,
            model=model, provider=trace.get("provider"), chat_id=response.headers.get("ZG-Res-Key"),
            tee_verified=trace.get("tee_verified"), latency_s=latency_s, probabilities=probabilities,
        )

    def _ask_chat_model(self, model: str, state: dict) -> Decision:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(state, separators=(",", ":"))},
            ],
            "temperature": 0.2,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
            "verify_tee": True,
        }
        response, data, latency_s = self._post("chat/completions", body)
        trace, request_id = self._proof(response, data)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BrainUnavailable(f"malformed response for request {request_id}") from exc

        action, zone, reason = parse_answer(content)
        if not offered(state, zone):
            raise BrainUnavailable(f"zone {zone} was not offered, request {request_id}")
        return Decision(
            action=action, zone=zone, reason=reason, request_id=request_id, model=model,
            provider=trace.get("provider"),
            chat_id=response.headers.get("ZG-Res-Key") or data.get("id"),
            tee_verified=trace.get("tee_verified"), latency_s=latency_s,
        )
