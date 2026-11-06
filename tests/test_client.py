import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from isaac.brain.client import BrainUnavailable, ZeroGBrain, parse_answer

GOOD = '{"action":"go_to_zone","zone":7,"reason":"most balls"}'


def reply(content=GOOD, trace=None):
    trace = {"request_id": "req-1", "provider": "0xabc", "tee_verified": True} if trace is None else trace
    return {"id": "chat-1", "choices": [{"message": {"content": content}}], "x_0g_trace": trace}


@pytest.fixture
def router():
    """Local stand-in for the Router. Maps model name to (status, body, headers)."""
    routes, seen = {}, []
    trust_modes = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append(body)
            trust_modes.append(self.headers.get("X-0G-Provider-Trust-Mode"))
            status, payload, headers = routes[body["model"]]
            raw = json.dumps(payload).encode() if isinstance(payload, dict) else payload
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    routes["_trust_modes"] = trust_modes
    yield routes, seen, f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def brain(url, *models):
    return ZeroGBrain("sk-test", list(models or ["a"]), base_url=url, answer_timeout_s=2)


def test_decision_keeps_request_id_and_proof(router):
    routes, seen, url = router
    routes["a"] = (200, reply(), {"ZG-Res-Key": "chat-header"})
    decision = brain(url).decide({"x": 1})
    assert (decision.action, decision.zone) == ("go_to_zone", 7)
    assert decision.request_id == "req-1" and decision.chat_id == "chat-header"
    assert decision.tee_verified is True and decision.provider == "0xabc"
    assert seen[0]["verify_tee"] is True


def test_request_id_falls_back_to_header(router):
    routes, _, url = router
    routes["a"] = (200, reply(trace={"provider": "0xabc"}), {"X-Request-Id": "hdr-9"})
    assert brain(url).decide({}).request_id == "hdr-9"


def test_missing_request_id_is_rejected(router):
    routes, _, url = router
    routes["a"] = (200, reply(trace={"provider": "0xabc"}), {})
    with pytest.raises(BrainUnavailable, match="no request id"):
        brain(url).decide({})


def test_failed_tee_check_is_rejected(router):
    routes, _, url = router
    routes["a"] = (200, reply(trace={"request_id": "r", "tee_verified": False}), {})
    with pytest.raises(BrainUnavailable, match="TEE"):
        brain(url).decide({})


@pytest.mark.parametrize("status", [401, 402, 429, 500, 502, 503])
def test_http_errors_give_no_decision(router, status):
    routes, _, url = router
    routes["a"] = (status, {"error": {"code": "x"}}, {})
    with pytest.raises(BrainUnavailable):
        brain(url).decide({})


def test_garbage_body_gives_no_decision(router):
    routes, _, url = router
    routes["a"] = (200, b"not json", {})
    with pytest.raises(BrainUnavailable):
        brain(url).decide({})


def test_next_0g_model_is_tried_after_failure(router):
    routes, _, url = router
    routes["a"] = (503, {"error": {}}, {})
    routes["b"] = (200, reply(), {})
    assert brain(url, "a", "b").decide({}).model == "b"


def test_unreachable_router_gives_no_decision():
    with pytest.raises(BrainUnavailable):
        ZeroGBrain("k", ["a"], base_url="http://127.0.0.1:9").decide({})


@pytest.mark.parametrize("text", [
    "", "no json", "{broken", '{"action":"spin"}', '{"action":"go_to_zone"}',
    '{"action":"go_to_zone","zone":24}', '{"action":"go_to_zone","zone":-1}',
    '{"action":"go_to_zone","zone":"3"}', '{"action":"go_to_zone","zone":true}',
])
def test_invalid_answers_are_rejected(text):
    with pytest.raises(BrainUnavailable):
        parse_answer(text)


def test_answer_inside_code_fence_parses():
    assert parse_answer(f"```json\n{GOOD}\n```")[:2] == ("go_to_zone", 7)


def test_finish_needs_no_zone():
    assert parse_answer('{"action":"finish","reason":"time"}') == ("finish", None, "time")


STATE = {"mission_brief": "b", "seconds_left": 60, "options": [
    {"zone": 7, "balls": 3, "metres": 6, "crosses_net": False, "unexplored": False},
    {"zone": 20, "balls": 0, "metres": 25, "crosses_net": True, "unexplored": True},
]}


def decision_reply(choice="zone_7", trace=None):
    trace = {"request_id": "req-d", "provider": "0xabc", "tee_verified": True} if trace is None else trace
    return {"answers": {"next_zone": {"choice": choice, "answer_confidence": 0.61,
                                      "probabilities": {"zone_7": 0.61, "zone_20": 0.39}}},
            "x_0g_trace": trace}


def test_decision_model_keeps_probabilities_and_proof(router):
    routes, seen, url = router
    routes["laya"] = (200, decision_reply(), {"ZG-Res-Key": "k"})
    decision = brain(url, "laya").decide(STATE)
    assert decision.zone == 7 and decision.request_id == "req-d" and decision.tee_verified is True
    assert decision.probabilities == {"zone_7": 0.61, "zone_20": 0.39} and decision.chat_id == "k"
    criteria = seen[0]["questions"]["next_zone"]["criteria"]
    assert criteria["zone_20"] == "0 known balls, 25 m away, across the net, unexplored"
    assert seen[0]["verify_tee"] is True


def test_zone_that_was_not_offered_is_rejected(router):
    routes, _, url = router
    routes["laya"] = (200, decision_reply("zone_3"), {})
    with pytest.raises(BrainUnavailable, match="not offered"):
        brain(url, "laya").decide(STATE)


def test_malformed_decision_answer_is_rejected(router):
    routes, _, url = router
    routes["laya"] = (200, {"answers": {}, "x_0g_trace": {"request_id": "r"}}, {})
    with pytest.raises(BrainUnavailable, match="malformed"):
        brain(url, "laya").decide(STATE)


def test_failed_tee_check_on_decision_model_is_rejected(router):
    routes, _, url = router
    routes["laya"] = (200, decision_reply(trace={"request_id": "r", "tee_verified": False}), {})
    with pytest.raises(BrainUnavailable, match="TEE"):
        brain(url, "laya").decide(STATE)


def test_combined_request_id_header_uses_the_router_id(router):
    routes, _, url = router
    routes["laya"] = (200, decision_reply(trace={"provider": "0x"}), {"X-Request-Id": "edge-1, router-2"})
    assert brain(url, "laya").decide(STATE).request_id == "router-2"


def test_every_request_demands_sealed_inference(router):
    routes, _, url = router
    routes["laya"] = (200, decision_reply(), {})
    brain(url, "laya").decide(STATE)
    assert routes["_trust_modes"] == ["private"]
