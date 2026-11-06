"""Checkpoint state machine. Motion needs a live decision from 0G; there is no local substitute.

The 0G call runs on a worker thread. It is sent shortly before a checkpoint so the rover can roll
straight into its next zone. If no valid answer is ready at the checkpoint the wheels stop, and a
failed call stops them as soon as it is seen.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from isaac.brain.client import FINISH, BrainUnavailable, Decision

STEPS_PER_SECOND = 24
RETRY_START_STEPS = 5 * STEPS_PER_SECOND
RETRY_MAX_STEPS = 30 * STEPS_PER_SECOND


NET_X = 0.2
ZONE_W = 5.0
ZONE_H = 3.75
COLS = 6
MAX_BALL_OPTIONS = 3
MAX_OPTIONS = 4
MAX_EXPLORE_WITH_BALLS = 1


def build_options(snap: "Snapshot", tried: set[int], near_left: bool | None = None) -> list[dict]:
    """Zones worth deciding between: known balls first, then the nearest unexplored zones.

    near_left limits the choice to that side of the net until it has nothing left to offer.
    """
    options = []
    for zone, balls in enumerate(snap.known_balls):
        unexplored = not snap.visited[zone] and balls == 0
        if zone in tried or (balls == 0 and not unexplored):
            continue
        x = -15.0 + (zone % COLS + 0.5) * ZONE_W
        y = -7.5 + (zone // COLS + 0.5) * ZONE_H
        options.append({
            "zone": zone, "balls": balls,
            "metres": round(((x - snap.x) ** 2 + (y - snap.y) ** 2) ** 0.5),
            "crosses_net": (x - NET_X) * (snap.x - NET_X) < 0,
            "unexplored": unexplored,
        })
    if near_left is not None:
        near = [o for o in options if (o["zone"] % COLS < COLS / 2) == near_left]
        options = near or options
    with_balls = sorted((o for o in options if o["balls"]), key=lambda o: -o["balls"])[:MAX_BALL_OPTIONS]
    empty = sorted((o for o in options if not o["balls"]), key=lambda o: o["metres"])
    room = MAX_EXPLORE_WITH_BALLS if with_balls else MAX_OPTIONS
    return with_balls + empty[:room]


class Phase(str, Enum):
    THINKING = "thinking"
    DRIVING = "driving"
    SAFE_STOP = "safe_stop"
    DONE = "done"


@dataclass(frozen=True)
class Snapshot:
    """What the rover itself knows. Must never contain undiscovered ball positions."""

    step: int
    x: float
    y: float
    yaw_deg: float
    zone: int
    stuck: bool
    collected: int
    known_balls: list[int]
    visited: list[bool]


@dataclass(frozen=True)
class Command:
    drive: bool
    focus_zone: int
    phase: Phase


class Brain(Protocol):
    def decide(self, state: dict) -> Decision: ...


class EventLog(Protocol):
    def decision(self, step: int, trigger: str, state: dict, decision: Decision) -> None: ...
    def safe_stop(self, step: int, trigger: str, error: str) -> None: ...
    def resumed(self, step: int) -> None: ...


class Mission:
    def __init__(self, brain: Brain, log: EventLog, brief: str, limit_steps: int,
                 zone_budget_steps: int = 45 * STEPS_PER_SECOND,
                 settle_steps: int = 3 * STEPS_PER_SECOND,
                 stuck_steps: int = 5 * STEPS_PER_SECOND,
                 near_side_first: bool = False):
        self.brain = brain
        self.log = log
        self.brief = brief
        self.limit_steps = limit_steps
        self.zone_budget_steps = zone_budget_steps
        self.settle_steps = settle_steps
        self.stuck_steps = stuck_steps
        self.near_side_first = near_side_first
        self.phase = Phase.THINKING
        self.focus_zone = -1
        self.history: list[dict] = []
        self.last_reason = ""
        self.end_reason = "time_limit"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="0g-brain")
        self._future: Future | None = None
        self._state: dict = {}
        self._trigger = "mission_start"
        self._stale = False
        self._near_left: bool | None = None
        self._assigned_step = 0
        self._arrived_empty = 0
        self._stuck_for = 0
        self._collected_at_assign = 0
        self._retry_at = 0
        self._retry_wait = RETRY_START_STEPS

    def update(self, snap: Snapshot) -> Command:
        if self.phase is not Phase.DONE and snap.step >= self.limit_steps:
            self.phase = Phase.DONE
        if self.phase is Phase.DRIVING:
            self._drop_stale_answer(snap)
            trigger = self._checkpoint(snap)
            if self._early_ask_failed():
                self.history.append(self._outcome(snap))
                self.phase = Phase.THINKING
            elif trigger:
                self.history.append(self._outcome(snap))
                self._trigger = trigger
                self.phase = Phase.THINKING
            else:
                self._ask_ahead(snap)
        if self.phase in (Phase.THINKING, Phase.SAFE_STOP):
            self._think(snap)
        drive = self.phase is Phase.DRIVING
        return Command(drive=drive, focus_zone=self.focus_zone if drive else -1, phase=self.phase)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _checkpoint(self, snap: Snapshot) -> str | None:
        self._stuck_for = self._stuck_for + 1 if snap.stuck else 0
        if self._stuck_for >= self.stuck_steps:
            return "stuck"
        if snap.zone == self.focus_zone and snap.known_balls[self.focus_zone] == 0:
            self._arrived_empty += 1
            if self._arrived_empty >= self.settle_steps:
                return "zone_done"
        else:
            self._arrived_empty = 0
        if snap.step - self._assigned_step >= self.zone_budget_steps:
            return "zone_budget_spent"
        return None

    def _outcome(self, snap: Snapshot) -> dict:
        return {
            "zone": self.focus_zone,
            "balls_collected": snap.collected - self._collected_at_assign,
            "seconds": round((snap.step - self._assigned_step) / STEPS_PER_SECOND, 1),
        }

    def _early_ask_failed(self) -> bool:
        future = self._future
        return future is not None and not self._stale and future.done() and future.exception() is not None

    def _drop_stale_answer(self, snap: Snapshot) -> None:
        """Balls found in the zone mean the rover is not leaving, so an early answer no longer applies."""
        if self._future is None:
            return
        self._stale = self._stale or snap.known_balls[self.focus_zone] > 0
        if self._stale and self._future.done():
            self._future = None
            self._stale = False

    def _ask_ahead(self, snap: Snapshot) -> None:
        """Send the next question early, so the answer is ready when this assignment ends."""
        if self._future is not None:
            return
        budget_ask = self.zone_budget_steps - self.settle_steps
        if self._arrived_empty >= 1:
            trigger = "zone_done"
        elif snap.step - self._assigned_step >= budget_ask:
            trigger = "zone_budget_spent"
        else:
            return
        state = self._build_state(snap, trigger, [self._outcome(snap)])
        if state["options"]:
            self._trigger, self._state = trigger, state
            self._future = self._pool.submit(self.brain.decide, state)

    def _think(self, snap: Snapshot) -> None:
        if self._future is None:
            if self.phase is Phase.SAFE_STOP and snap.step < self._retry_at:
                return
            self._state = self._build_state(snap, self._trigger)
            if not self._state["options"]:
                self.phase = Phase.DONE
                self.end_reason = "no_zones_left"
                return
            self._future = self._pool.submit(self.brain.decide, self._state)
            return
        if not self._future.done():
            return
        future, self._future = self._future, None
        if self._stale:
            self._stale = False
            return
        try:
            decision = future.result()
        except BrainUnavailable as exc:
            self._enter_safe_stop(snap, str(exc))
            return
        self._apply(snap, decision)

    def _enter_safe_stop(self, snap: Snapshot, error: str) -> None:
        if self.phase is Phase.SAFE_STOP:
            self._retry_wait = min(self._retry_wait * 2, RETRY_MAX_STEPS)
        else:
            self.phase = Phase.SAFE_STOP
            self._retry_wait = RETRY_START_STEPS
        self._retry_at = snap.step + self._retry_wait
        self.log.safe_stop(snap.step, self._trigger, error)

    def _apply(self, snap: Snapshot, decision: Decision) -> None:
        if self.phase is Phase.SAFE_STOP:
            self.log.resumed(snap.step)
        self.log.decision(snap.step, self._trigger, self._state, decision)
        self.last_reason = decision.reason
        if decision.action == FINISH:
            self.phase = Phase.DONE
            self.end_reason = "brain_finished"
            return
        self.focus_zone = decision.zone
        self.phase = Phase.DRIVING
        self._assigned_step = snap.step
        self._collected_at_assign = snap.collected
        self._arrived_empty = 0
        self._stuck_for = 0

    def _build_state(self, snap: Snapshot, trigger: str, current: list[dict] | None = None) -> dict:
        history = (self.history + (current or []))[-6:]
        tried = {item["zone"] for item in history if item["balls_collected"] == 0}
        if self.near_side_first and self._near_left is None:
            self._near_left = snap.x < NET_X
        return {
            "options": build_options(snap, tried, self._near_left),
            "mission_brief": self.brief,
            "trigger": trigger,
            "seconds_left": round((self.limit_steps - snap.step) / STEPS_PER_SECOND),
            "rover": {"x": round(snap.x, 1), "y": round(snap.y, 1), "heading_deg": round(snap.yaw_deg),
                      "zone": snap.zone, "stuck": snap.stuck},
            "balls_collected": snap.collected,
            "known_balls_by_zone": {str(z): n for z, n in enumerate(snap.known_balls) if n},
            "unvisited_zones": [z for z, seen in enumerate(snap.visited) if not seen],
            "previous_assignments": history,
        }
