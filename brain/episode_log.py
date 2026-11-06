"""Episode record: enough to audit every decision and replay the run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from isaac.brain.client import Decision

SCHEMA = "tennisbot-episode/1"


class EpisodeLog:
    def __init__(self, episode_id: str, meta: dict):
        self.data: dict = {
            "schema": SCHEMA, "episode_id": episode_id, **meta,
            "decisions": [], "safe_stops": [], "resumes": [],
            "collected": [], "trajectory": [], "result": None,
        }

    def decision(self, step: int, trigger: str, state: dict, decision: Decision) -> None:
        self.data["decisions"].append({"step": step, "trigger": trigger, "state": state, **decision.to_dict()})

    def safe_stop(self, step: int, trigger: str, error: str) -> None:
        self.data["safe_stops"].append({"step": step, "trigger": trigger, "error": error[:500], "ai_decision": False})

    def resumed(self, step: int) -> None:
        self.data["resumes"].append(step)

    def ball_collected(self, step: int, ball: int) -> None:
        self.data["collected"].append({"step": step, "ball": ball})

    def pose(self, step: int, x: float, y: float, yaw: float, phase: str, focus_zone: int) -> None:
        self.data["trajectory"].append([step, round(x, 3), round(y, 3), round(yaw, 3), phase, focus_zone])

    def finish(self, step: int, reason: str) -> None:
        self.data["result"] = {
            "end_step": step, "reason": reason,
            "balls_collected": len(self.data["collected"]),
            "decisions": len(self.data["decisions"]),
            "safe_stops": len(self.data["safe_stops"]),
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.data['episode_id']}.json"
        path.write_text(json.dumps(self.data, separators=(",", ":")))
        return path


def sha256_file(path: Path) -> str:
    return "0x" + hashlib.sha256(path.read_bytes()).hexdigest()
