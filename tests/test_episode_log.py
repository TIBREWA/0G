import json

from isaac.brain.client import Decision
from isaac.brain.episode_log import EpisodeLog, sha256_file


def test_every_decision_is_saved_with_its_request_id(tmp_path):
    log = EpisodeLog("ep1", {"seed": 7})
    log.decision(10, "mission_start", {"s": 1}, Decision("go_to_zone", 3, "r", "req-9", "m", "0xp", "c", None, 1.2))
    log.safe_stop(20, "stuck", "router down")
    log.resumed(50)
    log.ball_collected(30, 4)
    log.pose(6, 1.0, 2.0, 0.1, "driving", 3)
    log.finish(60, "time_limit")
    path = log.save(tmp_path)
    data = json.loads(path.read_text())
    assert data["decisions"][0]["request_id"] == "req-9"
    assert data["safe_stops"][0]["ai_decision"] is False
    assert data["result"] == {"end_step": 60, "reason": "time_limit", "balls_collected": 1, "decisions": 1, "safe_stops": 1}
    assert sha256_file(path).startswith("0x") and len(sha256_file(path)) == 66
