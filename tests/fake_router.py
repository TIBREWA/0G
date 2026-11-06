"""Local stand-in for the 0G Router, for integration tests only. Request ids are prefixed FAKE.

    python isaac/tests/fake_router.py 8799
    ZG_ROUTER_URL=http://127.0.0.1:8799/v1 ZG_API_KEY=test ZG_MODELS=fake python isaac/run_mission.py --headless
"""

import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def choose(state: dict) -> dict:
    known = state["known_balls_by_zone"]
    tried = {item["zone"] for item in state["previous_assignments"] if item["balls_collected"] == 0}
    options = {int(z): n for z, n in known.items() if int(z) not in tried}
    if options:
        zone = max(options, key=options.get)
        return {"action": "go_to_zone", "zone": zone, "reason": f"{options[zone]} known balls here"}
    unseen = [z for z in state["unvisited_zones"] if z not in tried]
    if unseen:
        return {"action": "go_to_zone", "zone": unseen[0], "reason": "exploring an unvisited zone"}
    return {"action": "finish", "reason": "nothing left to try"}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        state = json.loads(body["messages"][1]["content"])
        request_id = f"FAKE-{uuid.uuid4().hex[:12]}"
        raw = json.dumps({
            "id": request_id,
            "choices": [{"message": {"content": json.dumps(choose(state))}}],
            "x_0g_trace": {"request_id": request_id, "provider": "0xFAKE", "tee_verified": None},
        }).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
