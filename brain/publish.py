"""Upload an episode to 0G Storage and record its root on 0G Chain.

    python -m isaac.brain.publish isaac/episodes/<id>.json

Reads ZG_PRIVATE_KEY and ZG_REGISTRY_ADDRESS from the shell or isaac/.env. Failures leave the local
file in place, so publishing can simply be run again.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from isaac.brain.episode_log import sha256_file
from isaac.brain.settings import load_env

CHAIN_DIR = Path(__file__).resolve().parent.parent / "chain"


def publish(path: Path) -> dict:
    episode = json.loads(path.read_text())
    result = episode["result"] or {}
    run = subprocess.run(
        ["node", "publish.mjs", str(path.resolve()), sha256_file(path),
         str(result.get("decisions", 0)), str(result.get("safe_stops", 0))],
        cwd=CHAIN_DIR, capture_output=True, text=True,
    )
    if run.returncode != 0:
        raise RuntimeError(run.stderr.strip() or "publish failed")
    receipt = json.loads(run.stdout.strip().splitlines()[-1])
    path.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    load_env()
    print(json.dumps(publish(Path(sys.argv[1])), indent=2))
