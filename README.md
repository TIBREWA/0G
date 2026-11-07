# Tennisbot: a rover whose decisions run on 0G

A tennis-ball collecting rover in Isaac Sim. A trained PPO policy drives the wheels on the device
at 24 Hz. At each checkpoint 0G decides which zone to work next. Without a live decision from 0G
the rover stops.

## What existed before the event

Submitted on the ETHGlobal Continuity track.

**Existed before the hackathon** (first commit in this repo, dated from April 2026 by file time):
the Isaac Lab rover environment (`tennisbot_env.py`, `env_cfg.py`), training setup (`train.py`,
`agents/`), debug tools (`court_viewer.py`, `debug_obs.py`, `obs_monitor.py`), the court and rover
models (`model/`) and the trained policy (`runs/agent_44000.pt`).

**Built during the hackathon:** the 0G decision client, mission logic and safe stop (`brain/`), the
mission and comparison runners (`run_mission.py`, `run_baseline.py`), the focus-zone changes to
`tennisbot_env.py` and `env_cfg.py`, the `EpisodeRegistry` contract and publishing (`chain/`), the
replay page (`web/`), the tests (`tests/`), the recorded runs (`episodes/`) and the demo video.

This work ran from 22:19 on 6 November to the morning of 7 November 2026 IST. The git repository
was created at the end of that period, so the history was rebuilt afterwards:

- Each commit's author date is the real time its files were last changed, taken from file times
  and session records. Its commit date is the real time it entered git. See both with
  `git log --format=fuller`.
- Commits group files by when they last changed, so some early commits do not run on their own.

## Use of AI tools

This project was built with Claude Code (Anthropic). No spec-driven workflow was used.

- **Generated with AI, directed and reviewed by the team:** `brain/`, `run_mission.py`,
  `run_baseline.py`, `tests/`, `chain/`, `web/`, the focus-zone changes in `tennisbot_env.py` and
  `env_cfg.py`, this README and the assembly of the demo video.
- **Team:** the pre-existing rover, simulation and trained policy, the choice to put decisions on
  0G, the requirements, and the decisions on what to build and keep.

## How it works

1. The rover reaches a checkpoint: mission start, zone done, stuck, or zone time budget spent.
2. About 3 s before the rover finishes a zone, it builds a short option list from what it has actually seen.
3. `brain/client.py` sends the list to `POST /v1/decisions` on the 0G Compute Router, using the
   TEE-backed `laya` model with `X-0G-Provider-Trust-Mode: private` and `verify_tee: true`.
4. The chosen zone goes to the policy through the focus-zone hook in `tennisbot_env.py`.
5. Each decision is logged with its request id, provider, TEE result and probabilities.
6. `brain/publish.py` uploads the log to 0G Storage and records its root on 0G Chain.

## Safe stop

A timeout, HTTP error, malformed answer, zone that was not offered, failed TEE check or missing
request id all give the same result: wheels stop, no decision is recorded, and the rover retries
with backoff. There is no local decision path. The next question goes out about 3 s before the
checkpoint, so the rover usually rolls straight into its next zone. A failed call stops the wheels
as soon as it is seen, which can be up to about 42 s into an assignment.

## Setup

The Python package is named `isaac`, so clone into a folder of that name and run from its parent:
`git clone <repo-url> isaac`.

Requires Isaac Lab (tested with isaaclab 0.54.3) and skrl 1.4.3, an NVIDIA GPU, Node 20+ and Foundry.

```bash
cp isaac/.env.example isaac/.env          # add your keys
(cd isaac/chain && npm install)
python isaac/run_mission.py --headless --seed 8   # add --near_side_first for the near-side mission
python -m isaac.brain.publish isaac/episodes/<id>.json
```

Tests: `python -m pytest isaac/tests` and `cd isaac/chain && forge test`.

Deploy the registry with `isaac/chain/script/deploy.sh`.

Open the replay with any static server from `isaac/web`, for example `python -m http.server`.

Add `--frames <folder>` to `run_mission.py` to save overhead camera frames. The demo video is `isaac/video/tennisbot_demo.mp4`.

## Results

Three courses, 180 s limit, live 0G decisions with `laya`. One run per setup, so treat these as a small sample. Reproduce the first column with `python isaac/run_baseline.py --headless --seed 9`.

| Course | Policy alone | 0G | 0G, near side first |
|---|---|---|---|
| Seed 7 | 9 of 19 | 15 of 19 | 12 of 19 |
| Seed 8 | 27 of 27, 89 s | 27 of 27, 83 s | 24 of 27 |
| Seed 9 | 22 of 22, 82 s | 22 of 22, 131 s | 22 of 22, 168 s |
| Total | 58 of 68 | 64 of 68 | 58 of 68 |

The policy alone cannot be told what to do. With `--near_side_first`, the mission code holds back far-side zones until the starting side is finished, and 0G chooses among what remains:

| Course | Starting side finished: policy alone | 0G, near side first | Near-side balls collected before crossing: policy alone | 0G, near side first |
|---|---|---|---|---|
| Seed 7 | 85 s | 58 s | 3 of 8 | 8 of 8 |
| Seed 9 | 82 s | 39 s | 5 of 10 | 10 of 10 |

Seed 8 has no balls on the starting side, so the rule only costs time there. The net rule lives in our code, not in `laya`, which did not follow it reliably on its own.

## Deployment

Network: 0G Galileo testnet, chain 16602.

| Item | Value |
|---|---|
| `EpisodeRegistry` | [`0x398a7a21C2836a1BE3304433e2aA20954C5dA69C`](https://chainscan-galileo.0g.ai/address/0x398a7a21C2836a1BE3304433e2aA20954C5dA69C) |
| Deploy transaction | [`0x95ba0ebe…cccf8`](https://chainscan-galileo.0g.ai/tx/0x95ba0ebe00867d5eeae85e5e85a7fba84ec7330219fc7c3893d8d754fe2cccf8) |

Final runs:

| Episode | Result | 0G Storage root | Record transaction |
|---|---|---|---|
| `ep-1794002629-s9` | Near side first: 22 of 22 balls, 13 of 13 decisions TEE verified | `0xab3189e97d9b4ebfa275bcec1f00c238add50f1672f9df59e7f399e817ab8f4c` | [`0x861fd1b9…8ab4f`](https://chainscan-galileo.0g.ai/tx/0x861fd1b90ada082b7f0415dc0ce1681412817a8bd1c5d03ffb6f54a05982ab4f) |
| `ep-1794002210-s8` | Fastest: 27 of 27 balls in 83 s, 6 of 6 TEE verified | `0x377b2518674cf77107d0e5cd43d02be14af18ad09cbd14191225aa41ed3c3f25` | [`0x0c14768b…0b80f`](https://chainscan-galileo.0g.ai/tx/0x0c14768b5e84f8bf205293c72e323dde97a605c707bc4e042f9df08f54afb80f) |
| `ep-1794002975-s9` | Live outage: safe stop for 70 s, 0 balls collected during it, then resumed | `0x6ba4109d53b6babad495b9a6a39c26cd2caa73014e6c9aa0d4f6425d55871da9` | [`0xf52d8033…340ff`](https://chainscan-galileo.0g.ai/tx/0xf52d80337fdab031be0c308aa59a450c8208bd2adfbe6f7ed22ac56bd0a340ff) |

Runs recorded for the demo video:

| Episode | Result | 0G Storage root | Record transaction |
|---|---|---|---|
| `ep-1794004168-s9` | Near side first: 21 of 22 balls, near side cleared at 51 s, 14 of 14 TEE verified | `0xbb6727fe8589d685e93410857ac3291901483b21b2c4f0adbad5bcb049d262ca` | [`0x1ce2dbdd…6c2c4`](https://chainscan-galileo.0g.ai/tx/0x1ce2dbdda1268fd6d4137580215d7cd01481aaea3c48fc220551653ef7f6c2c4) |
| `ep-1794004361-s9` | Live outage: route cut at 30 s, safe stop from 60 s to 79 s, 5 of 5 TEE verified | `0xc47feb7656c0b72dc417aba13333431830a1855e6faf7bb1f140ece575a47f57` | [`0x381084f4…c86`](https://chainscan-galileo.0g.ai/tx/0x381084f4330d2314d2d99a74c465b7b2876d8430b3ae64ef5ba25ed342356c86) |

Earlier runs stay on chain, because the registry cannot delete records.

The replay page checks each log against the hash on 0G Chain. Open `?root=<storage root>` to load a log directly from 0G Storage.

## Limits

- The rover pose comes from the simulator, standing in for a localisation system, because the
  dead-reckoned odometry drifts by several metres.
- `laya` is a classifier, so `reason` holds its probability, not a written explanation. Chat models
  such as `glm-5.3` can be set in `ZG_MODELS` but need a Router balance, and have not been tested
  against a live 0G provider.
