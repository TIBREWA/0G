#!/usr/bin/env bash
# Deploys EpisodeRegistry to 0G Galileo with the wallet in ZG_PRIVATE_KEY as recorder.
set -euo pipefail
cd "$(dirname "$0")/.."
RECORDER=$(cast wallet address --private-key "$ZG_PRIVATE_KEY")
forge create src/EpisodeRegistry.sol:EpisodeRegistry \
  --rpc-url https://evmrpc-testnet.0g.ai --private-key "$ZG_PRIVATE_KEY" \
  --broadcast --constructor-args "$RECORDER"
