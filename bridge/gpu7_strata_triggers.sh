#!/usr/bin/env bash
# Run this ON gpu7 (yiqun@10.6.32.13). trux cannot reach gpu7, but gpu7 can reach trux,
# so every transfer here is gpu7-initiated.
#
# Purpose: compile prose trigger_conditions into executable applicability predicates using the
# strata-served model, then hand the result back to trux. The local Ollama models on trux are not
# strong enough (measured: qwen3:32b timed out after 800 s because the GPUs are ~80% occupied by
# other users; the smaller models returned 0 accepted predicates out of 4 cards).
set -euo pipefail

TRUX=trux-tsubame.uni.lux
TRUX_USER=yewei
KEY=$HOME/.ssh/id_ed25519_gm_transfer          # the dedicated gm-transfer key
REPO=$HOME/gm_scratch/GramMetaRL
SRC=$TRUX_USER@$TRUX:/raid/yewei/projects_yewei/GramMetaRL
CARDS=${CARDS:-data/rules/mbg_cards_precise.jsonl}
OUT=${OUT:-data/rules/mbg_cards_precise_app_strata.jsonl}
MODEL=${MODEL:-qwen3.8-flash-next-iq3_s}
KEYJSON=${KEYJSON:-$HOME/.local/state/strata/launch/iq3_s-256k.json}

SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $KEY"

# 1. pull the unified code + card sets from trux (never rsync the other direction)
mkdir -p "$REPO"
rsync -e "$SSH" -av \
  --include 'grammetarl/***' --include 'scripts/***' --include 'data/rules/***' \
  --include 'main.py' --include 'pyproject.toml' --exclude '*' \
  "$SRC/" "$REPO/"

cd "$REPO"
test -x .venv-gm/bin/python || { echo "no venv at .venv-gm; create it first (see README)"; exit 1; }

# 2. compile with strata. The key is read from the strata launch file without printing it, and
#    the token budget is mandatory: this is a thinking model and an unbounded request will burn
#    the whole 262k context and return content=null.
export GM_STRATA_KEY=$(.venv-gm/bin/python -c "import json,sys;print(json.load(open('$KEYJSON'))['api_key'])")
.venv-gm/bin/python scripts/compile_triggers.py \
  --cards "$CARDS" --out "$OUT" \
  --report artifacts/triggers/strata_report.json \
  --endpoint http://127.0.0.1:8081/v1 --model "$MODEL" \
  --api-key-env GM_STRATA_KEY --batch-size 6 --max-tokens 6000 --timeout 900
unset GM_STRATA_KEY

# 3. hand the result back to trux as a NEW file (never overwrite another session's card set)
rsync -e "$SSH" -av "$OUT" "$SRC/data/rules/"
rsync -e "$SSH" -av artifacts/triggers/ "$SRC/artifacts/triggers/"
echo "done: $OUT is on trux; re-measure there with"
echo "  python scripts/audit_constraint_precision.py --emit-pairs   # replays the judge cache, no LLM"
