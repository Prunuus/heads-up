#!/bin/bash
# Real-judge check: our image (build it first with ./tools/check_build.sh -> poker-agent-check:latest) at --cpus 1 vs opponent servers; halves run separately with fresh processes. Usage: bash docs/handoff/verify/realjudge.sh [outdir]
REPO=$(git rev-parse --show-toplevel); PY=${PY:-$REPO/.venv/bin/python}; OUT=${1:-/tmp/realjudge}; mkdir -p $OUT; cd $REPO
PIN=""; command -v taskset >/dev/null && PIN="taskset -c 4-7"   # Linux only; macOS has no taskset
portfree() { for i in $(seq 100); do (lsof -i :$1 >/dev/null 2>&1 || ss -ltn 2>/dev/null | grep -q ":$1 ") || return 0; sleep 0.1; done; return 1; }
OPPS="baseline-bots/baseline_random.py baseline-bots/baseline_calling_station.py baseline-bots/baseline_always_half.py baseline-bots/baseline_always_pot.py sim/opponents/llm/aggressive.py sim/opponents/llm/mc_equity.py"
for opp in $OPPS; do for seed in r1 r2; do
  name=$(basename $opp .py)
  for half in 1 2; do
    docker rm -f rj-ours >/dev/null 2>&1; portfree 15008; portfree 15009 || { echo "PORT 15009 BUSY"; exit 3; }
    docker run -d --name rj-ours --read-only --tmpfs /tmp:rw,nosuid,nodev,size=512m --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --cpus 1 --cpuset-cpus 0-3 --memory 4g -e PORT=5008 -p 127.0.0.1:15008:5008 poker-agent-check:latest >/dev/null
    PYTHONPATH=$REPO PORT=15009 $PIN $PY $opp >/dev/null 2>&1 &
    OPP=$!
    for i in $(seq 150); do curl -s -m 1 http://127.0.0.1:15008/ >/dev/null && curl -s -m 1 http://127.0.0.1:15009/ >/dev/null && break; sleep 0.2; done
    $PIN $PY judge_engine.py --agent1 http://127.0.0.1:15008 --agent2 http://127.0.0.1:15009 --half $half --seed $seed --json $OUT/${name}_${seed}_h${half}.json >/dev/null 2>&1
    kill $OPP 2>/dev/null; wait $OPP 2>/dev/null; docker rm -f rj-ours >/dev/null 2>&1
  done
  $PY judge_engine.py combine $OUT/${name}_${seed}_h1.json $OUT/${name}_${seed}_h2.json > $OUT/${name}_${seed}_combined.txt 2>&1
done; done
$PY - "$OUT" <<'PY'
import json, glob, sys
out = sys.argv[1]
for f in sorted(glob.glob(out + "/*_h*.json")):
    d = json.load(open(f)); s = d["agent_stats"]; a = s[0] if isinstance(s, list) else s.get("agent1", s)
    print(f.split("/")[-1], "|", d["agent2_name"], "| nets", sum(d["nets_agent1"]), "| errors", a.get("errors"), "forfeited", a.get("forfeited"), "bank", round(a.get("time_seconds", 0), 3), "timed_out", a.get("timed_out"))
for f in sorted(glob.glob(out + "/*_combined.txt")):
    t = open(f).read().strip().splitlines(); print(f.split("/")[-1], "::", " / ".join(l.strip() for l in t if any(k in l for k in ("winner", "WIN", "CHOP", "verdict", "bb/deal", "FORFEIT")))[:220])
PY
