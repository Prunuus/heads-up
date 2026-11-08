"""Official referee, fresh HTTP processes for every half, complete timing logs.

Run inside the official image with one CPU, read-only root, no network, 4GB.
The call_heavy/max_raise stress aliases are behaviorally identical to the
official station/pot baselines respectively, without relying on hand sampling.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path('/app') if Path('/app/agent.py').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import AgentConfig
from judge_engine import HttpAgentHandle, combine_halves, play_half
from poker_eval import FULL, REDUCED

OFFICIAL = ('baseline_random', 'baseline_calling_station', 'baseline_always_half', 'baseline_always_pot')
STRESS = OFFICIAL + ('call_heavy', 'max_raise')
SEEDS = {'E0': ('stress-41', 'stress-83'), 'dev': ('dev-11', 'dev-29'), 'final': ('final-73', 'final-97')}
FILES = ('agent.py', 'poker_eval.py', 'poker_memory.py', 'poker_policy.py', 'game.py', 'judge_engine.py', 'evaluator.py', 'requirements.txt')


class MeasuredHandle(HttpAgentHandle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latencies = []

    def _post(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return super()._post(*args, **kwargs)
        finally:
            self.latencies.append(time.perf_counter() - start)


def latency(values):
    ordered = sorted(values)
    return {'n': len(values), 'max_ms': max(values, default=0) * 1000,
            **{f'p{p}_ms': (ordered[min(len(ordered)-1, int(len(ordered)*p/100))] * 1000
                            if ordered else 0)
               for p in (50, 95, 99)}}


def run_half(mode, opponent, seed, half, out, adaptation, bluffs, deals, strategy='v04', keep_logs=False):
    alias = {'call_heavy': 'baseline_calling_station', 'max_raise': 'baseline_always_pot'}
    script = ROOT / 'baseline-bots' / (alias.get(opponent, opponent) + '.py')
    name = f'{mode}_{opponent}_{seed}_{half}'
    env = dict(os.environ, PORT='5008', PB_MODE=mode, PB_ADAPTATION=str(int(adaptation)),
               PB_PURE_BLUFFS=str(int(bluffs)), PB_STRATEGY=strategy,
               PB_MEMORY_OUTPUT=str(out / (name + '_memory.json')))
    logfile = (out / (name + '.log')).open('w')
    start = time.perf_counter()
    processes = []
    handles = []
    try:
        processes.append(subprocess.Popen([sys.executable, str(Path(__file__).with_name('serve_candidate.py'))], env=env, cwd=ROOT, stdout=logfile, stderr=logfile))
        processes.append(subprocess.Popen([sys.executable, str(script)], env=dict(os.environ, PORT='5009'), cwd=ROOT, stdout=logfile, stderr=logfile))
        token = os.environ.get('AGENT_TOKEN') or None
        handles = [MeasuredHandle('http://127.0.0.1:5008', AgentConfig(num_hands=deals), token),
                   MeasuredHandle('http://127.0.0.1:5009', AgentConfig(num_hands=deals), token)]
        if not all(h.wait_healthy(12.) for h in handles):
            raise RuntimeError('startup health failed')
        startup = time.perf_counter() - start
        result = play_half(*handles, seed, half, deals, keep_logs=keep_logs)
        raw = result.to_dict()
        extra = {'startup_seconds': startup, 'candidate_latency': latency(handles[0].latencies),
                 'opponent_latency': latency(handles[1].latencies)}
        status, state = handles[0]._request('GET', '/research/state', None, 1.)
        if status != 200:
            raise RuntimeError('research state inspection failed')
        extra['memory'] = state
        (out / (name + '_memory.json')).write_text(json.dumps(state), encoding='utf8')
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
        for h in handles:
            if h._conn:
                h._conn.close()
        logfile.close()
    raw['measurement'] = extra
    (out / (name + '.json')).write_text(json.dumps(raw, indent=2), encoding='utf8')
    s = result.stats_agent1
    runtime_pass = (s.errors == 0 and not s.forfeited and not s.timed_out
                    and s.time_seconds <= 5.5 and extra['candidate_latency']['max_ms'] <= 100
                    and startup <= 12 and extra['memory'].get('internal_errors') == 0)
    return result, extra, runtime_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=SEEDS, required=True)
    ap.add_argument('--mode', choices=(FULL, REDUCED, 'both'), default='both')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--adaptation', type=int, default=1)
    ap.add_argument('--bluffs', type=int, default=1)
    ap.add_argument('--deals', type=int, default=1000)
    ap.add_argument('--opponent', choices=STRESS)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    modes = (FULL, REDUCED) if args.mode == 'both' else (args.mode,)
    if args.stage != 'E0':
        freeze = json.loads((out.parent / 'freeze.json').read_text())
        if len(modes) != 1 or freeze['mode'] != modes[0] or freeze['status'] != 'APPROVED':
            raise RuntimeError('E0/E1 approved freeze required before strategy matches')
    if args.stage == 'final':
        final = json.loads((out.parent / 'final_freeze.json').read_text())
        if (bool(args.adaptation) != final['adaptation'] or bool(args.bluffs) != final['pure_bluffs']
                or modes[0] != final['mode']):
            raise RuntimeError('Final settings differ from development-selected frozen policy')
        for f,h in final['source_hashes'].items():
            if sha256((ROOT/f).read_bytes()).hexdigest() != h:
                raise RuntimeError('Source changed after final freeze: '+f)
    manifest = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                'platform': platform.platform(), 'stage': args.stage, 'deals': args.deals,
                'adaptation': args.adaptation, 'bluffs': args.bluffs,
                'hashes': {f: sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES},
                'official_commit': 'c83ab6eeb7660a2fd1fae7df224315d41fe985a9',
                'source_base_commit': 'e9ce98dc5749672c1d85fc9c12fcc3e901256969',
                'sampler_salt': 0, 'matches': []}
    manifest['configuration_hash'] = sha256(json.dumps({k: manifest[k] for k in ('hashes','adaptation','bluffs','sampler_salt')},sort_keys=True).encode()).hexdigest()
    registry = out / 'registry.json'
    if registry.exists():
        raise RuntimeError('Refusing to overwrite an experiment registry')
    opponents = (args.opponent,) if args.opponent else STRESS if args.stage == 'E0' else OFFICIAL
    for mode in modes:
        for opponent in opponents:
            for seed in SEEDS[args.stage]:
                halves, metrics, passing = [], [], []
                for half in (1, 2):
                    result, extra, ok = run_half(mode, opponent, seed, half, out, bool(args.adaptation), bool(args.bluffs), args.deals)
                    halves.append(result)
                    metrics.append(extra)
                    passing.append(ok)
                match = combine_halves(*halves).to_dict()
                match.update(mode=mode, opponent=opponent, measurements=metrics,
                             runtime_pass=all(passing) and args.deals == 1000,
                             configuration_hash=manifest['configuration_hash'])
                manifest['matches'].append(match)
                registry.write_text(json.dumps(manifest, indent=2), encoding='utf8')
                print(json.dumps({'mode':mode, 'opponent':opponent, 'seed':seed,
                                  'runtime_pass':match['runtime_pass'], 'charged':[h.stats_agent1.time_seconds for h in halves],
                                  'max_ms':[m['candidate_latency']['max_ms'] for m in metrics],
                                  'verdict': match.get('verdict')}), flush=True)


if __name__ == '__main__':
    main()
