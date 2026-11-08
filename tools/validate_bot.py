"""Validate the deployed configuration against the official HTTP opponents.

Run inside a restricted tournament container with the repository mounted at
/repo and an empty writable output directory under /tmp or another mount.
Each half starts fresh bot processes. No historical experiment files are used.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent
from judge_engine import combine_halves
from research.run_matches import FILES, OFFICIAL, ROOT as EXECUTION_ROOT, run_half


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', default='audit-fixes-20261107')
    parser.add_argument('--deals', type=int, default=1000)
    parser.add_argument('--opponents', nargs='+', choices=OFFICIAL, default=OFFICIAL)
    args = parser.parse_args()
    if args.deals <= 0:
        parser.error('--deals must be positive')
    hashes = {f: sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}
    for name, digest in hashes.items():
        if sha256((EXECUTION_ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                f'Executable source differs from the validation checkout: {name}; '
                'rebuild the container image before validating this checkout')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'seed': args.seed, 'deals': args.deals,
        'strategy': agent.STRATEGY, 'mode': agent.EVALUATION_MODE,
        'adaptation': agent.ADAPTATION, 'pure_bluffs': agent.PURE_BLUFFS,
        'source_hashes': hashes,
        'execution_root': str(EXECUTION_ROOT),
        'method': 'official HTTP judge; fresh servers each half', 'matches': [],
    }
    path = args.output / 'registry.json'
    # Reserve the run before starting subprocesses, including failed runs.
    path.write_text(json.dumps(report, indent=2), encoding='utf8')
    for opponent in dict.fromkeys(args.opponents):
        halves, measurements, passed = [], [], []
        for half in (1, 2):
            result, metrics, ok = run_half(
                agent.EVALUATION_MODE, opponent, args.seed, half, args.output,
                agent.ADAPTATION, agent.PURE_BLUFFS, args.deals, agent.STRATEGY)
            halves.append(result)
            measurements.append(metrics)
            passed.append(ok)
        match = combine_halves(*halves).to_dict()
        match.update(opponent=opponent, measurements=measurements,
                     runtime_pass=all(passed) and args.deals == 1000)
        report['matches'].append(match)
        path.write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps({
            'opponent': opponent, 'verdict': match['verdict'],
            'runtime_pass': match['runtime_pass'],
            'charged_seconds': [h.stats_agent1.time_seconds for h in halves],
            'internal_errors': [m['memory']['internal_errors'] for m in measurements],
        }), flush=True)
    return 0 if all(m['runtime_pass'] for m in report['matches']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
