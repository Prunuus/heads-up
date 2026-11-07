"""Evidence-only E1 diagnostics. Does not approve a mode without E0 evidence."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from game import Action, AgentConfig, Game, Street, make_deal
from poker_eval import FULL, REDUCED, assess
from poker_memory import initialize, ingest
from poker_policy import canonical, postflop, preflop

ORDER = {'WEAK':0, 'MEDIUM':1, 'STRONG':2, 'NUTS':3}


def wilson(k, n):
    if not n:
        return [0., 1.]
    z = 1.96
    center = (k / n + z*z / (2*n)) / (1 + z*z/n)
    spread = z * sqrt(k/n*(1-k/n)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center-spread, center+spread]


def texture(board):
    r = max(Counter(c >> 2 for c in board).values())
    s = max(Counter(c & 3 for c in board).values())
    return {'paired': r >= 2, 'trips': r >= 3, 'suited': s >= 3,
            'ordinary': r == 1 and s < 3}


def rank_values(values):
    result = [0.] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j - 1) / 2
        i = j
    return result


def spearman(x, y):
    a, b = rank_values(x), rank_values(y)
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    return sum((v-ma)*(w-mb) for v,w in zip(a,b)) / sqrt(sum((v-ma)**2 for v in a)*sum((v-mb)**2 for v in b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('reports'))
    args = ap.parse_args()
    out = args.output
    inputs = [Path('design/research/fresh_stratified_inputs_v04.csv'), out / 'fresh_cal31_inputs.csv']
    refs = [out / 'fresh_official.csv', out / 'fresh_cal31_official.csv']
    rows, salted = [], []
    tiers, reduced_tiers, streets, textures = Counter(), Counter(), Counter(), Counter()
    street_tiers = defaultdict(Counter)
    boundary = {'.03': 0, '.12': 0}
    by_stratum = defaultdict(Counter)
    tier_risks = defaultdict(list)
    upward = downward = 0
    for source, refpath in zip(inputs, refs):
        raw = list(csv.DictReader(source.open()))
        verified = list(csv.DictReader(refpath.open()))
        assert len(raw) == len(verified)
        for state, ref in zip(raw, verified):
            h = tuple(map(int, state['hole_ids'].split(',')))
            b = tuple(map(int, state['board_ids'].split(',')))
            full = assess(h, b, FULL)
            reduced = assess(h, b, REDUCED)
            assert (full.score, full.beating, full.ties, full.n) == tuple(int(ref[k]) for k in ('official_packed_hero_score','pair_beating','pair_tie','pair_N'))
            assert (full.tier, reduced.tier) == (ref['full_tier'], ref['reduced_tier'])
            tier_risks[full.tier].append((float(ref['opp5_risk']), int(ref['opp5_tieonly'])/int(ref['opp5_total'])))
            tiers[full.tier] += 1
            reduced_tiers[reduced.tier] += 1
            streets[len(b)] += 1
            street_tiers[len(b)][full.tier] += 1
            tags = texture(b)
            for key, yes in tags.items():
                textures[key] += yes
                if yes:
                    by_stratum[key]['n'] += 1
                    by_stratum[key]['upward'] += ORDER[reduced.tier] > ORDER[full.tier]
                    by_stratum[key]['downward'] += ORDER[reduced.tier] < ORDER[full.tier]
            for threshold in boundary:
                near = abs(full.risk-float(threshold)) <= .015
                boundary[threshold] += near
                if near:
                    by_stratum['boundary_'+threshold]['n'] += 1
                    by_stratum['boundary_'+threshold]['upward'] += ORDER[reduced.tier] > ORDER[full.tier]
                    by_stratum['boundary_'+threshold]['downward'] += ORDER[reduced.tier] < ORDER[full.tier]
            upward += ORDER[reduced.tier] > ORDER[full.tier]
            downward += ORDER[reduced.tier] < ORDER[full.tier]
            for salt in range(4):
                sample = assess(h, b, REDUCED, salt)
                salted.append({'source':str(source), 'id':state['id'], 'street':len(b)-2,
                               'salt':salt, 'full_tier':full.tier, 'reduced_tier':sample.tier,
                               'risk':sample.risk, 'hole_ids':state['hole_ids'], 'board_ids':state['board_ids']})
            rows.append((full, reduced, ref))
    coverage = (min(streets.values()) >= 60 and tiers['STRONG'] >= 60 and tiers['MEDIUM'] >= 30
                and tiers['WEAK'] >= 30 and min(textures[k] for k in ('paired','trips','suited')) >= 20
                and min(boundary.values()) >= 20)
    n = len(rows)
    severe = {mode:sum(a.tier == 'STRONG' and float(ref['opp5_risk']) >= .50
                     for f, r, ref in rows for a in (f if mode == FULL else r,)) for mode in (FULL, REDUCED)}
    strong = {FULL:tiers['STRONG'], REDUCED:reduced_tiers['STRONG']}
    actions, action_counts = [], Counter()
    rng = random.Random('threat-cal-59-reachable-actions')
    hand = 0
    while len(actions) < 1000:
        g = Game(make_deal('threat-cal-59', hand), hand % 2, hand_index=hand)
        while not g.is_terminal:
            obs = g.observation(g.to_act)
            if obs.street != Street.PREFLOP:
                m = {}
                initialize(AgentConfig(num_hands=10000), m)
                ingest(obs, m)
                label = ('UNKNOWN','STATION','AGGRESSIVE','TIGHT','OVERFOLDER')[len(actions)%5]
                m['resolved_post'] = 12
                full, samples = assess(obs.hole_cards,obs.board), [assess(obs.hole_cards,obs.board,REDUCED,s) for s in range(4)]
                fa = canonical(obs, postflop(obs,m,full,label,.55))
                ra = [canonical(obs, postflop(obs,m,a,label,.55)) for a in samples]
                row = {'hand':hand, 'street':obs.street.name, 'profile':label,
                       'hole_ids':','.join(map(str,obs.hole_cards)), 'board_ids':','.join(map(str,obs.board)),
                       'observation_json':json.dumps(obs.to_dict(),separators=(',',':')),
                       'full_tier':full.tier, 'full_action':fa.name,
                       **{f'salt{s}_action':a.name for s,a in enumerate(ra)},
                       **{f'salt{s}_tier':a.tier for s,a in enumerate(samples)}}
                actions.append(row)
                action_counts[obs.street.name+'_n'] += 1
                action_counts[obs.street.name+'_disagree'] += ra[0] != fa
                action_counts['full_'+fa.name] += 1
                action_counts['disagreement_from_'+fa.name] += ra[0] != fa
                for salt,a in enumerate(ra):
                    action_counts[f'salt{salt}_disagree'] += a != fa
            choices = [a for a in obs.legal_actions if a != Action.FOLD]
            g.apply(rng.choice(choices))
        hand += 1
    disagreement = action_counts['salt0_disagree'] / len(actions)
    full_ok = coverage and severe[FULL]/strong[FULL] <= .05 and wilson(severe[FULL],strong[FULL])[1] <= .10
    reduced_coverage = coverage and all(reduced_tiers[t] >= 40 for t in ('STRONG','MEDIUM','WEAK'))
    reduced_ok = (reduced_coverage and severe[REDUCED]/strong[REDUCED] <= .05
                  and wilson(severe[REDUCED],strong[REDUCED])[1] <= .10
                  and upward/n <= .10 and wilson(upward,n)[1] <= .20 and disagreement <= .05)
    result = {'fresh_states':n, 'streets':dict(streets), 'full_tiers':dict(tiers),
              'street_tiers':{k:dict(v) for k,v in street_tiers.items()},
              'spearman_fragment_vs_fivecard':spearman([f.risk for f,r,ref in rows],[float(ref['opp5_risk']) for f,r,ref in rows]),
              'reduced_tiers':dict(reduced_tiers), 'textures':dict(textures), 'boundary_coverage':boundary,
              'coverage_pass':coverage, 'reduced_coverage_pass':reduced_coverage,
              'strong_severe_mislabels': {mode:{'count':severe[mode], 'strong_n':strong[mode],
                                              'wilson_95':wilson(severe[mode],strong[mode])} for mode in (FULL,REDUCED)},
              'full_tier_fivecard_risks':{t:{'n':len(v), 'mean_stronger':sum(x[0] for x in v)/len(v),
                                               'mean_tieonly':sum(x[1] for x in v)/len(v)} for t,v in tier_risks.items()},
              'upward_tier': {'count':upward, 'rate':upward/n, 'wilson_95':wilson(upward,n)},
              'downward_tier': {'count':downward, 'rate':downward/n},
              'by_stratum':{k:dict(v) for k,v in by_stratum.items()},
              'reachable_actions':len(actions), 'action_disagreement':disagreement,
              'action_counts':dict(action_counts),
              'accuracy_diagnostics': {FULL:'PASS' if full_ok else 'INCONCLUSIVE' if not coverage else 'FAIL',
                                       REDUCED:'PASS' if reduced_ok else 'INCONCLUSIVE' if not reduced_coverage else 'FAIL'},
              'note':'Diagnostics only until E0 passes. T11/T12 approximation tolerances apply to REDUCED; FULL is the exact tier reference.',
              'inputs_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in inputs+refs}}
    for filename,data in (('salt_diagnostics.csv',salted),('action_comparison.csv',actions)):
        with (out / filename).open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    (out / 'accuracy.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
