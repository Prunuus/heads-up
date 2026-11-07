#!/usr/bin/env python3
# Corrected runner; the supplied original remains under design/research/.
"""Research-only reproducibility/calibration utility. NEVER imported by tournament agent.

Usage (from a clone of the official pinned heads-up-bundle):
 python path/to/calibrate_offline.py recover --legacy path/to/street_calibration_samples.csv --output research/pilot_recovered_v04.csv
 python path/to/calibrate_offline.py verify --repo path/to/heads-up-bundle --input research/pilot_recovered_v04.csv --output research/pilot_official_verification_v04.csv --slow 3

The 240 legacy cards are RECONSTRUCTED, not present in the v0.3 CSV. Reconstruction
uses the deterministic RNG recipe in the shipped v0.3 research script; agreement
with reported outcomes must be measured against the OFFICIAL evaluator.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, math, random, sys, time, subprocess
from itertools import combinations
from pathlib import Path

SEED = 8037
STREETS = (3, 4, 5)
COUNTS = {3:80, 4:80, 5:80}
SOURCE_COMMIT = 'c83ab6eeb7660a2fd1fae7df224315d41fe985a9'


def recover(legacy_path:Path, output:Path):
    rows = list(csv.DictReader(legacy_path.open(newline='')))
    assert len(rows)==240, f'Expected 240 legacy rows; got {len(rows)}'
    by_key={(int(r['board_size']),int(r['id'])):r for r in rows}
    assert len(by_key)==240, 'Duplicate pilot keys'
    rng=random.Random(SEED)
    output.parent.mkdir(parents=True,exist_ok=True)
    columns=['board_size','id','hole_ids','board_ids','hero_score_legacy','pair_beating_legacy','pair_threat_legacy','opp5_beating_legacy','opp5_risk_legacy','opp5_tieonly_legacy','provenance']
    with output.open('w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=columns,lineterminator='\n');wr.writeheader()
        for b in STREETS:
            for i in range(COUNTS[b]):
                selected=rng.sample(range(36),5+b)
                r=by_key[(b,i)]
                wr.writerow({'board_size':b,'id':i,'hole_ids':','.join(map(str,selected[:5])),'board_ids':','.join(map(str,selected[5:])),
                             'hero_score_legacy':r['hero_score'],'pair_beating_legacy':r['pair_beating'],
                             'pair_threat_legacy':r['pair_threat'],'opp5_beating_legacy':r['opp5_beating'],
                             'opp5_risk_legacy':r['opp5_risk'],'opp5_tieonly_legacy':r['opp5_tieonly'],
                             'provenance':'reconstructed_from_seed_8037_not_official_verified'})
    print('recovered_rows',len(rows),'output_sha256',hashlib.sha256(output.read_bytes()).hexdigest())


def generate_fresh(output:Path, seed:str, per_stratum:int):
    """Create version-independent SHA256-ranked raw card observations, unverified until 'verify'."""
    if per_stratum<1:raise ValueError('per_stratum must be >=1')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as out:
        wr=csv.DictWriter(out,fieldnames=['board_size','id','hole_ids','board_ids','stratum','generator_seed','trial'],lineterminator='\n');wr.writeheader()
        for b in STREETS:
            for stratum in ('ordinary','paired','trips','suited'):
                kept=0
                for trial in range(250000):
                    prefix=b'PB4/CAL\x00'+seed.encode('utf-8')+bytes([b])+stratum.encode('ascii')+trial.to_bytes(4,'big')
                    seq=sorted(range(36),key=lambda c:hashlib.sha256(prefix+bytes([c])).digest())[:5+b]
                    h,board=seq[:5],seq[5:]
                    from collections import Counter
                    ranks=Counter(x>>2 for x in board);suits=Counter(x&3 for x in board)
                    if stratum=='ordinary':valid=(max(ranks.values())==1 and max(suits.values())<3)
                    elif stratum=='paired':valid=(max(ranks.values())==2)
                    elif stratum=='trips':valid=(max(ranks.values())>=3)
                    else:valid=(max(suits.values())>=3)
                    if not valid:continue
                    wr.writerow({'board_size':b,'id':f'{stratum}_{kept}', 'hole_ids':','.join(map(str,h)),
                                 'board_ids':','.join(map(str,board)), 'stratum':stratum,
                                 'generator_seed':seed,'trial':trial})
                    kept+=1
                    if kept>=per_stratum:break
                if kept<per_stratum:raise SystemExit(f'Insufficient generator states for {b=} {stratum=}')
    print('fresh_states',3*4*per_stratum,'sha256',hashlib.sha256(output.read_bytes()).hexdigest())


def reduced_indices(hole, board, salt=0):
    """Exact REDUCED_64_v04 research fixture sampler (no Python RNG dependency)."""
    assert len(hole)==5 and len(board) in (3,4,5)
    assert 0<=salt<2**32
    street={3:1,4:2,5:3}[len(board)]
    u=tuple(x for x in range(36) if x not in set(hole)|set(board))
    n=math.comb(len(u),2)
    payload=b'PB4/REDUCED64\x00'+bytes(sorted(hole))+bytes([street,len(board)])+bytes(board)+salt.to_bytes(4,'big')
    digest=hashlib.sha256(payload).digest()
    offset=int.from_bytes(digest[0:8],'big')%n
    step=1+int.from_bytes(digest[8:16],'big')%(n-1)
    while math.gcd(step,n)!=1:step=step%(n-1)+1
    return tuple(sorted((offset+j*step)%n for j in range(min(64,n))))


def tiers(full_pair_risk, reduced_pair_risk, danger, k, full_B, locked):
    if locked:return 'NUTS','NUTS'
    full = 'STRONG' if full_B==0 or full_pair_risk<=.03 else 'MEDIUM' if full_pair_risk<=.12 else 'WEAK'
    reduced='STRONG' if reduced_pair_risk<=.03 and (not danger or k>=5) else 'MEDIUM' if reduced_pair_risk<=.12 else 'WEAK'
    return full,reduced


def load_official(repo: Path):
    path=repo/'evaluator.py'
    try:
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    except Exception as e:
        raise SystemExit(f'Official repo must be a git checkout: {e}')
    if head!=SOURCE_COMMIT:raise SystemExit(f'WRONG OFFICIAL COMMIT: got {head}; expected {SOURCE_COMMIT}')
    modified=subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--','evaluator.py'],text=True).strip()
    if modified:raise SystemExit('Official evaluator.py modified locally; refuse verification')
    if not path.is_file():raise SystemExit(f'Official evaluator.py not found: {path}')
    modspec=importlib.util.spec_from_file_location('pinned_official_evaluator_for_research',path)
    assert modspec and modspec.loader
    mod=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(mod)
    mod.warmup()
    return mod


def score_partial(cards,board,table):
    return max(table[sum(1<<x for x in pair+triple)]
               for pair in combinations(cards,2) for triple in combinations(board,3))


def fragment_graph(hole,board,table):
    known=set(hole)|set(board); unseen=tuple(c for c in range(36) if c not in known)
    hscore=score_partial(hole,board,table)
    beating=set(); ties=set()
    for a,b in combinations(unseen,2):
        sc=max(table[sum(1<<x for x in (a,b)+triple)] for triple in combinations(board,3))
        if sc>hscore:beating.add((a,b))
        elif sc==hscore:ties.add((a,b))
    return hscore,unseen,beating,ties


def exact_opp5_graph(unseen,beating,ties):
    count_beating=count_tieonly=0
    for hand in combinations(unseen,5):
        fragments=combinations(hand,2)
        if any(pair in beating for pair in fragments): count_beating+=1
        elif any(pair in ties for pair in combinations(hand,2)):count_tieonly+=1
    return count_beating,count_tieonly


def exact_opp5_slow(unseen,board,hero_score,table):
    beating=tieonly=0
    for hole in combinations(unseen,5):
        sc=score_partial(hole,board,table)  # independent of graph construction
        beating+=(sc>hero_score)
        tieonly+=(sc==hero_score)
    return beating,tieonly


def fmt_ids(s):return tuple(int(z) for z in s.split(','))


def verify(repo:Path, input_csv:Path, output:Path,slow:int):
    ev=load_official(repo);table=ev.get_table();data=list(csv.DictReader(input_csv.open(newline='')))
    if not data:raise SystemExit('Empty input state CSV')
    output.parent.mkdir(parents=True,exist_ok=True)
    cols=['board_size','id','hole_ids','board_ids','official_packed_hero_score','pair_beating','pair_tie','pair_N','opp5_beating','opp5_tieonly','opp5_total','pair_risk','opp5_risk','legacy_beating_match','legacy_opp5_match','legacy_hero_score_match','slow_reference_match','full_tier','reduced_tier','tier_disagreement','sampled_B','sampled_pair_risk','sampler_salt']
    mismatches=[];t0=time.perf_counter();slow_covered=set()
    with output.open('w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=cols,lineterminator='\n');wr.writeheader()
        for ix,r in enumerate(data):
            b=int(r['board_size']);h=fmt_ids(r['hole_ids']);board=fmt_ids(r['board_ids'])
            if len(h)!=5 or len(board)!=b or len(set(h+board))!=5+b or any(not 0<=c<36 for c in h+board):
                raise ValueError(f'Invalid physical cards at row {ix}')
            score,unseen,beating,ties=fragment_graph(h,board,table)
            allpairs=tuple(combinations(unseen,2))
            ids=reduced_indices(h,board,0)
            sample_B=sum(allpairs[index] in beating for index in ids)
            sampled_risk=sample_B/len(ids)
            k=score>>20
            danger=(max(sum((card&3)==s for card in board) for s in range(4))>=3
                    or max(sum(c>>2==r for c in board) for r in range(9))>=3
                    or len({c>>2 for c in board})<len(board) and k<5)
            locked=(k==8 and ((score>>16)&15)==8) or (b==5 and not beating)
            full_tier,reduced_tier=tiers(len(beating)/len(allpairs),sampled_risk,danger,k,len(beating),locked)
            maximum=(score>>20)==8 and ((score>>16)&15)==8
            if not maximum:
                reduced_tier='STRONG' if sampled_risk<=.03 and (not danger or k>=5) else 'MEDIUM' if sampled_risk<=.12 else 'WEAK'
            cnt,tieonly=exact_opp5_graph(unseen,beating,ties)
            if b==5 and score!=ev.omaha_score(h,board):raise AssertionError('Official complete-board comparison failed')
            # Official score from the legacy tuple is NOT assumed correct; it is checked.
            import ast
            if r.get('hero_score_legacy'):
                cat,ranklist=ast.literal_eval(r['hero_score_legacy'])
                packed=(cat<<20)
                for j,k in enumerate(ranklist):packed|=k<<(16-4*j)
            else:packed=None
            row={'board_size':b,'id':r['id'],'hole_ids':r['hole_ids'],'board_ids':r['board_ids'],
                'official_packed_hero_score':score,'pair_beating':len(beating),'pair_tie':len(ties),
                'pair_N':math.comb(len(unseen),2),'opp5_beating':cnt,'opp5_tieonly':tieonly,
                'opp5_total':math.comb(len(unseen),5),'pair_risk':len(beating)/math.comb(len(unseen),2),
                'opp5_risk':cnt/math.comb(len(unseen),5),
                'legacy_beating_match':int(len(beating)==int(r['pair_beating_legacy'])) if r.get('pair_beating_legacy') else '',
                'legacy_opp5_match':int(cnt==int(r['opp5_beating_legacy']) and abs(cnt/math.comb(len(unseen),5)-float(r['opp5_risk_legacy']))<=1e-12 and abs(tieonly/math.comb(len(unseen),5)-float(r['opp5_tieonly_legacy']))<=1e-12) if r.get('opp5_beating_legacy') else '',
                'legacy_hero_score_match':int(packed==score) if r.get('hero_score_legacy') else '','slow_reference_match':'',
                'full_tier':full_tier,'reduced_tier':reduced_tier,'tier_disagreement':int(full_tier!=reduced_tier),
                'sampled_B':sample_B,'sampled_pair_risk':sampled_risk,'sampler_salt':0}
            if b not in slow_covered and len(slow_covered)<slow:
                slow_covered.add(b)
                slow_counts=exact_opp5_slow(unseen,board,score,table)
                row['slow_reference_match']=int(slow_counts==(cnt,tieonly))
            wr.writerow(row)
            if any(row[k] == 0 for k in ('legacy_beating_match','legacy_opp5_match','legacy_hero_score_match','slow_reference_match')):
                mismatches.append((b,r['id']))
    print('verified_rows',len(data),'mismatches',mismatches[:12],'total_mismatches',len(mismatches),'seconds',round(time.perf_counter()-t0,2))
    print('official_eval_commit_expected',SOURCE_COMMIT,'IMPORTANT: verify repo HEAD externally before certifying')
    print('result_sha256',hashlib.sha256(output.read_bytes()).hexdigest())
    if mismatches:raise SystemExit(2)


def check_fixtures(path:Path):
    import json
    rows=json.loads(path.read_text())
    if len(rows)<6:raise SystemExit('Fixture file too small')
    for ix,r in enumerate(rows):
        actual=reduced_indices(tuple(r['hole_ids']),tuple(r['board_ids']),int(r['salt']))
        if actual!=tuple(r['sample_indices']):raise AssertionError(f'Fixture indices differ: {ix}')
        digest=hashlib.sha256(','.join(map(str,actual)).encode()).hexdigest()
        if digest!=r['index_digest_sha256']:raise AssertionError(f'Fixture digest differs: {ix}')
    print('sampler_fixture_count',len(rows),'PASS')


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('recover');p.add_argument('--legacy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sp.add_parser('generate');p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',default='threat-cal-13');p.add_argument('--per-stratum',type=int,default=20)
    p=sp.add_parser('check-fixtures');p.add_argument('--path',type=Path,required=True)
    p=sp.add_parser('verify');p.add_argument('--repo',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--slow',type=int,default=3)
    a=ap.parse_args()
    if a.cmd=='recover':recover(a.legacy,a.output)
    elif a.cmd=='generate':generate_fresh(a.output,a.seed,a.per_stratum)
    elif a.cmd=='check-fixtures':check_fixtures(a.path)
    else:verify(a.repo,a.input,a.output,a.slow)

if __name__=='__main__':main()
