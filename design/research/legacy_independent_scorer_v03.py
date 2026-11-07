# Offline research-only calibration of two-card fragment risk against actual five-card opponent holdings.
# Not a tournament bot or submission component.
from itertools import combinations
from collections import Counter
from math import comb
import random, statistics, time

def card(s):
    ranks='6789TJQKA'; suits='cdhs'; return 4*ranks.index(s[0])+suits.index(s[1])

def five(cards):
    ranks=tuple(sorted((c//4 for c in cards),reverse=True))
    flush=len({c%4 for c in cards})==1
    if len(set(ranks))==5:
        top=(ranks[0] if ranks[0]-ranks[4]==4 else (3 if ranks==(8,3,2,1,0) else None))
        if top is not None: return (8 if flush else 3,(top,))
        return (6 if flush else 0,ranks)
    g=sorted(Counter(ranks).items(),key=lambda rc:(rc[1],rc[0]),reverse=True)
    shape=tuple(n for _,n in g)
    cat={(4,1):7,(3,2):5,(3,1,1):4,(2,2,1):2,(2,1,1,1):1}[shape]
    return cat,tuple(r for r,_ in g)

def best(hole, board):
    return max(five(p+t) for p in combinations(hole,2) for t in combinations(board,3))

def risks(hero,board):
    own=best(hero,board)
    u=tuple(c for c in range(36) if c not in set(hero)|set(board))
    pscore={(a,b):best((a,b),board) for a,b in combinations(u,2)}
    beat={p for p,s in pscore.items() if s>own}
    tie={p for p,s in pscore.items() if s==own}
    bad=ties=0
    for v in combinations(u,5):
        xs=[(a,b) for a,b in combinations(v,2)]
        if any(p in beat for p in xs):bad+=1
        elif any(p in tie for p in xs):ties+=1
    return {'hero_score':own,'pair_beating':len(beat),'pair_threat':len(beat)/len(pscore), 'opp5_beating':bad,'opp5_risk':bad/comb(len(u),5),'opp5_tieonly':ties/comb(len(u),5)}

if __name__=='__main__':
    h=tuple(map(card,'Kh Kd 8c 7s 6h'.split())); b=tuple(map(card,'Ah Ac Kc Qd 6s'.split()))
    t=time.time(); r=risks(h,b);print('reference',r,'seconds',round(time.time()-t,2))
    rng=random.Random(8037);rows=[]
    for street_len in [3,4,5]:
      for i in range(80):
        deck=rng.sample(range(36),5+street_len);h,b=tuple(deck[:5]),tuple(deck[5:]); v=risks(h,b);rows.append((street_len,i,*v.values()))
    print('sample_count',len(rows),'seconds',round(time.time()-t,2))
    out='/mnt/data/poker_design_research/river_calibration_samples.csv'
    import csv
    with open(out,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['board_size','id','hero_score','pair_beating','pair_threat','opp5_beating','opp5_risk','opp5_tieonly']);w.writerows(rows)
    
    for st in (3,4,5):
      for lo,hi in ((0,0),(0,.03),(.03,.12),(.12,1)):
        r5=[float(x[6]) for x in rows if int(x[0])==st and ((lo < float(x[4])<=hi) if lo else (float(x[4])<=hi))]
        print('street',st,'pairbucket',lo,hi,'n',len(r5),'mean5risk',round(statistics.mean(r5),4) if r5 else '-')
