"""Regression tests for opponent-conditioned defense; v04 remains unchanged."""
from dataclasses import replace

import pytest
from game import Action,AgentConfig,Game,Street,make_deal
from poker_eval import Assessment,FULL
from poker_memory import initialize
from poker_memory import profile
from poker_policy import call_ceiling,preflop,postflop,STRATEGIES
from poker_policy import choose
from poker_memory import finish


def context(street=Street.FLOP):
    g=Game(make_deal('v05-policy',0),0)
    g.apply(Action.CHECK_CALL);g.apply(Action.CHECK_CALL)
    while g.street<street:
        g.apply(Action.CHECK_CALL);g.apply(Action.CHECK_CALL)
    g.apply(Action.RAISE_POT)
    return g.observation(g.to_act)


def assessment(tier):
    return Assessment(4<<20,325,1,0,.02,0.,tier,False,False,False,False,0.,FULL)


@pytest.mark.parametrize('street',[Street.FLOP,Street.TURN,Street.RIVER])
@pytest.mark.parametrize('tier',['MEDIUM','STRONG'])
def test_confirmed_aggression_does_not_veto_profitable_price_bands(street,tier):
    o=context(street)
    a=assessment(tier)
    assert o.to_call/(o.pot+o.to_call)==1/3
    assert call_ceiling(o,a,'AGGRESSIVE','v05_defend30')==.34
    assert call_ceiling(replace(o,raises_this_street=4),a,'AGGRESSIVE','v05_defend30')==.34
    m={};initialize(AgentConfig(),m);m['strategy']='v05_defend30'
    assert postflop(o,m,a,'AGGRESSIVE',.32)==Action.CHECK_CALL
    assert call_ceiling(o,a,'AGGRESSIVE')<1/3


@pytest.mark.parametrize('label',['UNKNOWN','TIGHT','STATION','OVERFOLDER'])
@pytest.mark.parametrize('tier',['WEAK','MEDIUM','STRONG'])
def test_nonaggressive_opponents_retain_v04_price_policy(label,tier):
    o=context()
    a=assessment(tier)
    assert call_ceiling(o,a,label,'v05_limp')==call_ceiling(o,a,label,'v04')


def test_weak_hands_do_not_gain_blanket_aggressive_calls():
    o=context()
    a=assessment('WEAK')
    assert call_ceiling(o,a,'AGGRESSIVE','v05_defend30')==call_ceiling(o,a,'AGGRESSIVE','v04')
    assert call_ceiling(o,a,'AGGRESSIVE','v05_defend30')<1/3


@pytest.mark.parametrize('configuration,threshold',[
    ('v05_defend35',35),('v05_defend30',30),('v05_limp20',20),
    ('v05_limp25',25),('v05_limp35',35),('v05_limp40',40),('v05_limp45',45)])
def test_preflop_defense_boundary(configuration,threshold,monkeypatch):
    import poker_policy
    g=Game(make_deal('v05-pre',0),0);g.apply(Action.RAISE_POT)
    o=g.observation(1)
    m={};initialize(AgentConfig(),m);m['strategy']=configuration
    for score,expected in ((threshold-1,Action.FOLD),(threshold,Action.CHECK_CALL)):
        monkeypatch.setattr(poker_policy,'preflop_score',lambda h:score)
        assert preflop(o,m,'AGGRESSIVE',.32)==expected
    assert preflop(o,m,'UNKNOWN',.32)==Action.FOLD


def test_trapping_requires_demonstrated_aggression(monkeypatch):
    import poker_policy
    monkeypatch.setattr(poker_policy,'preflop_score',lambda h:85)
    g=Game(make_deal('v05-trap',0),0);g.apply(Action.RAISE_HALF)
    o=g.observation(1)
    m={};initialize(AgentConfig(),m);m['strategy']='v05_trap'
    assert preflop(o,m,'AGGRESSIVE',.32)==Action.CHECK_CALL
    assert preflop(o,m,'UNKNOWN',.32)==Action.RAISE_POT


def test_open_limp_change_is_conditioned_on_aggression(monkeypatch):
    import poker_policy
    monkeypatch.setattr(poker_policy,'preflop_score',lambda h:85)
    o=Game(make_deal('v05-limp',0),0).observation(0)
    m={};initialize(AgentConfig(),m);m['strategy']='v05_limp'
    assert preflop(o,m,'AGGRESSIVE',.32)==Action.CHECK_CALL
    assert preflop(o,m,'UNKNOWN',.32)==Action.RAISE_POT


def test_frequent_raiser_who_never_folds_is_aggressive_not_station():
    m={};initialize(AgentConfig(),m)
    m.update(first_observed=100,vpip=100,pre_raise_hands=95,post_nonfold=200,
             post_raises=180,resolved_post=30,post_folds=0)
    assert profile(m)[0]=='STATION'  # preserve the frozen control
    assert profile(m,aggression_first=True)[0]=='AGGRESSIVE'
    m.update(pre_raise_hands=0,post_raises=0)
    assert profile(m,aggression_first=True)[0]=='STATION'


@pytest.mark.parametrize('configuration',['v05_aggressive30','v05_aggressive_limp','v05_limp20'])
def test_ten_thousand_reachable_v05_decisions(configuration):
    from collections import Counter
    from pathlib import Path
    import json,random
    rng=random.Random('v05-reachability')
    memories=[{},{}]
    for m in memories:
        initialize(AgentConfig(num_hands=10000),m,diagnostics=True)
        m['strategy']=configuration
    count=0;hand=0;coverage=Counter()
    while count<10000:
        g=Game(make_deal('v05-correctness-only',hand),hand%2,hand_index=hand)
        while not g.is_terminal:
            m=memories[g.to_act]
            obs=g.observation(g.to_act)
            proposed=choose(obs,m)
            assert proposed in obs.legal_actions
            assert m['internal_errors']==0,m['diagnostics']
            coverage[obs.street.name]+=1
            coverage['chosen_'+proposed.name]+=1
            coverage['raise_cap']+=obs.raises_this_street==4
            coverage['aggressive_profile']+=profile(m,aggression_first=True)[0]=='AGGRESSIVE'
            count+=1
            executed=rng.choice([a for a in obs.legal_actions if a!=Action.FOLD])
            if executed!=proposed:
                m['pending']=None  # deliberately unexecuted proposal creates no opportunity
            g.apply(executed)
        for seat in (0,1):
            finish(g.hand_result(seat),memories[seat])
            assert memories[seat]['internal_errors']==0,memories[seat]['diagnostics']
        hand+=1
    assert all(coverage[s.name]>100 for s in Street)
    assert coverage['raise_cap']>100 and coverage['aggressive_profile']>1000
    Path('reports/v05/reachability_'+configuration+'.json').write_text(json.dumps(dict(coverage),indent=2),encoding='utf8')
