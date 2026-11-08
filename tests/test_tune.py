"""Task 12: the HeuristicParams tuner (sim/tune.py)."""

import dataclasses
import json
import math

import numpy as np
import pytest

from bot.heuristic import HeuristicParams
from game import Action
from sim.match import play_match
from sim.opponents import HOLDOUT, TRAIN, ZOO
from sim.tune import (MATCH_DEALS, Score, _seed_groups, default_space, expected_points,
                      main, objective, params_diff, score_matches, search, train_opponents,
                      tune)

DEFAULT_SPACE = default_space()

SMALL = {"open_min": [0.40, 0.44, 0.48], "call_margin": [0.0, 0.02, 0.04]}


def _score(obj):
    return Score(obj, obj, obj, 0.0, {}, 0, 0, 0.0)


def _synthetic(target, calls=None):
    """An evaluator whose objective peaks at `target` (a dict of field values)."""
    def evaluate(cands, batch):
        if calls is not None:
            calls.append(list(cands))
        return [_score(-sum(abs(getattr(p, k) - v) for k, v in target.items()))
                for p in cands]
    return evaluate


def test_expected_points_follows_the_verdict_rule():
    sd = 10.0
    se = sd / math.sqrt(MATCH_DEALS)
    assert expected_points(0.0, sd) == pytest.approx(0.5)
    assert expected_points(5.0, sd) == pytest.approx(1.0)
    assert expected_points(-5.0, sd) == pytest.approx(0.0)
    # an edge of exactly Z standard errors wins half the time and almost never loses
    assert expected_points(1.96 * se, sd) == pytest.approx(0.75, abs=1e-4)
    for m in (0.1, 0.4, 0.9):
        assert expected_points(m, sd) + expected_points(-m, sd) == pytest.approx(1.0)
        assert expected_points(m, sd) < expected_points(m + 0.1, sd)       # smooth, rising
    assert (expected_points(0.2, 0.0), expected_points(0.0, 0.0),
            expected_points(-0.2, 0.0)) == (1.0, 0.5, 0.0)


def test_objective_is_points_plus_a_hundredth_of_the_bb():
    assert objective(0.9, 2.0) == pytest.approx(0.92)
    assert objective(1.0, -3.0) == pytest.approx(0.97)


def test_score_pools_the_seeds_and_a_forfeit_can_never_win():
    cs = ZOO["calling_station"]
    ms = [play_match(ZOO["nit"], cs, s, 30) for s in ("u1", "u2")]
    sc = score_matches({"cs": ms})
    d = np.concatenate([(np.asarray(m.halves[0].nets_a) + m.halves[1].nets_a) / 2 for m in ms])
    mu, sd, e = sc.per_opponent["cs"]
    assert len(d) == 60 and mu == pytest.approx(d.mean()) and sd == pytest.approx(d.std(ddof=1))
    assert e == pytest.approx(expected_points(mu, sd)) and sc.points == pytest.approx(e)
    assert sc.objective == pytest.approx(objective(e, mu))
    assert sc.match_points == pytest.approx(np.mean([m.points_a for m in ms]))

    class Crash:
        def new_half(self, config, memory):
            pass

        def choose_action(self, obs, memory):
            raise RuntimeError("boom")

        def on_hand_end(self, result, memory):
            pass

    bad = score_matches({"cs": [play_match(Crash, cs, "u1", 10)]})
    assert bad.forfeits > 0 and bad.errors >= 5 and bad.objective == -1.0


def test_search_climbs_to_the_best_of_a_synthetic_objective():
    target = {"open_min": 0.48, "call_margin": 0.0}
    best, steps = search(SMALL, HeuristicParams(), 30, _synthetic(target),
                         np.random.default_rng(0))
    assert (best.open_min, best.call_margin) == (0.48, 0.0)
    assert len(steps) <= 30 and steps[0].params == HeuristicParams()
    assert steps[0].phase == "random" and steps[-1].phase.startswith("coord:")


def test_search_spends_at_most_the_budget_and_rechecks_the_incumbent():
    calls = []
    target = {"open_min": 0.48, "bluff_freq": 0.05}
    best, steps = search(DEFAULT_SPACE, HeuristicParams(), 17, _synthetic(target, calls),
                         np.random.default_rng(1))
    assert sum(len(c) for c in calls) == len(steps) <= 17

    def value(p):
        return -sum(abs(getattr(p, k) - v) for k, v in target.items())

    incumbent = max(calls[0], key=value)    # the best of the random batch
    for c in calls[1:]:                     # every descent batch starts with the incumbent
        assert c[0] == incumbent and len(c) <= 3   # ... and at most two neighbours
        challenger = max(c[1:], key=value)
        if value(challenger) > value(c[0]):
            incumbent = challenger
    assert best == incumbent
    assert search(SMALL, HeuristicParams(), 0, _synthetic({}), np.random.default_rng(0)) \
        == (HeuristicParams(), [])


def test_search_is_reproducible_for_an_rng_seed():
    runs = [search(DEFAULT_SPACE, HeuristicParams(), 12, _synthetic({"open_min": 0.4}),
                   np.random.default_rng(7))[1] for _ in range(2)]
    assert [s.params for s in runs[0]] == [s.params for s in runs[1]]


def test_min_gain_keeps_the_incumbent_against_noise():
    def tiny(cands, batch):                 # every move "gains" 0.001
        return [_score(0.001 * sum(p != HeuristicParams() for _ in [0])) for p in cands]
    best, _ = search(SMALL, HeuristicParams(), 20, tiny, np.random.default_rng(0),
                     random_frac=0.0, min_gain=0.01)
    assert best == HeuristicParams()
    best, _ = search(SMALL, HeuristicParams(), 20, tiny, np.random.default_rng(0),
                     random_frac=0.0, min_gain=0.0)
    assert best != HeuristicParams()


@pytest.mark.parametrize("space", [{"open_min": [0.4, float("nan")]},
                                   {"value_raise": [(0.7, 0.75)]},
                                   {"not_a_param": [1.0]},
                                   {"bluff_freq": []}])
def test_search_rejects_a_bad_space_before_any_match(space):
    calls = []
    with pytest.raises(ValueError):
        search(space, HeuristicParams(), 5, _synthetic({}, calls), np.random.default_rng(0))
    assert calls == []


@pytest.mark.parametrize("base", [HeuristicParams(), HeuristicParams(
    open_min=0.48, call_margin=0.0, value_raise=(0.74, 0.79, 0.84), bluff_freq=0.05)])
def test_default_space_is_valid_and_centred_on_the_base(base):
    space = default_space(base)
    assert set(space) == {f.name for f in dataclasses.fields(HeuristicParams)}
    for name, values in space.items():
        assert len(values) == 3 and values[1] == getattr(base, name), name
        for v in values:
            HeuristicParams(**{name: v})
    assert default_space(base)["open_min"] == [round(base.open_min + d, 4) for d in (-0.04, 0, 0.04)]
    # Task 12's run: centred on the T10 v1 values
    t10 = default_space(HeuristicParams(open_min=0.44))
    assert t10["open_min"] == [0.40, 0.44, 0.48]


def test_seed_groups_rotate_through_the_pool():
    g = _seed_groups(["a", "b", "c", "d", "e"], 2)
    assert [g(k) for k in range(4)] == [("a", "b"), ("c", "d"), ("e", "a"), ("b", "c")]
    assert _seed_groups(["a", "b"], None)(3) == ("a", "b")
    assert _seed_groups(["a", "b"], 5)(0) == ("a", "b")


def test_tune_runs_real_matches_on_common_seeds():
    seen = []
    # budget 6: a random batch of 4 (the base + 3), then one descent batch of 2
    best = tune(SMALL, ["calling_station"], ["u1", "u2", "u3"], deals=8, budget=6,
                workers=1, time_scale=2.5, batch_seeds=2, on_step=seen.append)
    assert isinstance(best, HeuristicParams) and best in [s.params for s in seen]
    assert [s.batch for s in seen] == [0, 0, 0, 0, 1, 1]
    assert seen[0].seeds == ("u1", "u2") and seen[-1].seeds == ("u3", "u1")
    assert all(s.score.errors == 0 and s.score.forfeits == 0 for s in seen)
    assert all(set(s.score.per_opponent) == {"calling_station"} for s in seen)


def test_cli_logs_every_evaluation(tmp_path, capsys):
    out = tmp_path / "t.jsonl"
    assert main(["--budget", "2", "--seeds", "1", "--deals", "4", "--workers", "1",
                 "--opponents", "nit,calling_station", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["params"] == {} and rows[0]["seeds"] == ["t1"]
    assert set(rows[0]["per_opponent"]) == {"nit", "calling_station"}
    assert "best (vs defaults)" in capsys.readouterr().out


# ---- the tuner sees TRAIN opponents only, by construction (plan: never tune on HOLDOUT)

def test_train_opponents_is_the_train_split_or_train_members():
    assert set(train_opponents("train")) == TRAIN
    assert all(train_opponents("train")[n] is ZOO[n] for n in TRAIN)
    assert list(train_opponents(["nit", "calling_station"])) == ["nit", "calling_station"]
    assert train_opponents(("nit",))["nit"] is ZOO["nit"]      # factories come from ZOO


@pytest.mark.parametrize("bad", ["holdout", "all", "field", "zoo", "", [], ["llm_opp_model"],
                                 ["nit", "threshold_cat"], ["agentic_gto"], ["no_such_bot"],
                                 {"nit": ZOO["nit"]}, {"llm_opp_model": ZOO["llm_opp_model"]},
                                 "nit", None, 3])
def test_the_tuner_refuses_anything_but_train(bad):
    assert HOLDOUT and not HOLDOUT & TRAIN
    with pytest.raises(ValueError):
        train_opponents(bad)
    calls = []
    with pytest.raises(ValueError):
        tune(SMALL, bad, ["u1"], deals=4, budget=2, workers=1,
             on_step=lambda st: calls.append(st))
    assert calls == []                              # refused before any match


@pytest.mark.parametrize("bad", ["holdout", "all", "llm_opp_model", "nit,threshold_cat"])
def test_the_cli_refuses_anything_but_train(bad, tmp_path):
    with pytest.raises(SystemExit):
        main(["--budget", "1", "--seeds", "1", "--deals", "2", "--workers", "1",
              "--opponents", bad, "--out", str(tmp_path / "t.jsonl")])
    assert not (tmp_path / "t.jsonl").exists() or (tmp_path / "t.jsonl").read_text() == ""


def test_params_diff_lists_only_changed_fields():
    assert params_diff(HeuristicParams()) == {}
    p = HeuristicParams(open_min=0.4, value_raise=(0.7, 0.75, 0.81))
    assert params_diff(p) == {"open_min": 0.4, "value_raise": (0.7, 0.75, 0.81)}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_search_never_exceeds_any_budget(seed):
    for budget in range(1, 40):
        calls = []
        search(DEFAULT_SPACE, HeuristicParams(), budget,
               _synthetic({"open_min": 0.48, "range_weight": 0.8}, calls),
               np.random.default_rng(seed))
        assert sum(len(c) for c in calls) <= budget
