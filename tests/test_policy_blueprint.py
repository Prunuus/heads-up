"""The blueprint layer of bot/policy.py (Task 20): node_of -> bucket -> Blueprint.act,
then the exploit layer; the heuristic answers whenever the blueprint cannot."""

import dataclasses
import functools
import pickle

import numpy as np
import pytest

import game
from bot import abstraction, exploit, policy, strategy
from bot.tree import get_tree, node_of
from game import Action, AgentConfig, Street
from tests.test_policy import _fresh_bot, _observations

policy.warmup()

TREE = get_tree()


class FakeBlueprint:
    """Duck-typed bot.strategy.Blueprint: records every act() call and answers with
    `pick(legal)` (default: the highest legal action)."""

    def __init__(self, meta=None, pick=None):
        self.meta = dict(policy.DEFAULT_META) if meta is None else meta
        self.calls = []
        self.pick = pick if pick is not None else (lambda legal: max(legal))

    def act(self, node, bucket, legal, rng, purify=0.1):
        self.calls.append((node, bucket, tuple(legal), purify))
        return self.pick(tuple(legal))


def _bp_bot(fake=None, config=None, **kw):
    bot = policy.PolicyBot(**kw)
    bot.blueprint = fake if fake is not None else FakeBlueprint()
    memory = {}
    bot.new_half(config or AgentConfig(), memory)
    return bot, memory


def test_blueprint_layer_used_when_config_ok():
    bot, memory = _bp_bot(use_exploits=False, purify=0.05)
    assert memory["config_ok"] is True
    obs_all = _observations("bp-used", 30)
    assert {o.street for o in obs_all} == set(Street)
    for obs in obs_all:
        a = bot.choose_action(obs, memory)
        node, b, legal, purify = bot.blueprint.calls[-1]
        assert node == node_of(obs, TREE)
        assert b == abstraction.bucket(obs.street, obs.hole_cards, obs.board)
        assert legal == obs.legal_actions and purify == 0.05
        assert a == max(obs.legal_actions)            # the blueprint's answer is played
    assert len(bot.blueprint.calls) == len(obs_all)
    assert memory["layers"]["blueprint"] == len(obs_all)
    assert memory["layers"]["heuristic"] == 0
    assert sum(memory["faults"].values()) == 0


def test_blueprint_path_skips_the_heuristic(monkeypatch):
    # T11 M3, the positive path. Postflop the heuristic never runs. Preflop it runs only as
    # the CPU-factor sample (T11 M2: the online sample does not depend on which layer
    # answers), and its answer is discarded.
    calls = []
    real = policy.heuristic_action

    def spy(obs, *a, **k):
        calls.append(obs.street)
        return real(obs, *a, **k)

    monkeypatch.setattr(policy, "heuristic_action", spy)
    bot, memory = _bp_bot(use_exploits=False)
    obs_all = _observations("bp-skip", 30)
    for obs in obs_all:
        assert bot.choose_action(obs, memory) == max(obs.legal_actions)
    n_pre = sum(o.street == Street.PREFLOP for o in obs_all)
    assert calls == [Street.PREFLOP] * n_pre
    assert memory["cpu"].n == min(n_pre, policy.WINDOW)
    assert sum(memory["faults"].values()) == 0


@pytest.mark.parametrize("change", [{"raise_cap": 3}, {"big_blind": 4, "small_blind": 2},
                                    {"starting_stack": 100}])
def test_config_mismatch_never_consults_the_blueprint(change):
    # Review Focus 1: a retuned raise cap still maps most nodes, so config_ok must gate.
    fake = FakeBlueprint(pick=lambda legal: pytest.fail("blueprint consulted"))
    bot, memory = _bp_bot(fake, AgentConfig(**change))
    assert memory["config_ok"] is False
    for obs in _observations("bp-cfg", 15):
        assert bot.choose_action(obs, memory) in obs.legal_actions
    assert fake.calls == [] and memory["layers"]["blueprint"] == 0
    assert sum(memory["faults"].values()) == 0


def test_a_blueprint_meta_mismatch_disables_it():
    fake = FakeBlueprint(meta=dict(policy.DEFAULT_META, cap=3))
    bot, memory = _bp_bot(fake)                     # the default game, another blueprint
    assert memory["config_ok"] is False
    obs = _observations("bp-meta", 1)[0]
    assert bot.choose_action(obs, memory) in obs.legal_actions and fake.calls == []


def test_node_miss_falls_back_to_the_heuristic():
    fake = FakeBlueprint(pick=lambda legal: pytest.fail("blueprint consulted"))
    bot, memory = _bp_bot(fake)
    obs_all = _observations("bp-miss", 15)
    for obs in obs_all:                             # chips that the tree cannot show
        odd = dataclasses.replace(obs, my_stack=obs.my_stack - 1)
        assert node_of(odd, TREE) is None
        assert bot.choose_action(odd, memory) in odd.legal_actions
    assert memory["layers"]["node_miss"] == len(obs_all)
    assert memory["layers"]["heuristic"] == len(obs_all)
    assert sum(memory["faults"].values()) == 0


@pytest.mark.parametrize("bad", ["raise", "illegal", "not_an_action"])
def test_a_failing_blueprint_counts_a_fault_and_the_heuristic_answers(bad):
    def pick(legal):
        if bad == "raise":
            raise IndexError("no infoset")
        if bad == "illegal":
            return next((a for a in Action if a not in legal), "x")
        return 2

    bot, memory = _bp_bot(FakeBlueprint(pick=pick))
    obs_all = [o for o in _observations("bp-bad", 15) if len(o.legal_actions) < 4]
    for obs in obs_all:
        assert bot.choose_action(obs, memory) in obs.legal_actions
    assert memory["faults"]["blueprint"] == len(obs_all)
    assert memory["faults"]["heuristic"] == 0
    assert memory["layers"]["heuristic"] == len(obs_all)


def test_a_failing_bucket_counts_a_fault(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("data/buckets.npz")

    monkeypatch.setattr(abstraction, "bucket", boom)
    bot, memory = _bp_bot()
    obs_all = _observations("bp-bkt", 10)
    for obs in obs_all:
        assert bot.choose_action(obs, memory) in obs.legal_actions
    assert memory["faults"]["blueprint"] == len(obs_all) and bot.blueprint.calls == []


def test_the_bucket_is_computed_once_per_street(monkeypatch):
    seen = []
    real = abstraction.bucket

    def spy(street, hole, board, *a, **k):
        seen.append((street, tuple(hole), tuple(board)))
        return real(street, hole, board, *a, **k)

    monkeypatch.setattr(abstraction, "bucket", spy)
    bot, memory = _bp_bot(use_exploits=False)
    obs = next(o for o in _observations("bp-cache", 30)
               if o.street == Street.FLOP and o.to_call == 0)
    for _ in range(3):                              # e.g. LBR's re-simulations, or a re-raise
        bot.choose_action(obs, memory)
    assert len(seen) == 1
    other = dataclasses.replace(obs, hole_cards=tuple(sorted(
        c for c in range(36) if c not in obs.board)[:5]))
    bot.choose_action(other, memory)                # another hole: never the cached bucket
    assert len(seen) == 2 and len(bot.blueprint.calls) == 4


def test_blueprint_streets_limits_the_layer():
    bot, memory = _bp_bot(use_exploits=False,
                          blueprint_streets=(Street.FLOP, Street.TURN, Street.RIVER))
    obs_all = _observations("bp-streets", 20)
    for obs in obs_all:
        bot.choose_action(obs, memory)
    n_pre = sum(o.street == Street.PREFLOP for o in obs_all)
    assert memory["layers"]["heuristic"] == n_pre
    assert memory["layers"]["blueprint"] == len(obs_all) - n_pre
    assert all(TREE.street[c[0]] != Street.PREFLOP for c in bot.blueprint.calls)


# ------------------------------------------------------------ exploits on the blueprint

def test_check_folder_exploit_applies_to_a_blueprint_base(monkeypatch):
    monkeypatch.setattr(exploit, "active_mode", lambda opp, strict_maniac=False: "check_folder")
    bot, memory = _bp_bot(FakeBlueprint(pick=lambda legal: Action.CHECK_CALL))
    for obs in _observations("bp-cf", 15):
        want = Action.RAISE_HALF if Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL
        assert bot.choose_action(obs, memory) == want
    assert memory["exploits"]["check_folder"] > 0


@pytest.mark.parametrize("mode", ["maniac", "station", None])
def test_re_run_modes_get_features_only_when_active(monkeypatch, mode):
    # With a blueprint base the maniac/station re-runs need Features: they are computed
    # only while such a mode is active (with no mode the blueprint path stays cheap).
    monkeypatch.setattr(exploit, "active_mode", lambda opp, strict_maniac=False: mode)
    got = []
    real = policy.apply_exploits

    def spy(base, obs, feats, *a, **k):
        got.append((obs.street, feats))
        return real(base, obs, feats, *a, **k)

    monkeypatch.setattr(policy, "apply_exploits", spy)
    bot, memory = _bp_bot(FakeBlueprint(pick=lambda legal: Action.CHECK_CALL))
    obs_all = _observations("bp-mode", 15)
    for obs in obs_all:
        assert bot.choose_action(obs, memory) in obs.legal_actions
    post = [f for s, f in got if s != Street.PREFLOP]
    assert len(got) == len(obs_all) and post
    if mode is None:
        assert all(f is None for f in post)
    else:
        assert all(f is not None and f.street == s for s, f in got)
        assert memory["exploits"][mode] == len(obs_all)
    assert sum(memory["faults"].values()) == 0


def test_exploit_features_off_keeps_the_blueprint_path_feature_free(monkeypatch):
    monkeypatch.setattr(exploit, "active_mode", lambda opp, strict_maniac=False: "maniac")
    got = []
    monkeypatch.setattr(policy, "apply_exploits",
                        lambda base, obs, feats, *a, **k: got.append((obs.street, feats)) or base)
    bot, memory = _bp_bot(exploit_features=False)
    for obs in _observations("bp-nofeat", 10):
        bot.choose_action(obs, memory)
    assert all(f is None for s, f in got if s != Street.PREFLOP)


# ------------------------------------------------------------- the real Blueprint class

@pytest.fixture(scope="module")
def max_action_blueprint():
    """A real bot.strategy.Blueprint with the runtime's bucket counts whose every infoset
    puts all its mass on the highest legal action."""
    counts = np.array([abstraction.NUM_BUCKETS[s] for s in Street], dtype=np.int64)
    per_node = counts[TREE.street.astype(np.int64)]
    offset = np.concatenate([[0], np.cumsum(per_node)]).astype(np.int64)
    legal = np.repeat(TREE.child != -1, per_node, axis=0)
    top = 3 - np.argmax(legal[:, ::-1], axis=1)
    table = np.zeros((int(offset[-1]), 4), dtype=np.float16)
    table[np.arange(len(table)), top] = 1.0
    meta = dict(policy.DEFAULT_META, buckets=tuple(int(c) for c in counts), iterations=0)
    return strategy.Blueprint(meta, table, offset, TREE.child)


def test_a_real_blueprint_object_drives_the_bot(max_action_blueprint):
    bot, memory = _bp_bot(max_action_blueprint, use_exploits=False)
    obs_all = _observations("bp-real", 30)
    for obs in obs_all:
        assert bot.choose_action(obs, memory) == max(obs.legal_actions)
    assert memory["layers"]["blueprint"] == len(obs_all)
    assert sum(memory["faults"].values()) == 0


# ---------------------------------------------------------------- warmup and the file

@pytest.mark.parametrize("damage", ["missing", "corrupt", "stale"])
def test_a_missing_or_bad_blueprint_file_means_heuristic_only(tmp_path, monkeypatch, damage):
    path = tmp_path / "blueprint.npz"
    if damage == "corrupt":
        path.write_bytes(b"PK\x03\x04 not a blueprint" * 100)
    elif damage == "stale":                         # readable, but for another tree
        np.savez(path, format=np.int64(strategy.FORMAT), probs=np.zeros((4, 4), np.float16),
                 stack=np.int64(80), sb=np.int64(1), bb=np.int64(2), cap=np.int64(4),
                 iterations=np.int64(1), seed=np.int64(0), linear=np.bool_(True),
                 n_dec=np.int64(1), n_term=np.int64(1), tree_child_sha256=np.array("0"),
                 buckets_sha256=np.array("0"), features_sha256=np.array("0"),
                 buckets=np.array([1, 1, 1, 1]))
    monkeypatch.setattr(policy, "BLUEPRINT_PATH", path)
    monkeypatch.setattr(policy, "WARMUP", None)
    policy.warmup()                                 # never raises
    w = policy.WARMUP
    assert w.blueprint is None
    assert ("blueprint" in w.failed) == (damage != "missing")
    bot, memory = _fresh_bot(use_blueprint=True)
    assert bot.blueprint is None
    for obs in _observations("bp-file", 15):
        assert bot.choose_action(obs, memory) in obs.legal_actions
    assert memory["layers"]["blueprint"] == 0 and sum(memory["faults"].values()) == 0


def test_warmup_drops_a_blueprint_that_fails_its_self_test(monkeypatch):
    class Broken(FakeBlueprint):
        def act(self, *a, **k):
            raise RuntimeError("broken table")

    monkeypatch.setattr(policy, "_load_blueprint", lambda: Broken())
    monkeypatch.setattr(policy, "WARMUP", None)
    policy.warmup()
    assert policy.WARMUP.blueprint is None and "blueprint_selftest" in policy.WARMUP.failed


def test_warmup_keeps_a_working_blueprint(monkeypatch, max_action_blueprint):
    monkeypatch.setattr(policy, "_load_blueprint", lambda: max_action_blueprint)
    monkeypatch.setattr(policy, "WARMUP", None)
    policy.warmup()
    assert policy.WARMUP.blueprint is max_action_blueprint and policy.WARMUP.failed == ()
    bot, memory = _fresh_bot(use_blueprint=True)
    assert bot.blueprint is max_action_blueprint and memory["config_ok"] is True
    obs = _observations("bp-warm", 1)[0]
    bot.choose_action(obs, memory)
    assert memory["layers"]["blueprint"] == 1


def test_blueprint_bot_factories_are_picklable():
    fac = pickle.loads(pickle.dumps(functools.partial(
        policy.PolicyBot, purify=0.2, blueprint_streets=(Street.RIVER,), exploit_features=False)))
    bot = fac()
    assert bot.purify == 0.2 and bot.blueprint_streets == frozenset({Street.RIVER})
    assert bot.exploit_features is False


# ------------------------------------------------------------ the auto base (Task 20b)

class _Opp:
    """Duck-typed OpponentModel for policy.select_base: only raise_rate(street)."""

    def __init__(self, rate=0.0, n=0):
        self.rate, self.n = rate, n

    def raise_rate(self, street):
        return (self.rate, self.n) if int(street) == 0 else (0.5, 999)


def test_select_base_defaults_to_the_blueprint_without_evidence():
    assert policy.select_base(None, "blueprint") == "blueprint"
    assert policy.select_base(object(), "blueprint") == "blueprint"       # no raise_rate
    assert policy.select_base(_Opp(0.0, policy.AUTO_MIN_N - 1), "blueprint") == "blueprint"
    assert policy.select_base(_Opp(float("nan"), 500), "blueprint") == "blueprint"


def test_select_base_goes_heuristic_vs_a_passive_opponent():
    assert policy.select_base(_Opp(0.0, policy.AUTO_MIN_N), "blueprint") == "heuristic"
    assert policy.select_base(_Opp(policy.AUTO_ENTER, 300), "blueprint") == "heuristic"
    assert policy.select_base(_Opp(policy.AUTO_ENTER + 0.01, 300), "blueprint") == "blueprint"
    assert policy.select_base(_Opp(0.4, 300), "blueprint") == "blueprint"


def test_select_base_has_hysteresis():
    mid = (policy.AUTO_ENTER + policy.AUTO_EXIT) / 2
    assert policy.select_base(_Opp(mid, 300), "heuristic") == "heuristic"   # stays
    assert policy.select_base(_Opp(mid, 300), "blueprint") == "blueprint"   # stays
    assert policy.select_base(_Opp(policy.AUTO_EXIT, 300), "heuristic") == "blueprint"
    assert policy.select_base(_Opp(0.0, 10), "heuristic") == "heuristic"    # no n needed to stay


def test_select_base_never_raises():
    class Bad:
        def raise_rate(self, street):
            raise RuntimeError("boom")

    assert policy.select_base(Bad(), "heuristic") == "blueprint"
    assert policy.select_base(_Opp(0.0, 500), "garbage") == "heuristic"


def _passive_model(hands=40):
    from bot.opponent import OpponentModel
    opp = OpponentModel()
    opp._raise_n[0], opp._raise_k[0] = 100, 2
    return opp


def test_auto_bot_switches_the_base_per_half_and_blueprint_is_the_default():
    obs = _observations("auto-base", 12)
    bot, memory = _bp_bot(use_exploits=False, base="auto")
    assert memory["base"] == "blueprint"
    for o in obs[:4]:
        assert bot.choose_action(o, memory) == max(o.legal_actions)        # blueprint
    memory["opp"] = _passive_model()
    for o in obs[4:]:
        bot.choose_action(o, memory)
    assert memory["base"] == "heuristic"
    assert memory["layers"]["blueprint"] == 4
    assert memory["layers"]["heuristic"] >= len(obs) - 4
    assert sum(memory["faults"].values()) == 0
    bot2, memory2 = _bp_bot(use_exploits=False, base="auto")       # a fresh half starts over
    assert memory2["base"] == "blueprint"


def test_fixed_bases_ignore_the_opponent():
    obs = _observations("fixed-base", 10)
    bot, memory = _bp_bot(use_exploits=False, base="blueprint")
    memory["opp"] = _passive_model()
    for o in obs:
        assert bot.choose_action(o, memory) == max(o.legal_actions)
    assert memory["layers"]["heuristic"] == 0
    bot, memory = _bp_bot(use_exploits=False, base="heuristic")
    for o in obs:
        bot.choose_action(o, memory)
    assert memory["layers"]["blueprint"] == 0
    with pytest.raises(ValueError):
        policy.PolicyBot(base="nonsense")


def test_a_failing_selector_keeps_the_blueprint(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("selector")

    monkeypatch.setattr(policy, "select_base", boom)
    bot, memory = _bp_bot(use_exploits=False, base="auto")
    for o in _observations("auto-fault", 8):
        assert bot.choose_action(o, memory) == max(o.legal_actions)
    assert memory["layers"]["heuristic"] == 0
    assert sum(memory["faults"].values()) == 0


def test_the_default_base_is_auto_and_starts_on_the_blueprint():
    assert policy.DEFAULT_BASE == "auto" and policy.PolicyBot().base == "auto"
    bot, memory = _bp_bot(use_exploits=False)
    assert memory["base"] == "blueprint"
    for o in _observations("default-auto", 6):
        assert bot.choose_action(o, memory) == max(o.legal_actions)
