"""Exact betting tree: every betting sequence the rules allow, enumerated with game.Game.

Betting never depends on the cards, so one exhaustive DFS of game.Game (its default
constants, any deal) reaches every decision node (F1: 15,952 = preflop 62, flop 810,
turn 3,678, river 11,402) and every terminal (25,538). The CFR trainer indexes its
tables by decision node. At runtime the bot maps an observation's action history to
its node with node_of().

Ids: decision nodes are numbered in DFS pre-order from the root (0), children in Action
order, so every build gives the same ids. Terminals are numbered in the order the DFS
reaches them.

Arrays are relative to the button, not to a seat:
- actor_is_button[node]: True when the button acts at the node.
- street_contrib[node]: (button, bb) chips put in on the node's street so far.
- term_contrib[tid]: (button, bb) chips put in over the whole hand.
- term_fold_by_button[tid]: 1 = the button folded, 0 = the BB folded, -1 = showdown
  (all-in run-outs included).
- child[node, action]: -1 = illegal, >= 0 = decision id, <= -2 = terminal -2 - code.

get_tree() builds once per process and caches. Every array is non-writeable, `legal`
is a tuple and `key_to_node` a read-only mapping, so the cache is read-only data
(ruling R22). build_tree() has no side effects.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np

from game import Action, Deal, Game, Observation

# Any deal works: cards only decide who wins a showdown, never what can be bet.
_DEAL = Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))
_INT16_MAX = int(np.iinfo(np.int16).max)


@dataclass(frozen=True, eq=False)
class BettingTree:
    """Decision nodes and terminals of the betting game (conventions: module docstring)."""

    n_dec: int
    n_term: int
    starting_stack: int                      # each seat's stack before the blinds
    street: np.ndarray                       # int8 [n_dec]
    actor_is_button: np.ndarray              # bool [n_dec]
    legal: tuple[tuple[Action, ...], ...]    # [n_dec], in Game.legal_actions() order
    child: np.ndarray                        # int32 [n_dec, 4]
    pot_at: np.ndarray                       # int16 [n_dec]: pot at the node
    to_call_at: np.ndarray                   # int16 [n_dec]: what the actor must add to call
    street_contrib: np.ndarray               # int16 [n_dec, 2]: (button, bb) street bets
    term_fold_by_button: np.ndarray          # int8 [n_term]
    term_contrib: np.ndarray                 # int16 [n_term, 2]: (button, bb) hand totals
    key_to_node: Mapping[tuple[int, ...], int]   # action codes from the root -> node

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        if not isinstance(self.key_to_node, MappingProxyType):
            object.__setattr__(self, "key_to_node", MappingProxyType(dict(self.key_to_node)))

    def __reduce__(self):
        """Pickle (and copy) through plain fields: a mappingproxy cannot be pickled."""
        state = {f.name: getattr(self, f.name) for f in fields(self)}
        state["key_to_node"] = dict(self.key_to_node)
        return _from_fields, (state,)


def _from_fields(state: dict) -> BettingTree:
    return BettingTree(**state)


def _chips(values: list) -> np.ndarray:
    """Chip amounts as int16. Raises ValueError rather than wrap a value that won't fit."""
    arr = np.array(values, dtype=np.int64)
    if arr.size and not (0 <= arr.min() and arr.max() <= _INT16_MAX):
        raise ValueError(f"chip amounts {arr.min()}..{arr.max()} do not fit int16")
    return arr.astype(np.int16)


def build_tree() -> BettingTree:
    """Enumerate every betting sequence by DFS over game.Game. Returns a new tree."""
    street: list[int] = []
    actor_is_button: list[bool] = []
    legal: list[tuple[Action, ...]] = []
    child: list[list[int]] = []
    pot: list[int] = []
    to_call: list[int] = []
    bets: list[tuple[int, int]] = []
    fold_by_button: list[int] = []
    contrib: list[tuple[int, int]] = []
    key_to_node: dict[tuple[int, ...], int] = {}
    shared: dict[tuple[Action, ...], tuple[Action, ...]] = {}   # one object per legal set

    root = Game(_DEAL, button_seat=0)
    btn = root.button_seat
    # Iterative pre-order DFS (no recursive closure, so no reference cycle keeps these
    # lists alive after the build). Entry: (game, key, parent's child row, action into
    # that row); the root's row is a throwaway.
    stack: list = [(root, (), [0], 0)]
    while stack:
        g, key, parent_row, via = stack.pop()
        node = parent_row[via] = len(street)
        key_to_node[key] = node
        seat, acts = g.to_act, g.legal_actions()
        street.append(int(g.street))
        actor_is_button.append(seat == btn)
        legal.append(shared.setdefault(acts, acts))
        pot.append(g.pot)
        to_call.append(g._to_call(seat))       # the engine's own rule, not a copy of it
        bets.append((g.bets[btn], g.bets[1 - btn]))
        row = [-1, -1, -1, -1]
        child.append(row)
        later = []
        for a in acts:
            nxt = g.clone()
            nxt.apply(a)
            if nxt.is_terminal:
                res = nxt.result()
                row[a] = -2 - len(fold_by_button)
                # A fold's winner is the seat that did not fold.
                fold_by_button.append(-1 if res.went_to_showdown else int(res.winner_seat != btn))
                contrib.append((nxt.contributed[btn], nxt.contributed[1 - btn]))
            else:
                later.append((nxt, key + (int(a),), row, a))
        stack.extend(reversed(later))          # lowest action on top: ids in Action order

    return BettingTree(
        n_dec=len(street),
        n_term=len(fold_by_button),
        starting_stack=root.stacks[btn] + root.contributed[btn],
        street=np.array(street, dtype=np.int8),
        actor_is_button=np.array(actor_is_button, dtype=bool),
        legal=tuple(legal),
        child=np.array(child, dtype=np.int32),
        pot_at=_chips(pot),
        to_call_at=_chips(to_call),
        street_contrib=_chips(bets),
        term_fold_by_button=np.array(fold_by_button, dtype=np.int8),
        term_contrib=_chips(contrib),
        key_to_node=MappingProxyType(key_to_node),    # sole reference: no copy needed
    )


@functools.cache
def get_tree() -> BettingTree:
    """The process-wide tree, built on first call (call it in warmup, not mid-hand).

    A failed build raises and is not cached, so the next call would build again: the
    caller should call this once and keep the result.
    """
    return build_tree()


def _codes(values, numpy_ints: bool = False) -> tuple[int, ...] | None:
    """Action codes as plain ints, or None if any value is not an int.

    An Action from any copy of game.py and a plain int code both count. A bool, float,
    str or numpy integer does not, even though int() would turn it into a code. With
    numpy_ints, a numpy integer counts too (a numpy bool is not a numpy integer).
    """
    codes = []
    for x in values:
        if x.__class__ is bool or not (isinstance(x, int) or numpy_ints and isinstance(x, np.integer)):
            return None
        codes.append(int(x))
    return tuple(codes)


def node_from_actions(actions: tuple[int, ...], tree: BettingTree) -> int | None:
    """Decision node reached by these action codes from the root (() is the root).

    The codes may be ints, Actions or numpy integers (trainer input, e.g. a row of a numpy
    array), never bools. None for a sequence outside the tree (illegal, or ending in a
    terminal) or malformed input. Never raises.
    """
    try:
        key = _codes(actions, numpy_ints=True)
        return None if key is None else tree.key_to_node.get(key)
    except Exception:
        return None


def node_of(obs: Observation, tree: BettingTree) -> int | None:
    """Decision node of the actor's observation: an O(1) lookup of its action history.

    Returns the node only if the observation also shows that node's state: street,
    actor seat, legal actions (same order), pot, amount to call, and the tree's chips in
    play (both stacks plus the pot). Otherwise None: the history is outside the tree or
    malformed, or the observation is corrupt or comes from other rules. A retuned stack
    misses at every node. A retuned blind or raise cap misses wherever it changes the
    pot, the amount to call or the legal actions, so callers must still compare their
    config (Review Focus 1). On None the caller falls back to the heuristic. Never raises.
    """
    try:
        key = _codes([a.action for a in obs.action_history])
        node = None if key is None else tree.key_to_node.get(key)
        if node is not None and (
                obs.legal_actions == tree.legal[node]
                and obs.pot == tree.pot_at[node]
                and obs.to_call == tree.to_call_at[node]
                and obs.is_button == tree.actor_is_button[node]
                and obs.street == tree.street[node]
                and obs.my_stack + obs.opp_stack + obs.pot == 2 * tree.starting_stack):
            return node
    except Exception:
        pass
    return None
