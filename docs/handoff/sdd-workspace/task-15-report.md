# Task 15 report: exact betting tree

Commit `51ee0ae` `feat(bot): exact betting tree with history lookup`, on branch `kevin-cfr` in the worktree `/Users/kchen/Desktop/code/FALL26/heads-up-bundle-cfr`. Base `df265bd`. Not pushed.

Files: `bot/tree.py` (new, 212 lines) and `tests/test_tree.py` (new, 417 lines, 37 tests). No other file was touched.

## Implementation (`bot/tree.py`; imports only stdlib, numpy and `game`)

**`build_tree() -> BettingTree`** (pure: returns a new tree on every call)
- Runs an iterative pre-order DFS over `game.Game(_DEAL, button_seat=0)` using `clone()` + `apply()` on each `legal_actions()` entry.
  - The deal is a fixed literal; cards only decide showdowns.
  - Every value comes from the engine: `to_act`, `button_seat`, `street`, `legal_actions()`, `pot`, `bets`, `_to_call(seat)`, `is_terminal`, `result()` and `contributed`. No betting rule is re-implemented.
  - Note that `_to_call` is the engine's own private method.
- Ids:
  - Decision ids are DFS pre-order from root 0, children in Action order, so every build is deterministic.
  - Terminal ids follow discovery order.
- Result: 15,952 decision nodes (62 / 810 / 3,678 / 11,402 by street), 25,538 terminals, maximum history length 13. This matches F1 exactly.

**`BettingTree`** (`@dataclass(frozen=True, eq=False)`)

| field | type | meaning |
|---|---|---|
| `n_dec`, `n_term` | int | 15952, 25538 |
| `starting_stack` | int | 80. Derived from the root Game; `node_of` uses it for the chips-in-play check |
| `street` | int8 [n_dec] | |
| `actor_is_button` | bool [n_dec] | |
| `legal` | **tuple** of `tuple[Action, ...]` | `Game.legal_actions()` order; only 3 distinct shared tuple objects |
| `child` | int32 [n_dec, 4] | −1 illegal, ≥ 0 decision id, ≤ −2 terminal `−2 − tid` |
| `pot_at`, `to_call_at` | int16 [n_dec] | equal to `obs.pot` and `obs.to_call` for the actor |
| `street_contrib` | int16 [n_dec, 2] | (button, bb) chips put in on the node's street, i.e. `my_bet`/`opp_bet` mapped by seat |
| `term_fold_by_button` | int8 [n_term] | 1 = button folded, 0 = BB folded, −1 = showdown, including all-in run-outs |
| `term_contrib` | int16 [n_term, 2] | (button, bb) chips put in over the whole hand |
| `key_to_node` | **read-only Mapping** (`MappingProxyType`) | tuple of plain-int action codes → decision id |

**Read-only (R22)**
- `__post_init__` makes every array non-writeable and wraps `key_to_node` in a `MappingProxyType`.
- The dataclass is frozen.
- `__reduce__` keeps the tree picklable and copyable; a raw mappingproxy is not, which Task 18 multiprocessing may need. An unpickled tree comes back read-only too.
- `_chips()` raises ValueError instead of silently wrapping a chip value that does not fit int16. This cannot happen with the default constants.

**`get_tree()`**: `functools.cache` around `build_tree()`. It builds once per process (per bot load in the simulator, since `load_bot_file` re-imports `bot`). A failed build is not cached, so the caller should call it once in warmup and keep the result.

**`node_of(obs, tree) -> int | None`**
- Key: `obs.action_history` → plain-int codes → one dict lookup. No Game replay.
- Returns the node only if the observation also shows the node's state:
  - `legal_actions` (same order), `pot`, `to_call`, `is_button`, `street`;
  - chips in play: `my_stack + opp_stack + pot == 2 * starting_stack`.
- Returns None otherwise, and wraps everything in `try/except Exception`, so it never raises. Cases covered:
  - a history outside the tree;
  - a terminal history;
  - an element missing `.action`;
  - a non-int code (str, None, float, bool, numpy int);
  - an obs or tree that is None, a dict, or missing fields;
  - a state mismatch.
- **Interpretation to check:** "non-Action value" is read as any value that is not an int code. An `Action` from any copy of `game.py` counts, and so does a plain int 0–3. `1.0`, `True`, `"1"`, `np.int64(1)` and `"CHECK_CALL"` return None, even though `int()` would map them to CHECK_CALL.
  - A strict `type(x) is Action` check would silently miss every observation if a loader ever re-imported `game.py`. The test `test_actions_from_another_copy_of_game_py_still_map` locks this in.

**`node_from_actions(actions, tree) -> int | None`**: the same code validation plus a dict lookup, with no state check. It accepts any sequence of int codes or Actions, and never raises.

## Measurements (dev laptop, Python 3.14, numpy 2.5.4)

**Build time** (limit < 2 s)
- First `get_tree()` in a fresh process: 0.361–0.366 s over 3 runs (module import adds 33 ms).
- Warm `build_tree()`: 0.237–0.243 s.
- Roughly half of the build goes to `omaha_score` at the 12,769 showdown terminals, which the engine evaluates unavoidably.

**Memory footprint** (tracemalloc, live after the build): 3.55–3.69 MiB.
- Key tuples: 1.87 MiB
- Dict table: 0.56 MiB
- Node-id ints: 0.48 MiB
- numpy arrays: 0.53 MiB (`child` alone 0.24 MiB)
- `legal` tuple: 0.12 MiB
- Peak during the build: 10.6 MiB, all transient.

**Lookup speed** (13,011 real observations)

| function | median | p99 | max |
|---|---|---|---|
| `node_of` | 1.42 µs | 2.12 µs | 43 µs, one outlier |
| `node_of`, histories of 10+ actions | 2.17 µs | | |
| `node_from_actions` | 0.25 µs | 0.58 µs | |

**Retuned rules** (2,000 random hands each, monkeypatched `game` constants)

| retune | hits | misses |
|---|---|---|
| STARTING_STACK=60 | 0 | 8,286 |
| STARTING_STACK=100 | 0 | 8,602 |
| BIG_BLIND=4 | 0 | 7,929 |
| SMALL_BLIND=2 | 8,782 | 2,000 |
| RAISE_CAP=3 | 8,135 | 464 |
| RAISE_CAP=5 | 8,318 | 230 |

- The SMALL_BLIND=2 misses are the roots. Its hits are nodes whose betting state really is identical, for example after the button limps.
- Every hit was verified field by field.

## Tests (`tests/test_tree.py`, 37 tests, ~1.3 s)

**Required by the brief**
- `test_tree_size`: F1 counts, per-street bincount, dtypes and shapes of the interface.
- `test_preflop_nodes`
- `test_every_live_obs_maps_to_node_with_same_legal_actions`: 2,000 decisions, random button seats, weighted random legal actions so hands reach rivers and all-ins.
  - Asserts `node_of` is not None and `legal[node] == obs.legal_actions` (same order).
  - Asserts `pot_at`, `to_call_at`, `actor_is_button`, `street` and `street_contrib` (vs `my_bet`/`opp_bet`).
  - Asserts `node_from_actions(key) == node`.
  - Asserts `node_of(Observation.from_dict(obs.to_dict()))` gives the same node (the server's JSON path).
  - Every street is reached.
- `test_unknown_history_returns_none`: histories outside the tree, malformed elements and malformed observations (Review Focus 5).
- `test_terminal_contributions_match_game`: 500 random paths with random button seats, replayed down `child`.
  - Checks `term_contrib` against `game.contributed` mapped via the button seat.
  - Checks the fold flag against the result.
  - Checks the button's net as the trainer will compute it, from (flag, contrib, showdown winner), against `result.nets`.
  - All four terminal kinds occur: showdown, run-out, button folds, BB folds.

**Extra tests**
- Hand-derived literal tables: 11 nodes and 6 terminals, for example the pot ladder 6→18→54→80 and the all-in node `(F, C)` with pot 134 and to_call 26. I cross-checked the values against the engine before writing them.
- Structural invariants:
  - `child == -1` exactly where an action is illegal;
  - FOLD is legal exactly when `to_call > 0`, and a fold is always terminal;
  - every node and terminal has exactly one parent edge;
  - keys agree with edges;
  - the actor alternates within a street and the BB acts first on a new street;
  - showdown contributions are equal and a folder has less in.
- `test_every_node_and_terminal_matches_game_exhaustively`: walks the whole game with button_seat=1 (the tree is built with seat 0) and checks all 15,952 nodes, including `node_of`, and all 25,538 terminals.
- Build < 2 s, purity and caching; read-only and pickle round trip.
- Lookalike codes; Actions from another `game.py` copy; observations whose state disagrees with the node.
- Retuned rules, 6 cases: a hit must match the node exactly, and a retuned stack must give no hits.
- `node_from_actions`.
- `node_of` median < 20 µs (14× headroom).

## TDD evidence

1. **Cycle 1 RED.** Construction tests only: collection failed with `ModuleNotFoundError: No module named 'bot.tree'`.
2. **Cycle 1 GREEN.** Dataclass + `build_tree` + `get_tree`: 24 passed in 1.09 s.
3. **Cycle 2 RED.** Lookup tests added: `ImportError: cannot import name 'node_from_actions'`.
4. **Cycle 2 step 1.** The brief's literal lookup (`tuple(int(a.action) ...)` + `try/except`): **9 failed, 26 passed.**
   - Accepted `1.0` and `(1.0, 1)`.
   - Mapped `SimpleNamespace(action_history=())` to the root.
   - Mapped an observation with a changed pot.
   - Under every retune it returned nodes with wrong state. For example, STARTING_STACK=60 mapped to a node with pot 128 when the observation said 108, and RAISE_CAP=3 gave different legal sets.
5. **Cycle 2 GREEN.** Strict codes + state validation + `starting_stack`: 35 passed.
6. **Refactor.** The recursive closure DFS became iterative with byte-identical ids and arrays, keeping the suite green (see self-review).
   - I then added the SMALL_BLIND retune case and the `game.py` copy test: 37 passed.
7. **Mutation check.** All 24 realistic mutants are caught (one at a time, restored afterwards):
   - fold flag inverted; showdown flag 0; `term_contrib` swapped; actor inverted;
   - terminal code off by one; street bets swapped; `to_call` of the wrong seat; int8 chips;
   - starting stack without the blind; no cache; writeable arrays; `legal` as a list; no `__reduce__`;
   - each of the 6 validation checks removed;
   - bool accepted; plain `int()` coercion; no `try` in either lookup;
   - `type(x) is Action`.
8. **Full default suite** (worktree, before commit): **132 passed, 1 deselected, 1 warning** in 24 s. The warning is the pre-existing starlette/httpx deprecation from `test_agent.py`.
9. **Python 3.12.** Both files parse with `ast.parse(..., feature_version=(3, 12))`. No 3.12 interpreter exists locally. The APIs used are `functools.cache`, `MappingProxyType` and dataclasses, all available in 3.12.

## Self-review

- **Found and fixed a reference cycle before committing.** The recursive `visit` closure referred to itself through its own cell. That kept about 6.5 MiB of temporary build lists (and a copy of the key dict) alive until a full GC: 10.1 MiB retained instead of 3.6 MiB.
  - Fixed by switching to an iterative DFS and passing a `MappingProxyType` around build_tree's sole dict, so no copy is made.
  - Verified that ids and arrays are identical to the recursive build.
- **Spec deviations, both for R22 read-only.** `legal` is a tuple, not a list. `key_to_node` is a read-only Mapping, not a dict. Every read use works unchanged: `[]`, `.get`, `in`, `len`, iteration.
- **Extras beyond the brief:**
  - fields `n_term` and `starting_stack`;
  - `node_of`'s state check;
  - pickle support;
  - the int16 range guard.

## Concerns / notes for later tasks

1. **The state check in `node_of` is my design decision, beyond "None for an unknown history".** It turns a corrupt or other-rules observation into a heuristic fallback instead of a lookup in the wrong blueprint row. A retuned stack or BB misses everywhere.
   - A retuned **raise cap** still maps the nodes it doesn't change. So Task 20 must still compare `AgentConfig` with the blueprint meta (Review Focus 1).
2. **Ids depend on `game.py` and on the DFS order in `bot/tree.py`.**
   - Suggestion for Task 19: store `n_dec`, `n_term` and a hash of `child.tobytes()` in the blueprint meta, and verify them at load.
3. **Sim cost (Task 20).** `load_bot_file` re-imports `bot` on every load, so each half rebuilds the tree: about 0.36 s per half per bot when the policy calls `get_tree()`. That is fine for the tournament (startup ≤ 6 s) but adds up in long tuning runs.
   - Call `get_tree()` once in warmup or `new_half`, never lazily inside `choose_action`. The first call there would cost about 0.36 s of time bank.
4. **`get_tree()` thread-safety.** Two simultaneous first calls (FastAPI's threadpool) could build twice. That is harmless: equal trees, one wins. warmup() avoids it anyway.
5. **`node_of` returns None for an observation of a seat that is not to act** (empty `legal_actions`). Use `node_from_actions` for opponent-history mapping (Task 13/21).
6. **Two timing tests can flake on a very slow CI machine**: build < 2 s (5–8× headroom) and `node_of` median < 20 µs (14×).
