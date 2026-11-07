### Task 4: LLM-generated opponent population (I7)

**Files:** Create `sim/opponents/llm/<name>.py` (≥ 8 bots), `sim/opponents/llm/PROMPTS.md`, and a `sim/opponents/__init__.py` registration under the `llm_` prefix with `HOLDOUT: set[str]`. Test: extend `tests/test_zoo.py`.

- [ ] **Step 1:** Write 8 prompt variants and record them in `PROMPTS.md`:
  - "simple rules bot"
  - "Monte Carlo equity bot"
  - "opponent-modelling bot"
  - "aggressive bot"
  - "GTO-inspired bot"
  - "beginner team, 2 hours"
  - "ML-minded team"
  - "careful about time bank"

  Each prompt gets only the guide PDF text and `agent.py`.
- [ ] **Step 2:** Dispatch one fresh subagent per prompt. Each writes a complete `agent.py`-style file. Save only the YOUR CODE section plus imports, adapted into a class.
- [ ] **Step 3:** Mark 3 of the 8 bots, plus `threshold_cat`, as `HOLDOUT`. Never tune against them.
- [ ] **Step 4:** Run the parametrized zoo test → PASS. Bots that crash stay in the zoo and are recorded as "forfeit-prone", because they're realistic opponents. The test for those asserts `forfeit_a or errors_a == 0`.
- [ ] **Step 5:** Commit: `feat(sim): LLM-generated opponent population with holdout split`. (The diversity report is produced in Task 5, Step 5.)

