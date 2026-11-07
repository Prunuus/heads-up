# LLM-generated opponent bots — generation prompts

Each bot was written by a fresh AI agent with no access to our own bot or plan. It saw only the participant guide (text extracted from the PDF), the starter `agent.py`, `evaluator.py`, `game.py` and the README, which is everything a real competing team gets. Each agent got the shared preamble below plus one team-request line.

## Shared preamble
> You are helping a student team in the TAMU Datathon Poker Bot Challenge. Read the participant guide (guide.txt), the README, and the starter files agent.py, evaluator.py and game.py in the repo. Then write our complete agent.py: the starter file with the three functions filled in, and the server plumbing unchanged. Team request: <variant>

## Variants
| file | model | team request |
|---|---|---|
| simple_rules.py | haiku | "Write a simple rules-based bot that beats the baseline bots. Keep it simple and reliable." |
| beginner_2h.py | haiku | "We're beginners with about 2 hours. Give us something decent that works." |
| aggressive.py | haiku | "Write an aggressive bot that puts lots of pressure on opponents; we think most bots fold too much." |
| mc_equity.py | sonnet | "Write a strong bot using Monte Carlo equity estimation and pot odds." |
| opp_model.py | sonnet | "Write a bot that tracks the opponent's tendencies and adapts its strategy during the match." |
| gto_inspired.py | sonnet | "Write a balanced, game-theory-inspired bot that is hard to exploit (mix bluffs and value bets by frequency)." |
| ml_minded.py | sonnet | "We like machine learning. Write the most data-driven bot you can without needing a long training run (small precomputed tables are fine if generated inside the file at startup)." |
| time_careful.py | sonnet | "Write the strongest bot you can, but be very careful with the 8-second time bank and never crash." |
