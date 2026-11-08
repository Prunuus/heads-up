import random, agent
from bot import policy
from game import Game, AgentConfig, make_deal
w = policy.WARMUP
print("warmup failed:", w.failed, "| blueprint loaded:", w.blueprint is not None,
      "| iterations:", (w.blueprint.meta["iterations"] if w.blueprint else None))
bot, mem = agent.BOT, {}
bot.new_half(AgentConfig(), mem)
rng = random.Random(1)
for h in range(60):
    g = Game(make_deal("probe", h), button_seat=h % 2, hand_index=h)
    while not g.is_terminal:
        s = g.to_act
        obs = g.observation(s)
        g.apply(bot.choose_action(obs, mem) if s == 0 else rng.choice(obs.legal_actions))
print("layers:", dict(mem.get("layers", {})), "| faults:", dict(mem.get("faults", {})))
