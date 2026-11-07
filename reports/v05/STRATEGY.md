# v0.5 strategy contract

This revision is authorized by the user's request to fix the four baseline
matchups. The official rules/protocol remain authoritative. These documented
changes supersede the corresponding strategic hypotheses in DESIGN v0.4;
the original design and evidence remain archived unchanged.

Deployment: fixed `v05_limp20` strategy, `FULL_PAIR_v04` evaluator,
adaptation enabled, pure bluffs disabled, sampler salt 0. There is no mode or
configuration switching during a half. Opponent adjustments use only observed
action history and resolved responses, not opponent names, hidden cards, match
seeds or reconstructed baseline RNG state.

The feature score, packed-score lookup evaluator, fragment tiers, draw formulas,
legal-action guards, aliases, bounded memory and terminal transactions are unchanged.
The revision makes these conditional changes:

1. Keep OVERFOLDER first. When the original aggression evidence threshold is met,
   AGGRESSIVE now precedes STATION. Aggression requires either at least 16 observed
   postflop nonfold actions with smoothed raise rate >=.42, or at least 20 observed
   first preflop actions with smoothed preflop raise rate >=.50. A player who raises
   constantly and never folds is therefore treated as aggressive. CallingStation
   does not satisfy this raising evidence.
2. Against AGGRESSIVE, MEDIUM and STRONG postflop hands have an actual-call price
   ceiling of .34, including dangerous boards and repeated pot raises. These
   categories are policy features, not equity estimates. Other tiers retain their
   original ceilings and draw/context rules. Proven locked nuts retain .34.
3. Against AGGRESSIVE, a preflop score >=20 sets the facing-raise ceiling to .34,
   including repeated raises. Lower scores use the original willingness ceiling.
   This is a wider defense, not an assumption that every hand is profitable.
4. After a call is approved against AGGRESSIVE, call instead of reraising. A
   rejected call still folds. This prevents recursive pot-building against an
   opponent already willing to supply the bets.
5. On the button's initial action against AGGRESSIVE, fold scores <20 as before;
   otherwise complete the small blind rather than opening a raise. The ordinary
   BB-option branch and all nonaggressive opening bands remain as specified in v0.4.

UNKNOWN/TIGHT/STATION/OVERFOLDER call pricing is unchanged. Initial hands use
UNKNOWN until sufficient observations exist. Every half resets all evidence.

The changes were isolated through named cumulative screens: strong-hand pricing,
medium-hand pricing, preflop defense at 35 then 30, trapping, button limping, and
the profile-priority correction. The complete failed and successful screens are
retained. Finalists were tested through the official HTTP referee on fresh train
and confirmation seeds before selection. The final configuration was frozen before
five new audit seeds were opened; original `final-73`/`final-97` were not reused.

This policy is engineered to exploit the measured baseline styles. It is not a
solver, a proof of optimal play, or a promise to win every future poker match.

The first selected revision used score 30 and recorded 19 wins/one chop on its
new final audit. A separately preregistered phase B varied only this conditional
score threshold on new development/confirmation seeds; score 20 met its promotion
criteria. The original audit remains intact, and the refined deployment uses a
new five-seed audit. See `reports/v05b/experiment_plan.json` and `selection.json`.
