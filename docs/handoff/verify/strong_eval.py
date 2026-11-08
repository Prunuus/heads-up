"""Extra evaluation of the FINAL release c1ded16 (v3.1) and v2 vs all 6 competitor bots, 5 seeds (controller analysis, R39)."""
import functools, json, os, sys
import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[3])


def main():
    os.chdir(ROOT); sys.path.insert(0, ROOT)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(range(0, 8)))
    import sim.gate_v3 as g
    from sim.tournament import focus_eval
    from sim.opponents import ZOO, FIELD, FIELD2
    cands = {"v3.1": g._policy_bot, "v2": functools.partial(g._policy_bot, use_blueprint=False)}
    seeds = ["x1", "x2", "x3", "x4", "x5"]
    res = {}
    for o in sorted(FIELD) + sorted(FIELD2):
        for name, f in cands.items():
            r = focus_eval(f, {o: ZOO[o]}, seeds, 1000, workers=2)
            ms = r.matches[o]
            pts = sum(m.points_a for m in ms.values()) / len(ms)
            mean = sum(m.mean_bb for m in ms.values()) / len(ms)
            verdicts = "".join({"A": "W", "B": "L", "CHOP": "C"}.get(m.verdict, "?") for m in ms.values())
            res[f"{o}|{name}"] = {"points": pts, "mean_bb": mean, "verdicts": verdicts}
            print(f"{o:16s} {name:5s} points {pts:.2f}  mean {mean:+.2f} bb/deal  {verdicts}", flush=True)
    json.dump(res, open("train_data/strong_eval_c1ded16.json", "w"), indent=1)


if __name__ == "__main__":
    main()
