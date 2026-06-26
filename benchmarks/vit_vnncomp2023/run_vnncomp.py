"""Full VNN-COMP-style protocol on the ViT benchmark at eps=1/255.

Per instance: (1) sound bound (symbolic-av reach + margins) -> VERIFIED if every
margin lower bound > 0; else (2) manual falsifier (random+PGD) -> FALSIFIED if a
concrete counterexample is found; else UNKNOWN. A "result" = VERIFIED + FALSIFIED
(a definitive verdict); UNKNOWN is a non-result. This is our verifier's honest
score on the benchmark.
"""
import sys, os, time, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from model import build_model
from reach import ViTReacher
from bab_vit import verify_instance_bab

EPS = 1.0 / 255


def run_model(name, n, writer, max_nodes, timeout_s):
    m = build_model(name, os.path.join(HERE, "onnx", f"{name}.onnx"))
    R = ViTReacher(m, mode="symbolic-av")
    z = np.load(os.path.join(HERE, "instances", f"{name}.npz"))
    imgs, labels = z["images"], z["labels"]
    n = min(n, len(imgs))
    counts = {"verified": 0, "falsified": 0, "unknown": 0}
    for k in range(n):
        x = imgs[k].astype(np.float64); y = int(labels[k])
        # BaB: bound (->VERIFIED) + falsify (->FALSIFIED) + input-split branch,
        # under a per-instance node/time budget (-> UNKNOWN on budget).
        res = verify_instance_bab(R, m, x, y, EPS, max_nodes=max_nodes,
                                  timeout_s=timeout_s, falsify_method="random+pgd")
        verdict = res.verdict.lower()
        counts[verdict] += 1
        writer.writerow([name, k, y, res.verdict, res.nodes, res.splits,
                         round(res.time_s, 1), res.reason])
        res_so_far = counts["verified"] + counts["falsified"]
        print(f"[{name} {k+1}/{n}] {res.verdict:9s} nodes={res.nodes} "
              f"({res.time_s:.0f}s) | V={counts['verified']} "
              f"F={counts['falsified']} U={counts['unknown']} "
              f"results={res_so_far} | {res.reason}", flush=True)
    return counts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-instances", type=int, default=100)
    ap.add_argument("--models", nargs="+", default=["pgd_2_3_16", "ibp_3_3_8"])
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results_vnncomp_fulleps.csv"))
    a = ap.parse_args()
    tot = {"verified": 0, "falsified": 0, "unknown": 0}
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "instance", "label", "verdict", "nodes", "splits",
                    "time_s", "reason"])
        for name in a.models:
            c = run_model(name, a.num_instances, w, a.max_nodes, a.timeout)
            for kk in tot:
                tot[kk] += c[kk]
            print(f"== {name}: V={c['verified']} F={c['falsified']} "
                  f"U={c['unknown']} | results={c['verified']+c['falsified']} ==", flush=True)
    n = sum(tot.values())
    print(f"\n==== FULL eps=1/255, {n} instances ====", flush=True)
    print(f"VERIFIED={tot['verified']}  FALSIFIED={tot['falsified']}  "
          f"UNKNOWN={tot['unknown']}", flush=True)
    print(f"RESULTS PROVIDED (V+F) = {tot['verified'] + tot['falsified']}/{n}", flush=True)
    print("VNNCOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
