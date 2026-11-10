"""View saved results without numerical dependencies; opt in to fresh replay."""
import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path

def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "checksums.json").read_text())
    for name, digest in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Hash mismatch or unsafe path: " + name)
    return len(manifest)

def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def show(args):
    values = sorted(rows(args.data / "all151-methods.csv"), key=lambda x: (-float(x["mean_top8_hits_at32"]), x["arm"]))
    print("Saved development results: 200 shared starts; 32 cached measurement reveals; top-8 hit count.")
    for row in values[:args.limit]:
        print(f"{row['arm']:<43} {float(row['mean_top8_hits_at32']):.3f}  MCSE {float(row['MC_standard_error']):.4f}")
    if args.html:
        fields = ["arm", "mean_top8_hits_at32", "MC_standard_error", "mean_recall_at32", "original20_mean"]
        table = "<table><thead><tr>" + "".join("<th>" + html.escape(k) + "</th>" for k in fields) + "</tr></thead><tbody>"
        table += "".join("<tr>" + "".join("<td>" + html.escape(row[k]) + "</td>" for k in fields) + "</tr>" for row in values) + "</tbody></table>"
        doc = "<!doctype html><html lang='en'><meta charset='utf-8'><title>PepLab Bench results</title><style>body{max-width:1100px;margin:3em auto;font:16px system-ui}td,th{padding:.5em;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>PepLab Bench: all 151 configurations</h1><p>160 previously exposed HLA-A*02:01 nine-mers. 200 starts, 30,200 replay episodes. Each method receives the same initial eight per start and three batches of eight. MCSE describes random-start simulation precision. These are archived assay outcomes; no new experiments or prospective confirmation.</p>" + table + "</html>"
        with args.html.open("x", encoding="utf-8") as stream: stream.write(doc)

def replay(args):
    # This command performs real fitting only when explicitly requested.
    import os
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import subprocess
    import time
    import numpy as np
    from .adapters import FrozenPredictor, FrozenSelector
    from .engine import run_episode
    verify(args.data)
    meta = json.loads((args.data / "cohort.json").read_text())
    ids = meta["ids"]
    arrays = np.load(args.data / "sequence-chemistry.npz", allow_pickle=False)
    structural = json.loads((args.data / "structural-energy.json").read_text())
    if structural["ids"] != ids: raise ValueError("Structural candidate order mismatch")
    f = {"S": arrays["S"], "C": arrays["C"], "K": np.asarray(structural["K"], float), "E": np.asarray(structural["E"], float)}
    if len(ids) != 160 or [f[k].shape for k in "SCKE"] != [(160, 320), (160, 2058), (160, 36), (160, 2)]:
        raise ValueError("Wrong frozen feature dimensions")
    if any(not np.isfinite(x).all() for x in f.values()): raise ValueError("Nonfinite features")
    args.out.mkdir(parents=True, exist_ok=False)
    process = subprocess.Popen([sys.executable, "-m", "peplab_bench.oracle", str(args.data / "development-labels.json")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    class RemoteOracle:
        def ask(self, value):
            process.stdin.write(json.dumps(value) + "\n"); process.stdin.flush()
            reply = json.loads(process.stdout.readline())
            if not reply.get("ok"): raise ValueError(reply.get("error", "Oracle failure"))
            return reply
        def reveal(self, ids): return self.ask({"op": "reveal", "ids": ids})["values"]
        def seal(self, digest): return self.ask({"op": "seal", "sha256": digest})
    oracle = RemoteOracle()
    try:
        predictor = FrozenPredictor(ids, f, args.predictor, time.time() + args.max_seconds)
        initial = [ids[i] for i in np.random.default_rng(args.seed).permutation(160)[:8]]
        result = run_episode(ids, initial, predictor, FrozenSelector(args.selector), oracle,
                             args.out / "selections.jsonl", args.sources, args.seed)
        # A single new episode is separate from the frozen study, and does not
        # calculate or append an aggregate benchmark endpoint.
        result.update(status="NEW_USER_REQUESTED_SINGLE_EPISODE", sources=args.sources,
                      predictor=args.predictor, selector=args.selector, seed=args.seed,
                      new_assays=0, logical_visibility_only=True)
        (args.out / "selection-seal.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("Single episode sealed; 32 cached values revealed. Frozen study unchanged.")
    finally:
        process.stdin.close()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.terminate(); process.wait(timeout=5)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("results", help="Inspect archived results, no model fitting")
    p.add_argument("--data", type=Path, default=Path("data")); p.add_argument("--limit", type=int, default=10)
    p.add_argument("--html", type=Path); p.set_defaults(func=show)
    p = sub.add_parser("verify", help="Verify the allowlisted public data hashes")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.set_defaults(func=lambda args: print(f"Verified {verify(args.data)} data files"))
    p = sub.add_parser("replay", help="Opt-in NEW single-episode fitting; install [models]")
    p.add_argument("--data", type=Path, default=Path("data")); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sources", choices=["S", "C", "K", "E", "SC", "SK", "SE", "CK", "CE", "KE", "SCK", "SCE", "SKE", "CKE", "SCKE"], default="SCK")
    p.add_argument("--predictor", choices=["GP", "XGB_bootstrap16"], default="GP")
    p.add_argument("--selector", choices=["greedy", "EI", "UCB", "TS", "top8utility"], default="greedy")
    p.add_argument("--seed", type=int, default=2026110420)
    p.add_argument("--max-seconds", type=int, default=600); p.set_defaults(func=replay)
    p = sub.add_parser("benchmark", help="Opt-in NEW complete factorial replication; install [models]")
    p.add_argument("--data", type=Path, default=Path("data")); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--starts", type=int, choices=range(1,201), default=1)
    p.add_argument("--max-seconds", type=int, default=3600)
    def suite(args):
        from .suite import run
        run(args)
    p.set_defaults(func=suite)
    args = parser.parse_args()
    if hasattr(args, "max_seconds") and args.max_seconds < 30: parser.error("max-seconds must be >=30")
    args.func(args)

if __name__ == "__main__": main()
