"""Logical cached-label isolation, not an operating-system security boundary."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

class CachedOracle:
    def __init__(self, labels, budget=32, batch_size=8):
        if len(labels) < budget or budget <= 0 or batch_size <= 0 or budget % batch_size:
            raise ValueError("Invalid budget or batch size")
        if not all(math.isfinite(v) for v in labels.values()):
            raise ValueError("Nonfinite labels")
        self._labels = dict(labels)
        self.budget, self.batch_size = budget, batch_size
        self.seen, self.sealed = set(), False

    def reveal(self, ids):
        if self.sealed or len(ids) != self.batch_size or len(set(ids)) != len(ids):
            raise ValueError("Wrong batch size, duplicate or closed episode")
        if not set(ids) <= self._labels.keys() or set(ids) & self.seen:
            raise ValueError("Unknown or previously revealed candidate")
        if len(self.seen) + len(ids) > self.budget:
            raise ValueError("Measurement budget exceeded")
        self.seen.update(ids)
        return [self._labels[i] for i in ids]

    def seal(self, journal_sha256):
        if self.sealed or len(self.seen) != self.budget or len(journal_sha256) != 64:
            raise ValueError("Incomplete episode or malformed seal")
        try:
            if len(bytes.fromhex(journal_sha256)) != 32: raise ValueError("Malformed SHA256")
        except ValueError: raise ValueError("Malformed SHA256") from None
        self.sealed = True

    def evaluate(self):
        if not self.sealed:
            raise ValueError("Full truth requires a sealed complete episode")
        return dict(self._labels)

class SuiteOracle:
    """Admits exactly a frozen arm/seed plan before any full-table readout."""
    def __init__(self, labels, plan):
        self.labels, self.plan = dict(labels), {tuple(key) for key in plan}
        if len(self.plan) != len(plan) or not self.plan: raise ValueError("Invalid suite plan")
        self.done, self.current = set(), None
    def start(self, arm, seed):
        key = (arm, seed)
        if self.current is not None or key not in self.plan or key in self.done:
            raise ValueError("Unplanned, duplicate or concurrent episode")
        self.key, self.current = key, CachedOracle(self.labels)
    def reveal(self, ids):
        if self.current is None: raise ValueError("No active episode")
        return self.current.reveal(ids)
    def seal(self, digest):
        if self.current is None: raise ValueError("No active episode")
        self.current.seal(digest)
        self.done.add(self.key); self.current = None
    def evaluate(self):
        if self.current is not None or self.done != self.plan:
            raise ValueError("All planned episodes must be sealed before full truth")
        return dict(self.labels)

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    oracle = SuiteOracle(labels, json.loads(args.plan.read_text())) if args.plan else CachedOracle(labels)
    for line in sys.stdin:
        try:
            q = json.loads(line)
            if q["op"] == "start" and isinstance(oracle, SuiteOracle):
                oracle.start(q["arm"], q["seed"]); result = {"started": True}
            elif q["op"] == "reveal": result = {"values": oracle.reveal(q["ids"])}
            elif q["op"] == "seal": oracle.seal(q["sha256"]); result = {"sealed": True}
            elif q["op"] == "evaluate": result = {"truth": oracle.evaluate()}
            else: raise ValueError("Unknown operation")
            print(json.dumps({"ok": True, **result}, allow_nan=False), flush=True)
        except (ValueError, KeyError, TypeError) as error:
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)
            break

if __name__ == "__main__": serve()
