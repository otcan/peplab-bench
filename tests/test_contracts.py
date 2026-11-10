import json
import tempfile
import unittest
from pathlib import Path
from peplab_bench.contracts import Prediction, greedy, supported_selectors
from peplab_bench.engine import run_episode
from peplab_bench.oracle import CachedOracle, SuiteOracle

class Contracts(unittest.TestCase):
    def setUp(self): self.labels = {str(i): float(i) for i in range(40)}
    def test_early_truth_rejected(self):
        with self.assertRaises(ValueError): CachedOracle(self.labels).evaluate()
    def test_duplicate_reveal_atomic(self):
        o = CachedOracle(self.labels); o.reveal([str(i) for i in range(8)])
        with self.assertRaises(ValueError): o.reveal([str(i) for i in range(7, 15)])
        self.assertEqual(len(o.seen), 8)
    def test_budget_and_seal(self):
        o = CachedOracle(self.labels)
        for start in range(0, 32, 8): o.reveal([str(i) for i in range(start, start+8)])
        with self.assertRaises(ValueError): o.reveal([str(i) for i in range(32, 40)])
        with self.assertRaises(ValueError): o.seal("z"*64)
        with self.assertRaises(ValueError): o.seal(" "*64)
        o.seal("a"*64); self.assertEqual(o.evaluate(), self.labels)
        with self.assertRaises(ValueError): o.seal("a"*64)
    def test_partial_seal_rejected(self):
        with self.assertRaises(ValueError): CachedOracle(self.labels).seal("a"*64)
    def test_suite_requires_every_planned_episode_before_truth(self):
        oracle=SuiteOracle(self.labels,[("a",1),("b",1)])
        for arm in ['a','b']:
            with self.assertRaises(ValueError): oracle.evaluate()
            oracle.start(arm,1)
            with self.assertRaises(ValueError): oracle.start(arm,1)
            for start in range(0,32,8): oracle.reveal([str(i) for i in range(start,start+8)])
            oracle.seal('a'*64)
        self.assertEqual(oracle.evaluate(),self.labels)
        with self.assertRaises(ValueError): oracle.start('a',1)
    def test_unplanned_suite_episode_rejected(self):
        oracle=SuiteOracle(self.labels,[("a",1)])
        with self.assertRaises(ValueError): oracle.start('b',1)
    def test_missing_uncertainty_masks_actions(self):
        p = Prediction(("a", "b"), (1., 2.))
        self.assertEqual(supported_selectors(p), ("greedy",)); self.assertEqual(greedy(p, 1), ["b"])
    def test_prediction_order_and_nonfinite(self):
        with self.assertRaises(ValueError): Prediction(("b", "a"), (1., 2.)).validate(("a", "b"))
        with self.assertRaises(ValueError): Prediction(("a",), (float("nan"),)).validate(("a",))
    def test_bad_covariance(self):
        with self.assertRaises(ValueError): Prediction(("a",), (1.,), ((-1.,),)).validate(("a",))
    def test_greedy_ties_stable(self):
        self.assertEqual(greedy(Prediction(("a", "b", "c"), (1., 1., 0.)), 2), ["a", "b"])
    def test_visible_only_deterministic_cycle_and_pre_reveal_journal(self):
        def run(folder):
            path = Path(folder) / "journal.jsonl"; o = CachedOracle(self.labels)
            class CheckedOracle:
                def reveal(self, ids):
                    last = json.loads(path.read_text().splitlines()[-1])
                    assert last["selected_ids"] == list(ids) and last["selected_before_reveal"]
                    return o.reveal(ids)
                def seal(self, digest): o.seal(digest)
            def predictor(visible, query, sources, seed):
                self.assertEqual(set(visible), o.seen)
                self.assertFalse(set(visible) & set(query))
                self.assertEqual(len(visible) + len(query), 40)
                # A mock prediction, deliberately no numerical model or assay fit.
                return Prediction(tuple(query), tuple(0. for _ in query))
            result = run_episode(tuple(self.labels), list(self.labels)[:8], predictor,
                lambda p, y, n, seed: greedy(p, n), CheckedOracle(), path, ("S",), 1)
            self.assertEqual(len(result["observed_ids"]), 32); self.assertTrue(o.sealed)
            return path.read_bytes()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(run(a), run(b))
    def test_selector_duplicate_rejected_before_reveal(self):
        with tempfile.TemporaryDirectory() as folder:
            o = CachedOracle(self.labels)
            with self.assertRaises(ValueError):
                run_episode(tuple(self.labels), list(self.labels)[:8],
                    lambda v,q,s,z: Prediction(tuple(q), tuple(0. for _ in q)),
                    lambda *args: ["8"] * 8, o, Path(folder)/"x", ("S",), 1)
            self.assertEqual(len(o.seen), 8)

if __name__ == "__main__": unittest.main()
