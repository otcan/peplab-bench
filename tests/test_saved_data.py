import ast
import csv
import hashlib
import json
import unittest
import zipfile
from pathlib import Path
from peplab_bench.__main__ import verify

ROOT = Path(__file__).resolve().parents[1]
class SavedData(unittest.TestCase):
    def test_hashes(self): self.assertGreaterEqual(verify(ROOT / "data"), 9)
    def test_exact_cohort_and_source_shapes(self):
        d=ROOT/"data"; c=json.loads((d/"cohort.json").read_text()); s=json.loads((d/"structural-energy.json").read_text()); p=json.loads((d/"protocol.json").read_text())
        self.assertEqual(len(set(c["ids"])),160); self.assertEqual(c["ids"],s["ids"]); self.assertEqual(c["ids"],p["candidate_ids"])
        self.assertEqual(set(json.loads((d/"development-labels.json").read_text())),set(c["ids"]))
        self.assertEqual(len(s["K"]),160); self.assertTrue(all(len(row)==36 for row in s["K"]))
        self.assertEqual(len(s["E"]),160); self.assertTrue(all(len(row)==2 for row in s["E"]))
        with zipfile.ZipFile(d/"sequence-chemistry.npz") as z: self.assertEqual(set(z.namelist()),{"S.npy","C.npy"})
        self.assertFalse("reserved_ids" in p)
    def test_complete_saved_matrix_and_equal_start(self):
        with (ROOT/"data/all151-methods.csv").open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),151); self.assertEqual(len({x["arm"] for x in rows}),151)
        self.assertTrue(all(int(x["episodes"])==200 for x in rows))
        self.assertEqual({x["mean_hits_budget8"] for x in rows},{"0.51"})
        self.assertEqual(next(x for x in rows if x["arm"]=="random")["mean_top8_hits_at32"],"1.78")
    def test_saved_core_provenance(self):
        for r in json.loads((ROOT/"docs/numerical-core.json").read_text()):
            path=ROOT/"src/peplab_bench/frozen"/r["source_name"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),r["portable_sha256"])
            functions={n.name for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            self.assertEqual(functions,set(r["retained_functions_AST_identical"]))

if __name__ == "__main__": unittest.main()
