"""Runnable interface example with mock predictions, no assay fitting."""
from peplab_bench.contracts import Prediction, greedy
from peplab_bench.engine import run_episode
from peplab_bench.oracle import CachedOracle
import tempfile
from pathlib import Path

ids = tuple(f"toy-{i:02}" for i in range(40))
oracle = CachedOracle({i: float(n) for n, i in enumerate(ids)})
def mock_predictor(visible, query, sources, seed):
    # Replace with your trained model. It receives only selected responses.
    return Prediction(tuple(query), tuple(0.0 for _ in query))
with tempfile.TemporaryDirectory() as folder:
    result = run_episode(ids, ids[:8], mock_predictor,
        lambda prediction, visible, batch_size, seed: greedy(prediction, batch_size),
        oracle, Path(folder)/"journal.jsonl", ("toy-source",), 123)
    print("Mock contract cycle complete:", len(result["observed_ids"]), "reveals; journal sealed")
