"""Replay orchestration. Predictors see only selected response values."""
import hashlib
import json
import os
from pathlib import Path

def run_episode(ids, initial, predictor, selector, oracle, journal_path, sources, seed):
    ids = tuple(ids)
    if len(set(ids)) != len(ids) or len(initial) != 8 or len(set(initial)) != 8 or not set(initial) <= set(ids):
        raise ValueError("Invalid candidate universe or initial batch")
    path = Path(journal_path)
    visible = {}
    def reveal(chosen):
        values = oracle.reveal(chosen)
        if len(values) != len(chosen): raise ValueError("Oracle response length mismatch")
        visible.update(zip(chosen, values))
    with path.open("x", encoding="utf-8") as stream:
        def record(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        record({"stage": "initial", "selected_ids": list(initial), "selected_before_reveal": True})
        reveal(initial)
        for round_index in range(3):
            query = [i for i in ids if i not in visible]
            prediction = predictor(dict(visible), query, tuple(sources), seed + 1000 + round_index)
            prediction.validate(query)
            chosen = list(selector(prediction, tuple(visible.values()), 8, seed + 1000 + round_index))
            if len(chosen) != len(set(chosen)) or len(chosen) != 8 or not set(chosen) <= set(query):
                raise ValueError("Selector must choose eight distinct unseen candidates")
            record({"round": round_index + 1, "visible_ids": list(visible), "query_ids": query,
                    "predictions": list(prediction.means), "selected_ids": chosen,
                    "uncertainty": prediction.uncertainty_label, "selected_before_reveal": True})
            reveal(chosen)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    oracle.seal(digest)
    return {"journal_sha256": digest, "observed_ids": list(visible), "observed_values": list(visible.values())}
