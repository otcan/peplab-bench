"""Separate cached-label process. No learner imports; not OS-level blinding."""
from pathlib import Path
import hashlib, json, sys

class Oracle:
    def __init__(self, labels, ids, arms, seeds):
        assert set(labels) == set(ids) and len(ids) == 160
        self.labels, self.ids, self.arms, self.seeds = labels, set(ids), set(arms), set(seeds)
        self.done, self.current, self.seen = set(), None, set()

    def request(self, q):
        op = q['op']
        if op == 'start':
            key = (q['arm'], q['seed'])
            assert self.current is None and key not in self.done
            assert key[0] in self.arms and key[1] in self.seeds
            self.current, self.seen = key, set()
            return {'started': True}
        if op == 'reveal':
            assert self.current is not None
            ids = q['ids']
            assert len(ids) == 8 and len(set(ids)) == 8
            assert set(ids) <= self.ids and not set(ids) & self.seen
            assert len(self.seen) + len(ids) <= 32
            self.seen.update(ids)
            return {'ids': ids, 'values': [self.labels[i] for i in ids], 'observed_count': len(self.seen)}
        if op == 'seal':
            assert self.current is not None and len(self.seen) == 32
            assert len(q['selection_journal_SHA256']) == 64
            self.done.add(self.current)
            self.current = None
            return {'sealed_episodes': len(self.done)}
        if op == 'evaluate':
            assert self.current is None and len(self.done) == len(self.arms) * len(self.seeds)
            return {'all_development_truth': self.labels, 'sealed_episodes': len(self.done)}
        raise ValueError('Unknown oracle operation')

def main():
    r = Path(__file__).resolve().parent
    p = json.loads((r/'protocol.json').read_text())
    o = Oracle(json.loads((r/'development-labels.json').read_text()), p['candidate_ids'], p['arms'], p['seeds'])
    log = (r/'attempt/oracle-audit.jsonl').open('x')
    for line in sys.stdin:
        q = json.loads(line)
        try:
            reply = o.request(q)
            log.write(json.dumps({'request':q, 'observed_count':len(o.seen), 'completed_episodes':len(o.done)})+'\n')
            log.flush()
            print(json.dumps({'ok':True, **reply},allow_nan=False),flush=True)
        except Exception as e:
            print(json.dumps({'ok':False,'error':type(e).__name__+':'+str(e)}),flush=True)
            raise

if __name__ == '__main__': main()
