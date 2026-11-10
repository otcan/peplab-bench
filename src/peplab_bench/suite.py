"""Opt-in new factorial replication; preserves original saved results."""
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

def run(args):
    import os
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import numpy as np
    from .__main__ import verify
    from .adapters import FrozenPredictor, FrozenSelector
    from .contracts import Prediction
    from .engine import run_episode
    verify(args.data)
    p=json.loads((args.data/'protocol.json').read_text())
    ids=p['candidate_ids']; seeds=p['combined_seeds'][:args.starts]
    if not 1 <= args.starts <= 200: raise ValueError('starts must be 1..200')
    structural=json.loads((args.data/'structural-energy.json').read_text())
    a=np.load(args.data/'sequence-chemistry.npz',allow_pickle=False)
    f={'S':a['S'],'C':a['C'],'K':np.asarray(structural['K'],float),'E':np.asarray(structural['E'],float)}
    if ids != structural['ids'] or [f[k].shape for k in 'SCKE'] != [(160,320),(160,2058),(160,36),(160,2)]:
        raise ValueError('Feature order or shape mismatch')
    if any(not np.isfinite(x).all() for x in f.values()): raise ValueError('Nonfinite feature')
    config={row['id']:row for row in p['configurations']}
    if len(config)!=150 or len(p['arms'])!=151: raise ValueError('Wrong frozen factorial')
    args.out.mkdir(parents=True,exist_ok=False)
    plan=[(arm,seed) for seed in seeds for arm in p['arms']]
    (args.out/'plan.json').write_text(json.dumps(plan)+'\n')
    deadline=time.time()+args.max_seconds
    process=subprocess.Popen([sys.executable,'-m','peplab_bench.oracle',str(args.data/'development-labels.json'),
                              '--plan',str(args.out/'plan.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    class RPC:
        def ask(self,q):
            process.stdin.write(json.dumps(q)+'\n');process.stdin.flush()
            value=json.loads(process.stdout.readline())
            if not value.get('ok'): raise ValueError(value.get('error','Oracle failure'))
            return value
        def reveal(self,chosen): return self.ask({'op':'reveal','ids':chosen})['values']
        def seal(self,digest): return self.ask({'op':'seal','sha256':digest})
    oracle=RPC();records=[];started=time.monotonic()
    try:
        for seed in seeds:
            order=np.random.default_rng(seed).permutation(160).tolist()
            initial=[ids[i] for i in order[:8]]
            for arm in p['arms']:
                if time.time()>=deadline-10: raise TimeoutError('User execution deadline; incomplete suite is not evaluated')
                oracle.ask({'op':'start','arm':arm,'seed':seed})
                c=config.get(arm)
                if c:
                    predictor=FrozenPredictor(ids,f,c['predictor'],deadline)
                    selector=FrozenSelector(c['selector']);sources=c['sources']
                else:
                    def predictor(visible,query,sources,seed): return Prediction(tuple(query),tuple(0. for _ in query))
                    def selector(prediction,visible,batch_size,seed):
                        eligible=set(prediction.ids)
                        return [ids[i] for i in order if ids[i] in eligible][:8]
                    sources=[]
                journal=args.out/f'{seed}--{arm}.jsonl'
                result=run_episode(ids,initial,predictor,selector,oracle,journal,sources,seed)
                records.append({'arm':arm,'seed':seed,**result})
                (args.out/'progress.json').write_text(json.dumps({'complete':len(records),'planned':len(plan),'seed':seed,'arm':arm})+'\n')
        assert len(records)==len(plan)
        # The complete suite seal exists on disk before requesting full truth.
        seal={'status':'ALL_NEW_EPISODES_SEALED','episode_seals':records,'elapsed_s':time.monotonic()-started}
        raw=(json.dumps(seal,allow_nan=False)+'\n').encode()
        with (args.out/'all-selection-seal.json').open('xb') as stream:stream.write(raw)
        truth=oracle.ask({'op':'evaluate'})['truth']
        if set(truth)!=set(ids): raise ValueError('Wrong full-truth cohort')
        cutoff=sorted(truth.values())[-8]
        above={i for i in ids if truth[i]>cutoff};boundary={i for i in ids if truth[i]==cutoff}
        def hits(seen): return len(set(seen)&above)+min(8-len(above),len(set(seen)&boundary))
        with (args.out/'new-replication-episodes.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['arm','seed','top8_hits_budget8','top8_hits_budget16','top8_hits_budget24','top8_hits_budget32'])
            writer.writeheader()
            for row in records:
                writer.writerow({'arm':row['arm'],'seed':row['seed'],**{f'top8_hits_budget{n}':hits(row['observed_ids'][:n]) for n in [8,16,24,32]}})
        result={'status':'NEW_COMPLETE_PORTABLE_REPLICATION','episodes':len(records),'source_protocol_sha256':hashlib.sha256((args.data/'protocol.json').read_bytes()).hexdigest(),
                'selection_seal_sha256':hashlib.sha256(raw).hexdigest(),'elapsed_s':time.monotonic()-started,'original_study_modified':False,'new_assays':0}
        (args.out/'replication-complete.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    except BaseException as error:
        (args.out/'INCOMPLETE.json').write_text(json.dumps({'complete':len(records),'planned':len(plan),'error':type(error).__name__+': '+str(error),'no_automatic_retry':True})+'\n')
        raise
    finally:
        process.stdin.close()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.terminate();process.wait(timeout=5)
