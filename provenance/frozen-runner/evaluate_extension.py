"""One-time full200 summary, strictly after global seal. Runs on Vast only."""
from pathlib import Path
import datetime,hashlib,json,math,os,time
import numpy as np
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert str(R)=='/workspace/easylab-repetitions200-v1' and os.environ.get('EASYLAB_EXECUTION_SCOPE')=='vast-repetitions200-v1'
 p=json.loads((R/'protocol.json').read_text());a=json.loads((R/'attempt/admission.json').read_text());assert time.time()<a['science_deadline_epoch_s']
 assert not (R/'attempt/final-results.json').exists() and not (R/'attempt/evaluation-started.json').exists(),'One evaluation only'
 seal=json.loads((R/'attempt/all-new-selection-seal.json').read_text());assert seal['all_completed_before_full_truth'] and seal['new_episodes']==27180 and len(seal['seeds'])==180
 assert sha(R/'original20-results.json')==p['original_result_SHA256']==seal['original_result_SHA256']
 original=json.loads((R/'original20-results.json').read_text());assert len(original['episodes'])==3020
 new=[]
 for row in seal['seeds']:
  d=R/'attempt/seeds'/str(row['seed'])
  assert sha(d/'selection-seal.json')==row['selection_seal_SHA256'] and sha(d/'episodes-unevaluated.json')==row['episodes_SHA256'] and sha(d/'selections.jsonl')==row['journal_SHA256']
  ee=json.loads((d/'episodes-unevaluated.json').read_text());assert len(ee)==151 and all(e['seed']==row['seed'] for e in ee)
  new+=ee
 assert len(new)==27180 and len({(e['arm'],e['seed']) for e in new})==27180
 started={'at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'global_selection_seal_SHA256':sha(R/'attempt/all-new-selection-seal.json'),'new_episodes_verified':27180}
 (R/'attempt/evaluation-started.json').write_text(json.dumps(started,indent=2)+'\n')
 # Aggregate endpoint evaluation begins only after the global seal. Oracle
 # processes necessarily loaded the cached table earlier; this is software-flow
 # isolation, not OS secrecy. Qualification used only the old first-context labels.
 truth=json.loads((R/'development-labels.json').read_text());ids=p['candidate_ids'];assert set(truth)==set(ids) and len(ids)==160
 cutoff=sorted(truth.values())[-8];above={i for i in ids if truth[i]>cutoff};boundary={i for i in ids if truth[i]==cutoff}
 def count(seen):return len(set(seen)&above)+min(8-len(above),len(set(seen)&boundary))
 for ep in new:
  assert len(ep['history'])==4 and [len(h) for h in ep['history']]==[8,16,24,32]
  assert ep['observed_values']==[truth[i] for i in ep['observed_ids']]
  ep['curve']=[{'measurements':len(v),'true_top8_hits':count(v),'best_measured_pIC50':max(truth[i] for i in v)} for v in ep['history']]
 episodes=original['episodes']+new;by={(e['arm'],e['seed']):e for e in episodes}
 assert len(episodes)==len(by)==30200
 config={c['id']:c for c in p['configurations']};summary=[]
 for arm in p['arms']:
  es=[by[arm,s] for s in p['combined_seeds']];v=np.array([e['curve'][-1]['true_top8_hits'] for e in es],float)
  assert len(es)==200
  mean=float(v.mean());sd=float(v.std(ddof=1));se=sd/math.sqrt(200)
  summary.append({'arm':arm,**config.get(arm,{'sources':[],'predictor':'none','selector':'random'}),'episodes':200,
       'mean_top8_hits_at32':mean,'sd_across_initializations':sd,'MC_standard_error':se,'mean_recall_at32':mean/8,
       'mean_discovery_curve':[float(np.mean([e['curve'][j]['true_top8_hits'] for e in es])) for j in range(4)],
       'total_model_wall_s':sum(e['elapsed_s'] for e in es),'new_model_wall_s':sum(e['elapsed_s'] for e in es if e['seed'] in p['seeds']),
       'actual_fit_counts':{k:sum(e['actual_fit_counts'][k] for e in es) for k in ['GP','XGBoost']},
       'original20_mean':next(x['mean_top8_hits_at32'] for x in original['summary'] if x['arm']==arm)})
 def paired(bigger,base):
  d=np.array([by[bigger,s]['curve'][-1]['true_top8_hits']-by[base,s]['curve'][-1]['true_top8_hits'] for s in p['combined_seeds']],float)
  return {'paired_mean_delta_hits':float(d.mean()),'paired_sd':float(d.std(ddof=1)),'paired_MC_standard_error':float(d.std(ddof=1)/math.sqrt(200)),
          'wins':int(sum(d>0)),'ties':int(sum(d==0)),'losses':int(sum(d<0)),'starts':200}
 contrasts=[]
 for base in p['configurations']:
  for source in 'SCKE':
   if source in base['sources']:continue
   larger=''.join(k for k in 'SCKE' if k in base['sources']+[source])+'__'+base['predictor']+'__'+base['selector']
   contrasts.append({'base':base['id'],'added_source':source,'larger':larger,**paired(larger,base['id'])})
 comparisons=[{'larger':x,'base':y,**paired(x,y)} for x,y in p['predeclared_report_comparisons']]
 assert len(summary)==151 and len(contrasts)==280
 result={'status':'COMPLETE_200_START_DEVELOPMENT_EXTENSION','decision':'DEVELOPMENT_REPETITION_PRECISION_ONLY_NO_EXTERNAL_CONFIRMATION',
      'at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary':summary,'episodes':episodes,'matched_source_contrasts':contrasts,
      'predeclared_comparisons':comparisons,'new_shared_starts':180,'reused_original_starts':20,'total_shared_starts':200,
      'new_episodes':27180,'reused_episodes':3020,'total_episodes':30200,'source_result_SHA256':sha(R/'original20-results.json'),
      'global_new_selection_seal_SHA256':sha(R/'attempt/all-new-selection-seal.json'),'selection_sealed_at_UTC':seal['at_UTC'],
      'evaluation_started_at_UTC':started['at_UTC'],'new_fit_counts':{k:sum(e['actual_fit_counts'][k] for e in new) for k in ['GP','XGBoost']},
      'new_assays':0,'new_MD_or_docking':0,'reserved_values_opened':0,'new_resampling':0,
      'limits':p['limits']+['Original20 reused;180 new independent random seeds on the same exposed160 records. Not200 biological replicates.',
                          'MC standard errors describe random-start simulation precision only. No new significance/success thresholds.',
                          'One complete fixed extension; no favorable partial summary.']}
 with (R/'attempt/final-results.json').open('x') as f:json.dump(result,f,allow_nan=False);f.write('\n')
 (R/'attempt/evaluation-complete.json').write_text(json.dumps({'status':'PASS','at_UTC':result['at_UTC'],'final_result_SHA256':sha(R/'attempt/final-results.json'),
       'all30200_episodes':True,'all151_summaries':True,'all280_contrasts':True,'global_seal_before_evaluation':seal['at_UTC']<started['at_UTC']},indent=2)+'\n')
 print('ONE_TIME_EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
