"""Model-free coverage accounting and fixed stratified corpus-fit probe planning."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def coverage(rows,families,plan,checkpoint):
    if len(rows)!=len(families) or type(checkpoint) is not int or not 1<=checkpoint<=len(plan):
        raise ValueError('aligned corpus and in-range checkpoint required')
    indices=[i for step in plan[:checkpoint] for batch in step for i in batch]
    if any(type(i) is not int or not 0<=i<len(rows) for i in indices):raise ValueError('training index out of range')
    counts=Counter(families[i] for i in indices)
    return {'checkpoint':checkpoint,'presentations':len(indices),'unique_pairs':len(set(indices)),
            'corpus_pairs':len(rows),'unseen_pairs':len(rows)-len(set(indices)),
            'families':{name:{'available':families.count(name),'presentations':counts[name],
                              'unique_pairs':len({i for i in indices if families[i]==name})} for name in sorted(set(families))}}


def metric_summary(metrics):
    return {'correlated_views':metrics['cases'],'scenario_groups':metrics['permutation_groups'],
            'raw_correct':metrics['raw_exact_correct'],'eligible_hits':metrics['eligible_raw_target_hits'],
            'eligible_views':metrics['eligible_targets'],
            'wrong_groups':[g['group_id'] for g in metrics['group_results'] if not g['exact']]}


def probe_sample(rows,families,plan):
    seen={i for step in plan[:8] for batch in step for i in batch}
    pools={}
    for i,(row,family) in enumerate(zip(rows,families)):
        pools.setdefault((row['label'],family,i in seen),[]).append(i)
    rng=random.Random(20261020);result=[]
    for key,indices in sorted(pools.items()):
        for i in rng.sample(indices,min(4,len(indices))):
            result.append({**rows[i],'family':families[i],'presented_before_step8':i in seen,'training_index':i})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from curriculum_v14 import training_curriculum
    from sampling_v14 import stratified_indices
    rows,families=training_curriculum();plan=stratified_indices(rows,families);root=Path(__file__).resolve().parent
    report=json.loads((root/'result-v14.json').read_text());sample=probe_sample(rows,families,plan)
    value={'scope':'Model-free diagnosis; no threshold tuning, model inference or new final claim',
           'checkpoint_coverage':[coverage(rows,families,plan,n) for n in [8,16,32,64,128]],
           'development':{name:{'threshold':c['threshold'],'metrics':metric_summary(c['metrics'])} for name in ['initial','step8','step16','step32','step64','step128'] for c in [next(c for c in report['selection']['candidates'] if c['name']==name)]},
           'corpus_fit_sample':sample,'corpus_fit_sample_sha256':hashlib.sha256(json.dumps(sample,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
           'corpus_fit_sample_count':len(sample),'probe_scoring_budget_max':2*(len(sample)+4),
           'interpretation':'Coverage is measured; causal attribution and corpus fit require new measurements. Sample membership does not enter model inputs.'}
    args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:value[k] for k in ['corpus_fit_sample_count','corpus_fit_sample_sha256','probe_scoring_budget_max']}))

if __name__=='__main__':main()
