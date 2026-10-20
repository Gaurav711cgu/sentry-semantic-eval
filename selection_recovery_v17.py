"""Lossless development-ID compatibility repair; reuse scored prefixes, not inference."""
import argparse
import hashlib
import json
from pathlib import Path
from data_v17 import build_split
from evaluate_v10 import select_candidate
from run_v17 import admission_choice,checked_json,verify_sources,validate_replay_result


def development_cases(cases):
    result=[]
    for case in cases:
        if any(not case[key].startswith('dev17-') for key in ['id','group_id']):
            raise ValueError('only original v17 development identities may be projected')
        result.append({**case,'id':'dev-'+case['id'][3:],'group_id':'dev-'+case['group_id'][3:]})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=Path(__file__).resolve().parent
    raw=args.report.read_bytes()
    if len(raw)>32*1024**2:raise ValueError('bounded original report required')
    report=checked_json(raw);protocol=checked_json((root/'protocol_v17.json').read_bytes())
    verify_sources(root,protocol['source_sha256'])
    if report['protocol_sha256']!=hashlib.sha256((root/'protocol_v17.json').read_bytes()).hexdigest():raise ValueError('original protocol identity drift')
    if report.get('failure_message')!='explicit development case and group identities required':raise ValueError('only the named metadata failure may be recovered')
    if not report['owned_worker_stopped'] or not report['temporary_artifacts_removed']:raise ValueError('original cleanup required')
    validate_replay_result(report['training'],protocol)
    checkpoints=report['training']['checkpoints'];cases=development_cases(build_split('dev'))
    result=select_candidate(cases,{name:{key:item[key] for key in ['step','records']} for name,item in checkpoints.items()},protocol['thresholds'])
    admitted=admission_choice(result)
    value={'scope':'Post-run lossless identity repair, original failure preserved; zero new inference',
           'original_report_sha256':hashlib.sha256(raw).hexdigest(),'protocol_sha256':report['protocol_sha256'],
           'transformation':'dev17-* case/group IDs become dev-17-*; requests, snapshots, references, records and thresholds unchanged',
           'final_inference_performed':False,'admitted':admitted is not None,
           'selected':admitted,'ranked_development':result,'measured_demo_qualified':False,'independent_final_qualified':False}
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for name in ['base','initial','step8','step16','step32','step64','step128']:
        x=next(x for x in result['candidates'] if x['name']==name);m=x['metrics']
        print(json.dumps({'checkpoint':name,'threshold':x['threshold'],'raw_correct':m['raw_exact_correct'],'views':m['cases'],'eligible_hits':m['eligible_raw_target_hits'],'eligible_views':m['eligible_targets']}))
    print('DEVELOPMENT_ADMITTED',value['admitted'])

if __name__=='__main__':main()
