"""Score retained passing adapter without retraining; retain every stage response."""
import base64
import gzip
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
import run_v14 as core
from evaluate_v10 import score_cases,select_candidate
from final_v15 import build_split
ROOT=Path(__file__).resolve().parent


def retain_stage(report,name,result,runtime=None):
    report['stages'][name]=result
    if result.get('passed') is not True or result.get('parity',{}).get('passed') is not True:
        raise ValueError('retained stage '+name+' failed parity/completion')
    if runtime and any(result.get('metadata',{}).get(k)!=v for k,v in runtime.items()):
        raise ValueError('retained stage runtime drift')


def main():
    protocol=core.checked_json((ROOT/'protocol_v15.json').read_bytes())
    report={'experiment':protocol['experiment_id'],'protocol_sha256':core.sha(ROOT/'protocol_v15.json'),'execution_status':'not_started','stages':{},'final':{},'final_inference_started':False,'training_performed':False,'independent_final_qualified':False,'measured_demo_qualified':False}
    temp=None
    try:
        if os.environ.get('GITHUB_REPOSITORY')!='Gaurav711cgu/sentry-semantic-eval':raise ValueError('existing public runner required')
        core.verify_sources(ROOT,protocol['source_sha256'])
        grant_expected={'experiment_id':protocol['experiment_id'],'source_head':os.environ['GITHUB_SHA'],'protocol_sha256':report['protocol_sha256'],'scoring_computations_max':protocol['scoring_computations_max'],'run_id':int(os.environ['GITHUB_RUN_ID']),'attempt':1,'status':'consumed_before_execution'}
        if os.environ['GITHUB_RUN_ATTEMPT']!='1':raise ValueError('attempt1 only')
        until=time.monotonic()+360
        while True:
            try:
                with urlopen('https://raw.githubusercontent.com/Gaurav711cgu/sentry-semantic-eval/main/grants-v15/evaluate.json',timeout=15) as response:grant=core.checked_json(response.read(65537))
                if grant!=grant_expected:raise ValueError('exact consumed evaluation grant required')
                report['grant']=grant;break
            except HTTPError as error:
                if error.code!=404 or time.monotonic()>until:raise
                time.sleep(5)
        old=core.checked_json((ROOT/'result-v14.json').read_bytes());selected=old['selection']
        if selected['name']!='step8' or selected['threshold']!=.8 or selected['metrics']['raw_accuracy']<.9 or selected['metrics']['eligible_recall']<.9:raise ValueError('frozen admitted adapter required')
        report['selection']={'name':'step8','step':8,'threshold':.8,'selection_split':'v14 exposed development','source_selection_sha256':old['selection_sha256']}
        report['initial_threshold']=selected['initial_threshold']
        temp=tempfile.TemporaryDirectory(prefix='sentry-saved-v15-');work=Path(temp.name);work.chmod(0o755);public=work/'public';public.mkdir(mode=0o755);out=work/'output';out.mkdir();out.chmod(0o777);adapters=out/'adapters';adapters.mkdir();adapters.chmod(0o755)
        for name in ['pair_v10.py','lora_v10.py','worker_v10.py']:
            (public/name).write_bytes((ROOT/name).read_bytes());(public/name).chmod(0o444)
        for name in ['initial','step8']:
            item=old['retained_adapters'][name];raw=base64.b64decode(item['base64'],validate=True)
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('saved adapter drift')
            dest=adapters/old['training']['checkpoints'][name]['adapter_file'];dest.write_bytes(raw);dest.chmod(0o444)
        model=work/'model';core.download_model(model,protocol['model']['files'])
        # Reference worker is intentionally allowed only the50 exposed development inputs.
        (public/'worker_v10.py').chmod(0o644)
        worker_bytes=(public/'worker_v10.py').read_bytes();(public/'worker_v10.py').write_bytes((ROOT/'reference_worker_v15.py').read_bytes())
        (public/'original_worker.py').write_bytes(worker_bytes)
        # The reference module imports exact old helpers through its own separate name.
        (public/'worker_v10.py').write_text((public/'worker_v10.py').read_text().replace('from worker_v10 import','from original_worker import'))
        reference=core.run_worker(work,model,'score',{'dev_contents':core.unique_contents(build_split('dev'))},'reference',300)
        report['stages']['base_development']=reference
        (public/'worker_v10.py').write_bytes(worker_bytes)
        if reference.get('passed') is not True:raise ValueError('fresh base development reference failed')
        if any(reference.get('metadata',{}).get(k)!=v for k,v in protocol['runtime'].items()):raise ValueError('base reference runtime drift')
        base_selection=select_candidate(build_split('dev'),{'base':{'step':0,'records':reference['records']}},protocol['thresholds'])
        report['base_threshold']=base_selection['threshold']
        old_base={r['prefix_content']:r['positive_probability'] for r in old['training']['checkpoints']['base']['records']}
        report['old_base_cache_max_absolute_difference']=max(abs(r['positive_probability']-old_base[r['prefix_content']]) for r in reference['records'])
        report['selection_sha256']=hashlib.sha256(json.dumps(report['selection'],sort_keys=True).encode()).hexdigest()
        final=build_split('final');contents=core.unique_contents(final)
        for name in ['base','initial','step8']:
            checkpoint={'adapter_file':None,'records':reference['records']} if name=='base' else old['training']['checkpoints'][name]
            sentinel=checkpoint['records'][:4]
            payload={'contents':contents,'adapter_file':checkpoint.get('adapter_file'),'parity_contents':[r['prefix_content'] for r in sentinel],'expected_probabilities':[r['positive_probability'] for r in sentinel]}
            report['final_inference_started']=True
            result=core.run_worker(work,model,'score',payload,'final-'+name,600)
            retain_stage(report,name,result,protocol['runtime'])
            threshold=report['base_threshold'] if name=='base' else report['initial_threshold'] if name=='initial' else .8
            result['metrics']=score_cases(final,result['records'],threshold);report['final'][name]=result
        report.update(execution_status='complete',semantic_status='pending_actual_controller_audit')
    except BaseException as error:report.update(execution_status='failed',failure_type=type(error).__name__,failure=str(error)[:2000])
    finally:
        report['owned_worker_stopped']=all(p.poll() is not None for p in core.OWNED_PROCESSES)
        if temp is not None:temp.cleanup()
        report['temporary_artifacts_removed']=temp is None or not Path(temp.name).exists()
        raw=json.dumps(report,indent=2,allow_nan=False).encode();(ROOT/'result-v15.json').write_bytes(raw);encoded=base64.b64encode(gzip.compress(raw)).decode();print('V15_REPORT_SHA256 '+hashlib.sha256(raw).hexdigest(),flush=True)
        for i in range(0,len(encoded),3000):print('V15_REPORT_CHUNK '+encoded[i:i+3000],flush=True)
    return 0 if report['execution_status']=='complete' and report['temporary_artifacts_removed'] else 1

if __name__=='__main__':raise SystemExit(main())
