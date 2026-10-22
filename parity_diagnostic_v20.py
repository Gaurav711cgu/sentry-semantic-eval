"""One consumed selected-adapter parity diagnostic; no training or final inference."""
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

ROOT=Path(__file__).resolve().parent


def payload_for(checkpoint):
    rows=checkpoint['records'][:4]
    return {'contents':[r['prefix_content'] for r in rows],
            'parity_contents':[r['prefix_content'] for r in rows],
            'expected_probabilities':[r['positive_probability'] for r in rows],
            'adapter_file':checkpoint.get('adapter_file')}


def stage_record(name, result):
    return {'name':name,'worker':result,
            'parity_passed':result.get('passed') is True and result.get('parity',{}).get('passed') is True,
            'final_inference_performed':False}


def main():
    protocol=core.checked_json((ROOT/'parity_protocol_v20.json').read_bytes())
    output={'experiment':protocol['experiment_id'],'final_inference_performed':False,
            'training_performed':False,'stages':{},'protocol_sha256':core.sha(ROOT/'parity_protocol_v20.json')}
    temp=None
    try:
        if os.environ.get('GITHUB_REPOSITORY')!='Gaurav711cgu/sentry-semantic-eval':
            raise ValueError('existing public runner required')
        core.verify_sources(ROOT,protocol['source_sha256'])
        expected={'experiment_id':protocol['experiment_id'],'source_head':os.environ['GITHUB_SHA'],
                  'protocol_sha256':output['protocol_sha256'],'scoring_computations_max':8,
                  'run_id':int(os.environ['GITHUB_RUN_ID']),'attempt':1,'status':'consumed_before_execution'}
        if os.environ['GITHUB_RUN_ATTEMPT']!='1':raise ValueError('attempt1 only')
        until=time.monotonic()+360
        while True:
            try:
                with urlopen('https://raw.githubusercontent.com/Gaurav711cgu/sentry-semantic-eval/main/grants-parity-v20/probe.json',timeout=15) as response:
                    grant=core.checked_json(response.read(65537))
                if grant!=expected:raise ValueError('exact consumed diagnostic grant required')
                output['grant']=grant
                break
            except HTTPError as error:
                if error.code!=404 or time.monotonic()>until:raise
                time.sleep(5)
        packet=core.checked_json((ROOT/'parity-input-v20.json').read_bytes())
        checkpoint=packet['checkpoint']
        sample=[{'prefix_content':r['prefix_content']} for r in checkpoint['records']]
        report={'training':{'checkpoints':{'step64':checkpoint}},'retained_adapters':{'step64':packet['retained_adapter']}}
        temp=tempfile.TemporaryDirectory(prefix='sentry-parity-v20-');work=Path(temp.name);work.chmod(0o755)
        public=work/'public';public.mkdir(mode=0o755)
        out=work/'output';out.mkdir();out.chmod(0o777)
        adapters=out/'adapters';adapters.mkdir();adapters.chmod(0o755)
        for name in ['pair_v10.py','lora_v10.py','worker_v10.py']:
            (public/name).write_bytes((ROOT/name).read_bytes());(public/name).chmod(0o444)
        for name in ['step64']:
            item=report['retained_adapters'][name];raw=base64.b64decode(item['base64'],validate=True)
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('adapter integrity failure')
            dest=adapters/report['training']['checkpoints'][name]['adapter_file'];dest.write_bytes(raw);dest.chmod(0o444)
        model=work/'model';core.download_model(model,protocol['model']['files'])
        for name in ['step64']:
            # Fit probes are training-corpus diagnostics; parity sentinels are retained development inputs.
            value=payload_for(report['training']['checkpoints'][name])
            value['contents']=[r['prefix_content'] for r in sample]
            result=core.run_worker(work,model,'score',value,'diagnostic-'+name,400)
            output['stages'][name]=stage_record(name,result)
            if result.get('passed') is True and result.get('parity',{}).get('passed') is True:
                scores={r['prefix_content']:r['positive_probability'] for r in result['records']}
                if set(scores)!={r['prefix_content'] for r in sample}:raise ValueError('exact corpus probe coverage required')
                output['stages'][name]['verified_score_count']=len(scores)
            print('FIT_STAGE '+json.dumps({'name':name,'parity_passed':output['stages'][name]['parity_passed'],'fit_correct':output['stages'][name].get('fit',{}).get('correct')}),flush=True)
        if not all(s['parity_passed'] and s.get('verified_score_count')==4 for s in output['stages'].values()):
            raise ValueError('all retained adapters must pass scoring and parity')
        output['execution_status']='complete'
    except BaseException as error:
        output.update(execution_status='failed',failure_type=type(error).__name__,failure=str(error)[:2000])
    finally:
        output['owned_worker_stopped']=all(p.poll() is not None for p in core.OWNED_PROCESSES)
        if temp is not None:temp.cleanup()
        output['temporary_artifacts_removed']=temp is None or not Path(temp.name).exists()
        raw=json.dumps(output,indent=2,allow_nan=False).encode();(ROOT/'result-parity-v20.json').write_bytes(raw)
        encoded=base64.b64encode(gzip.compress(raw)).decode();print('FIT_REPORT_SHA256 '+hashlib.sha256(raw).hexdigest(),flush=True)
        for i in range(0,len(encoded),3000):print('FIT_REPORT_CHUNK '+encoded[i:i+3000],flush=True)
    return 0 if output['execution_status']=='complete' and output['owned_worker_stopped'] and output['temporary_artifacts_removed'] else 1


if __name__=='__main__':raise SystemExit(main())
