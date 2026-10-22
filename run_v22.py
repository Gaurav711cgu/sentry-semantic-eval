"""One consumed public CPU NLI experiment, with a gated frozen confirmation."""
import base64
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import run_v19 as core

ROOT = Path(__file__).resolve().parent


def wait_grant(protocol):
    expected = {'experiment_id': protocol['experiment_id'], 'source_head': os.environ['GITHUB_SHA'],
                'protocol_sha256': core.sha(ROOT / 'protocol_v22.json'), 'optimizer_steps_max': 128,
                'scoring_computations_max': protocol['scoring_computations_max'],
                'run_id': int(os.environ['GITHUB_RUN_ID']), 'attempt': 1, 'status': 'consumed_before_execution'}
    if os.environ['GITHUB_RUN_ATTEMPT'] != '1':
        raise ValueError('attempt1 only; consumed attempts cannot rerun')
    until = time.monotonic() + 360
    while True:
        try:
            with urlopen('https://raw.githubusercontent.com/Gaurav711cgu/sentry-semantic-eval/main/grants-v22/train.json', timeout=15) as response:
                grant = core.checked_json(response.read(65537))
            if grant != expected or any(type(grant[k]) is not type(v) for k, v in expected.items()):
                raise ValueError('exact consumed run/source/protocol/budget grant required')
            return grant
        except HTTPError as error:
            if error.code != 404 or time.monotonic() >= until:
                raise
            time.sleep(5)


def run_worker(work, model, stage, value, timeout):
    public, output = work / 'public', work / 'output'
    source = public / (stage + '.json')
    source.write_text(json.dumps(value, allow_nan=False)); source.chmod(0o444)
    destination = output / (stage + '.json')
    command = ['sudo', '-n', 'unshare', '--net', '--pid', '--fork', '--mount-proc', '--kill-child',
               'setpriv', '--reuid=65534', '--regid=65534', '--clear-groups', '--no-new-privs',
               'env', '-i', 'PATH=/usr/bin:/bin', 'HOME=/tmp', 'HF_HUB_OFFLINE=1',
               'TRANSFORMERS_OFFLINE=1', 'TOKENIZERS_PARALLELISM=false', 'OMP_NUM_THREADS=4',
               'MKL_NUM_THREADS=4', os.environ['PROBE_PYTHON'], str(public / 'worker_v22.py'),
               '--stage', stage, '--model', str(model), '--input', str(source), '--output', str(destination),
               '--adapter-dir', str(output / 'adapters')]
    process = subprocess.Popen(command, cwd=public, stdin=subprocess.DEVNULL, start_new_session=True)
    core.OWNED_PROCESSES.append(process)
    failure = None
    try:
        code = process.wait(timeout=timeout)
    except BaseException as error:
        failure = str(error)
        subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', '-' + str(process.pid)], check=False)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', '-' + str(process.pid)], check=False)
            process.wait(timeout=10)
        code = process.returncode
    result = {'passed': False}
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size > 2 * 1024**2:
            raise ValueError('bounded regular worker result required')
        result = core.checked_json(destination.read_bytes())
    if code != 0 or failure:
        result.update(passed=False, worker_exit_code=code, parent_failure=failure)
    return result


def stage_ok(result, protocol):
    return (result.get('passed') is True and result.get('rss_peak_bytes', float('inf')) <= 8 * 1024**3
            and all(result.get('metadata', {}).get(k) == v for k, v in protocol['runtime'].items()))


def main():
    from contrast_v19 import curriculum, development, paired_indices
    from confirmation_v22 import cases as confirmation
    from evaluate_v10 import select_candidate, score_cases
    report = {'execution_status': 'not_started', 'measured_demo_qualified': False,
              'independent_final_qualified': False, 'references': 'AI-authored synthetic',
              'final_inference_performed': False, 'training': {}, 'retained_adapters': {}}
    temporary = None
    try:
        if os.environ.get('GITHUB_REPOSITORY') != 'Gaurav711cgu/sentry-semantic-eval':
            raise ValueError('existing public CPU repository required')
        p = core.checked_json((ROOT / 'protocol_v22.json').read_bytes())
        core.verify_sources(ROOT, p['source_sha256'])
        report.update(experiment=p['experiment_id'], protocol_sha256=core.sha(ROOT / 'protocol_v22.json'),
                      model=p['model'], grant=wait_grant(p), execution_status='running')
        rows, families, _ = curriculum()
        dev, final = development(), confirmation()
        payload = {'training_pairs': rows, 'training_families': families, 'training_plan': paired_indices(rows, families),
                   'dev_contents': core.unique_contents(dev)}
        for name, value in [('training', payload), ('development', dev), ('confirmation', final)]:
            digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if digest != p['dataset_sha256'][name]:
                raise ValueError('frozen dataset identity changed: ' + name)
        temporary = tempfile.TemporaryDirectory(prefix='sentry-nli-v22-')
        work = Path(temporary.name); work.chmod(0o755)
        public = work / 'public'; public.mkdir(mode=0o755)
        output = work / 'output'; output.mkdir(); output.chmod(0o777)
        for name in ['worker_v22.py', 'nli_v22.py', 'pair_v10.py', 'worker_v10.py', 'run_v19.py']:
            (public / name).write_bytes((ROOT / name).read_bytes()); (public / name).chmod(0o444)
        model = work / 'model'; core.download_model(model, p['model']['files'])
        train = run_worker(work, model, 'train', payload, 3100)
        report['training'] = train
        if not stage_ok(train, p) or train.get('completed_optimizer_steps') != 128 or train.get('scoring_requests') != 780:
            raise ValueError('training/runtime/computation completion gate failed')
        ranked = select_candidate(dev, {n: {k: c[k] for k in ['step', 'records']} for n, c in train['checkpoints'].items()}, p['thresholds'])
        admitted = core.admission_choice(ranked)
        report['ranking'] = [{k: c[k] for k in ['name', 'step', 'threshold']} | {
            'raw_accuracy': c['metrics']['raw_accuracy'], 'eligible_recall': c['metrics']['eligible_recall']}
            for c in ranked['candidates']]
        report['best_development'] = ranked['candidates'][0]
        report['selection'] = admitted
        report['selection_sha256'] = hashlib.sha256(json.dumps(admitted, sort_keys=True, allow_nan=False).encode()).hexdigest()
        print('V22_SELECTION_FROZEN ' + json.dumps({'name': admitted['name'] if admitted else None, 'sha256': report['selection_sha256']}), flush=True)
        report['semantic_status'] = 'development_rejected'
        if admitted:
            if not admitted['metrics']['draft_latency_gate_passed']:
                raise ValueError('development draft latency exceeded10 seconds')
            item = train['checkpoints'][admitted['name']]
            sentinels = item['records'][:4]
            report['final_inference_performed'] = True
            scored = run_worker(work, model, 'score', {
                'contents': core.unique_contents(final), 'adapter_file': item['adapter_file'],
                'parity_contents': [r['prefix_content'] for r in sentinels],
                'expected_probabilities': [r['positive_probability'] for r in sentinels]}, 600)
            report['confirmation_worker'] = scored
            if not stage_ok(scored, p) or scored.get('parity', {}).get('passed') is not True or scored.get('adapter_sha256') != item['adapter_sha256']:
                raise ValueError('confirmation runtime/parity/identity gate failed')
            if scored.get('scoring_requests') != 53:
                raise ValueError('confirmation computation count mismatch')
            report['confirmation'] = score_cases(final, scored['records'], admitted['threshold'])
            metrics = report['confirmation']
            passed = metrics['raw_accuracy'] >= .9 and metrics['eligible_recall'] >= .9 and metrics['draft_latency_gate_passed'] and metrics['consistent_groups'] == metrics['permutation_groups']
            report['semantic_status'] = 'confirmation_raw_passed' if passed else 'confirmation_raw_failed'
        report['execution_status'] = 'complete'
    except BaseException as error:
        report.update(execution_status='failed', failure_type=type(error).__name__, failure=str(error)[:2000])
    finally:
        if temporary:
            for name, item in report['training'].get('checkpoints', {}).items():
                try:
                    filename = item['adapter_file']
                    if Path(filename).name != filename:
                        raise ValueError('unsafe checkpoint filename')
                    record = core.retained_file(Path(temporary.name) / 'output/adapters' / filename, 4_000_000)
                    if record['sha256'] != item['adapter_sha256']:
                        raise ValueError('adapter retention hash mismatch')
                    report['retained_adapters'][name] = record
                except (OSError, ValueError, KeyError) as error:
                    report.setdefault('artifact_retention_errors', []).append(str(error))
        report['owned_worker_stopped'] = all(p.poll() is not None for p in core.OWNED_PROCESSES)
        if temporary:
            temporary.cleanup()
        report['temporary_artifacts_removed'] = temporary is None or not Path(temporary.name).exists()
        raw = json.dumps(report, indent=2, allow_nan=False).encode()
        (ROOT / 'result-v22.json').write_bytes(raw)
        encoded = base64.b64encode(gzip.compress(raw)).decode()
        print('V22_REPORT_SHA256 ' + hashlib.sha256(raw).hexdigest(), flush=True)
        for i in range(0, len(encoded), 3000):
            print('V22_REPORT_CHUNK ' + encoded[i:i + 3000], flush=True)
        print(json.dumps({k: report.get(k) for k in ['execution_status', 'semantic_status', 'failure']}), flush=True)
    return 0 if report['execution_status'] == 'complete' and report['owned_worker_stopped'] and report['temporary_artifacts_removed'] and not report.get('artifact_retention_errors') else 1


if __name__ == '__main__':
    raise SystemExit(main())
