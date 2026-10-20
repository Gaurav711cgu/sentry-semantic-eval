"""Model-free audit of the exact V17 sampler and retained development scores."""
import json
from collections import Counter
from data_v17 import training_curriculum, build_split
from sampling_v14 import stratified_indices
from pair_v10 import pair_inputs, choose_target


def exposure(rows, families, plan, steps):
    if not 1 <= steps <= len(plan):
        raise ValueError("checkpoint outside retained plan")
    indices = [i for step in plan[:steps] for batch in step for i in batch]
    counts = Counter((rows[i]['label'], families[i]) for i in indices)
    return {'steps': steps, 'presentations': len(indices),
            'unique_pairs': len(set(indices)), 'available_pairs': len(rows),
            'class_family': [{'label': label, 'family': family,
                'presentations': count,
                'available': sum(r['label'] == label and f == family for r, f in zip(rows, families)),
                'unique_pairs': len({i for i in indices if rows[i]['label'] == label and families[i] == family})}
                for (label, family), count in sorted(counts.items())]}


def error_kind(expected, predicted, positives):
    if expected == predicted:
        return 'correct'
    if expected is None:
        return 'false_authorization'
    if predicted is None:
        return 'multiple_positive_abstention' if positives > 1 else 'missed_authorization'
    return 'wrong_target'


def main():
    from pathlib import Path
    root = Path(__file__).resolve().parent
    report = json.loads((root / 'result-v17.json').read_text())
    rows, families = training_curriculum()
    plan = stratified_indices(rows, families)
    result = {'scope': 'Retained development only; zero inference; groups are correlated views',
              'exposure': [exposure(rows, families, plan, n) for n in [8, 16, 32, 64, 128]],
              'checkpoints': {}}
    for name, threshold in [('initial', .8), ('step32', .8), ('step128', .8)]:
        records = report['training']['checkpoints'][name]['records']
        scores = {r['prefix_content']: r['positive_probability'] for r in records}
        errors, counts = [], Counter()
        seen = set()
        for case in build_split('dev'):
            if case['group_id'] in seen:
                continue
            seen.add(case['group_id'])
            pairs = pair_inputs(case['request'], [o['id'] for o in case['orders']])
            scored = [{'candidate': r['candidate'], 'positive_probability': scores[r['prefix_content']]} for r in pairs]
            predicted = choose_target(scored, threshold)
            positives = sum(r['positive_probability'] >= threshold for r in scored)
            kind = error_kind(case['expected_choice'], predicted, positives)
            counts[kind] += 1
            if kind != 'correct':
                errors.append({'group_id': case['group_id'], 'family': case['kind'],
                    'request': case['request'], 'expected': case['expected_choice'],
                    'predicted': predicted, 'error': kind, 'positive_candidates': positives})
        result['checkpoints'][name] = {'threshold': threshold, 'scenario_counts': dict(counts), 'errors': errors}
    path = root.parent / 'sentry-trained-v17' / 'EXPOSURE_ERROR_AUDIT.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(path)
    for name, value in result['checkpoints'].items():
        print(name, value['scenario_counts'])


if __name__ == '__main__':
    main()
