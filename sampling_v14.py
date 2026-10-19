"""Deterministic training-only class/family balancing; no datasets or inference."""
import random
from collections import Counter


def pools_for(rows, families):
    if type(rows) is not list or type(families) is not list or len(rows) != len(families) or not rows:
        raise ValueError('aligned nonempty training families required')
    pools = {0: {}, 1: {}}
    for i, (row, family) in enumerate(zip(rows, families)):
        if type(row) is not dict or type(row.get('label')) is not int or row['label'] not in pools:
            raise ValueError('binary training class required')
        if type(family) is not str or not family or len(family) > 64:
            raise ValueError('bounded training family required')
        pools[row['label']].setdefault(family, []).append(i)
    if not all(pools.values()):
        raise ValueError('both training classes required')
    return pools


def validate_plan(rows, families, plan, *, steps=128):
    pools = pools_for(rows, families)
    if type(steps) is not int or steps < 1 or type(plan) is not list or len(plan) != steps:
        raise ValueError('exact bounded training steps required')
    counts = {label: Counter({family: 0 for family in pools[label]}) for label in pools}
    cycles = {(label, family): set() for label in pools for family in pools[label]}
    for step in plan:
        if type(step) is not list or len(step) != 4:
            raise ValueError('four accumulation microbatches required')
        for batch in step:
            if type(batch) is not list or len(batch) != 2 or any(type(i) is not int or not 0 <= i < len(rows) for i in batch):
                raise ValueError('two in-range training indices required')
            if sorted(rows[i]['label'] for i in batch) != [0, 1]:
                raise ValueError('class-balanced training microbatch required')
            for i in batch:
                label, family = rows[i]['label'], families[i]
                counts[label][family] += 1
                key = (label, family)
                if i in cycles[key]:
                    raise ValueError('training family repeated before cycle exhaustion')
                cycles[key].add(i)
                if len(cycles[key]) == len(pools[label][family]):
                    cycles[key].clear()
    for counter in counts.values():
        if max(counter.values()) - min(counter.values()) > 1:
            raise ValueError('training family starvation')
    return plan


def stratified_indices(rows, families, seed=1019, steps=128):
    pools = pools_for(rows, families)
    if type(seed) is not int or type(steps) is not int or not 1 <= steps <= 128:
        raise ValueError('bounded deterministic training plan required')
    rng = random.Random(seed)
    names = {label: sorted(pools[label]) for label in pools}
    for values in names.values():
        rng.shuffle(values)
    queues, cursors, family_cursors = {}, {}, {0: 0, 1: 0}
    for label in pools:
        for family, values in pools[label].items():
            key = label, family
            queues[key] = list(values)
            rng.shuffle(queues[key])
            cursors[key] = 0
    plan = []
    for _ in range(steps):
        step = []
        for _ in range(4):
            batch = []
            for label in [0, 1]:
                family = names[label][family_cursors[label] % len(names[label])]
                family_cursors[label] += 1
                key = label, family
                if cursors[key] == len(queues[key]):
                    rng.shuffle(queues[key])
                    cursors[key] = 0
                batch.append(queues[key][cursors[key]])
                cursors[key] += 1
            rng.shuffle(batch)
            step.append(batch)
        plan.append(step)
    return validate_plan(rows, families, plan, steps=steps)
