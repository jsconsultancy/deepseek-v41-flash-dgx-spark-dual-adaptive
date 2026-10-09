"""Synthetic domain-shift fixtures; answers come from the supplied records.

Medical text is deliberately fictional and is never graded as clinical advice.
The warm workload mixes CS explanations, executable Python and fictional news.
"""
from prefill_expert_probes import probe


CS_NOTES = (
    'A hash table maps keys to buckets. Collision chains and load factors affect '
    'lookup cost. Resizing rebuilds the bucket array; amortized analysis spreads '
    'that cost across insertions.',
    'A directed graph can be traversed with a queue for breadth-first search. '
    'A visited set prevents repeated expansion. A predecessor map reconstructs '
    'the shortest path in an unweighted graph.',
    'A database transaction groups changes. Isolation controls which concurrent '
    'writes a reader can observe. A query planner compares scans, index lookups '
    'and joins using estimates of row counts and selectivity.',
    'A compiler parses source into a syntax tree, resolves names and lowers '
    'operations into an intermediate representation. Register allocation and '
    'instruction scheduling operate on control-flow and data-flow information.',
    'A network protocol uses sequence numbers, acknowledgements and retransmission '
    'to manage lost packets. Congestion control changes the sending window; '
    'latency and throughput respond differently to queue growth.',
    'An operating system schedules runnable threads, maps virtual pages and '
    'handles page faults. Locks and condition variables coordinate shared state; '
    'deadlock analysis considers resource ownership and waiting cycles.',
)


def fit(make, target, render):
    n, best = max(8, target // 60), None
    for _ in range(6):
        row = make(n)
        ids = render(row['body'])
        distance = abs(len(ids) - target)
        if best is None or distance < best[0]:
            best = distance, row, ids
        n = max(8, int(n * target / len(ids)))
    return best[1], best[2]


def warm_probe(domain, seed, target, render):
    def make(n):
        if domain == 'cs':
            records = [f'Lecture {seed}-{i}: {CS_NOTES[(i + seed) % len(CS_NOTES)]}'
                       for i in range(n)]
        elif domain == 'python':
            records = [f'def transform_{seed}_{i}(items):\n'
                       f'    values = [x * {i % 7 + 1} for x in items if x % 3 != 0]\n'
                       f'    return sorted(set(values))\n' for i in range(n)]
        else:
            records = [f'Fictional dispatch {seed}-{i}, district {i % 19}: '
                       f'The council debated its annual budget and transport policy. '
                       f'A survey counted {120 + i % 81} households. Businesses discussed '
                       f'employment, exports and energy costs. The committee will meet on '
                       f'day {i % 28 + 1}; officials requested public comment on the proposal.'
                       for i in range(n)]
        return probe(f'warm-{domain}-{seed}', 'json',
                     f'Read these {domain} notes:\n' + '\n'.join(records) +
                     '\nReturn only JSON {"ready":true}.', {'ready': True}, 32)
    return fit(make, target, render)


def medical_probe(kind, seed, target, render):
    """Chart extraction, invented trial rules, and arithmetic across visits."""
    key = f'PATIENT-{seed}'
    def make(n):
        records = [f'Synthetic chart C-{seed}-{i}: age {25 + i % 55}; complaint '
                   f'{("headache", "fatigue", "cough", "joint discomfort")[i % 4]}; '
                   f'visit day {i % 28 + 1}; history reviewed; imaging report pending; '
                   f'fictional assay Z value {3 + i % 17}; study participation '
                   f'{("new", "previous")[i % 2]}. Follow-up documentation requested.'
                   for i in range(n)]
        if kind == 'chart':
            records.insert(n // 2, f'Synthetic chart {key}: age 47; complaint fatigue; '
                           'fictional assay Z value 11; imaging status reviewed; '
                           'study participation new.')
            question = (f'For {key}, extract exactly age, complaint, assay_z, and imaging_status. '
                        'Return only JSON, using an integer for age and assay_z.')
            expected = dict(age=47, complaint='fatigue', assay_z=11, imaging_status='reviewed')
        elif kind == 'trial':
            subjects = [(45, 8, 'new'), (39, 8, 'new'), (70, 12, 'new'),
                        (55, 13, 'new'), (62, 9, 'previous'), (40, 5, 'new')]
            for j, (age, assay, previous) in reversed(list(enumerate(subjects))):
                records.insert(int(n * (j + 1) / 7), f'Trial record {key}-{j}: age {age}; '
                               f'fictional assay Z value {assay}; study participation {previous}.')
            question = (f'This invented trial accepts age 40 through 70 inclusive, assay Z 5 '
                        f'through 12 inclusive, and new participation. Evaluate only {key}-0 '
                        f'through {key}-5. Return only JSON with key eligible containing accepted '
                        'patient IDs in ascending numeric order. These are fictional rules.')
            expected = dict(eligible=[f'{key}-{j}' for j in (0, 2, 5)])
        else:
            records.insert(n // 5, f'Laboratory visit {key} day 2: fictional assay Z value 18.')
            records.insert(4 * n // 5, f'Laboratory visit {key} day 19: fictional assay Z value 11.')
            question = (f'For {key}, order the two laboratory visits by day. Return only JSON '
                        'with keys first_value, last_value, change (last minus first), '
                        'and direction (increased, decreased, or unchanged).')
            expected = dict(first_value=18, last_value=11, change=-7, direction='decreased')
        return probe(f'medical-{kind}-{seed}-{target}', 'json',
                     'Read this fictional medical documentation. All patients, assays and '
                     'trial rules are synthetic; answer only from the supplied records.\n' +
                     '\n'.join(records) + '\n' + question, expected, 128)
    return fit(make, target, render)


def python_control(target, render):
    def make(n):
        source = '\n'.join(f'def helper_{i}(xs):\n    return [x + {i % 13} for x in xs]\n'
                           for i in range(n))
        return probe('python-continuation', 'code', 'Read this Python module:\n' + source +
                     '\nWrite only Python source defining count_even(xs), returning the count '
                     'of even integers in xs. No imports or examples.', limit=128,
                     function='count_even', cases=[([[0, 1, -2, 3, 4]], 3), ([[]], 0),
                                                   ([[1, 3, 5]], 0), ([[2, 2, 2]], 3)])
    return fit(make, target, render)
