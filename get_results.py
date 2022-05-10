import os
from os.path import join
from glob import glob
import json
import numpy as np
from collections import defaultdict



exp_path = 'outputs'
experiments = os.listdir(exp_path)
experiments = sorted(experiments)


r = defaultdict(list)

for experiment in experiments:
    results = glob(f'{exp_path}/{experiment}/overall_*.json')
    results = sorted(results)

    method, source = experiment.split('_')[:2]
    for result in results:
        metrics = json.load(open(result))
        name = result.split('/')[-1].replace('overall_', '').replace('.json', '')
        print('\t' + name)

        target = name.split('_')[1]

        print(method, source, target)
        for metric, scores in metrics.items():
            # if metric not in ['accuracy', 'macro_f1']:
            if metric != 'macro_f1':
                continue
            mean = np.mean(scores) * 100
            std = np.std(scores) * 100
            # f1 = f'\t\t{metric}: {mean:.1f}$\\pm${std:.1f}'
            # print(f'\t\t{metric}: {mean:.1f}$\\pm${std:.1f}')

        r[method].append((source, target, mean, std))
            # print(f'\t\t{scores}')
            #
            #

print(r)

for method, res in r.items():
    print(method)

    a = []

    for s, t, mean, std in res:
        if s == '32VNH' and t == '30TXT':
            a.append(f'{mean:.1f}$\\pm${std:.1f}')
    for s, t, mean, std in res:
        if s == '32VNH' and t == '31TCJ':
            a.append(f'{mean:.1f}$\\pm${std:.1f}')
    for s, t, mean, std in res:
        if s == '32VNH' and t == '33UVP':
            a.append(f'{mean:.1f}$\\pm${std:.1f}')
    for s, t, mean, std in res:
        if s == '30TXT' and t == '32VNH':
            a.append(f'{mean:.1f}$\\pm${std:.1f}')
    for s, t, mean, std in res:
        if s == '30TXT' and t == '31TCJ':
            a.append(f'{mean:.1f}$\\pm${std:.1f}')
    ms, ss = np.mean([x[2] for x in res]), np.mean([x[3] for x in res])
    a.append(f'{ms:.1f}$\\pm${ss:.1f}')

    print(' & '.join(a))


