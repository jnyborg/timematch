import os
from os.path import join
from glob import glob
import json
import numpy as np



exp_path = 'outputs'
experiments = os.listdir(exp_path)
experiments = sorted(experiments)


for experiment in experiments:
    results = glob(f'{exp_path}/{experiment}/overall_*.json')
    results = sorted(results)
    print(experiment)
    for result in results:
        metrics = json.load(open(result))
        name = result.split('/')[-1].replace('overall_', '').replace('.json', '')
        print('\t' + name)
        for metric, scores in metrics.items():
            if metric not in ['accuracy', 'macro_f1']:
                continue
            mean = np.mean(scores) * 100
            std = np.std(scores) * 100
            print(f'\t\t{scores}')
            print(f'\t\t{metric}: {mean:.1f}$\\pm${std:.1f}')



