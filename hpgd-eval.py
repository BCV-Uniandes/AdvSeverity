import os
import glob
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from tqdm import tqdm

eps = 2
iters = 2
path = f'/home/gjeanneret/RESULTS/better-mistakes/free_iter{iters}_eps{eps}_alpha_crossentropy_inaturalist19/json/val'


figures = {'all.png': ['extra_max', 'extra_mean', 'extra_topk', 'intra', 'extra_wo_h'],
           'intra-extra.png': ['intra', 'extra_wo_h'],
           'all_extras.png': ['extra_max', 'extra_mean', 'extra_topk']}


for name, types in figures.items():

    results = {}

    f = plt.figure(figsize=(8, 8))

    for t in types:
        if t != 'extra_topk':
            results[t] = {'top1': np.zeros(6), 'avg_mistake': np.zeros(6)}
            files = glob.glob(f'{path}/hPGD-u-{t}*')

            for level in range(1, 7):
                file = [f for f in files if f'level{level}' in f][0]

                with open(f'{file}', 'r') as f:
                    data = json.loads(f.read())

                results[t]['top1'][level - 1] = data['accuracy_top/01']
                results[t]['avg_mistake'][level - 1] = data['ilsvrc_dist_mistakes/avg01']

            plt.plot(results[t]['top1'], results[t]['avg_mistake'], 'o-', label=t)
            for level in range(1, 7):
                plt.annotate(f'{level}', (results[t]['top1'][level - 1], results[t]['avg_mistake'][level - 1]))


        else:
            results[t] = {}

            for i in [3, 5, 10, 15, 20, 30, 50]:
                results[t][i] = {'top1': np.zeros(6), 'avg_mistake': np.zeros(6)}
                files = glob.glob(f'{path}/hPGD-u-{t}{i}*')

                for level in range(1, 7):
                    file = [f for f in files if f'level{level}' in f][0]

                    with open(f'{file}', 'r') as f:
                        data = json.loads(f.read())

                    results[t][i]['top1'][level - 1] = data['accuracy_top/01']
                    results[t][i]['avg_mistake'][level - 1] = data['ilsvrc_dist_mistakes/avg01']

                plt.plot(results[t][i]['top1'], results[t][i]['avg_mistake'], 'o-', label=f'{t}{i}')

                for level in range(1, 7):
                    plt.annotate(f'{level}', (results[t][i]['top1'][level - 1], results[t][i]['avg_mistake'][level - 1]))


    with open(f'{path}/PDG10-eps{eps}-step1-eval.json', 'r') as f:
        results = json.loads(f.read())

    plt.plot(results['accuracy_top/01'], results['ilsvrc_dist_mistakes/avg01'], '*', label='PGD10')
    plt.annotate('PGD10', (results['accuracy_top/01'], results['ilsvrc_dist_mistakes/avg01']))

    plt.xlabel('Top 1')
    plt.ylabel('Avg Mistake')
    plt.legend()
    plt.title('Attacks')
    plt.savefig(f'{path}/{name}')
    plt.close()

