import os
import json
import itertools
import matplotlib
import numpy as np

from tqdm import tqdm
from os.path import join
from matplotlib import cm
from matplotlib import pyplot as plt
matplotlib.use('Agg')

output_path = 'RESULTS/TRUE'
os.makedirs(output_path, exist_ok=True)

results_path = '/home/gjeanneret/RESULTS/newRH'

fr_results = os.listdir(join(results_path, 'free'))
fc_results = os.listdir(join(results_path, 'curr'))

common_results = [i for i in fr_results if i in fc_results]

measures = ('accuracy_top/01', 'ilsvrc_dist_mistakes/avg01')

h_results = ['hPGD-u-intra-level', 'hPGD-u-extra_max-level', 'hPGD-u-extra_wo_h-level']
clean_json = 'clean-eval.json'
results_to_check = ['clean', 'PGD-u']
results_to_check.extend([x + f'{y}' for (x, y) in itertools.product(h_results,  range(1, 7))])

epses = [4, 6, 8]
steps = [1, 2, 4, 6]
iters = [2, 4, 6, 8]

# we create a cube to store the results to easy plot any dimension
nanmat = np.zeros((len(epses), len(steps), len(iters)))
nanmat[...] = 'nan'


def get_data(path, json_name):
    path = join(path, json_name)
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return {k: v for k,v in data.items() if k in measures}


def plot_info(curr_data, free_data, metric, ax):

    to_plot = (curr_data - free_data)
    vmax = np.abs(to_plot)
    vmax = vmax[~np.isnan(to_plot)].max()
    cmap = matplotlib.cm.get_cmap('coolwarm_r') if metric == 'acc' else cm.coolwarm
    return ax.imshow(to_plot, cmap=cmap, vmin=-vmax, vmax=vmax)


def insert_variables(metric, ax, im):
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(metric)
    plt.xlabel('Iterations')
    plt.xticks(np.arange(len(iters)), iters)
    plt.ylabel('Steps')
    plt.yticks(np.arange(len(steps)), steps)


results = {}

results['free'] = {r: {'acc': nanmat.copy(), 'mistakes': nanmat.copy()} for r in results_to_check}
results['curr'] = {r: {'acc': nanmat.copy(), 'mistakes': nanmat.copy()} for r in results_to_check}


for e_idx, eps in enumerate(epses):
    for a_idx, a in enumerate(steps):
        for m_idx, m in enumerate(iters):

            exp_name = f'eps{eps}_iter{m}_step{a}'

            free_path = join(results_path, 'free', exp_name, 'json/val')
            curr_path = join(results_path, 'curr', exp_name, 'json/val')

            attack = f'-iter50-eps{eps}-step1-eval.json'

            # import pdb; pdb.set_trace()

            if not os.path.exists(free_path) or not os.path.exists(curr_path):
                print('skipping experiment', exp_name)
                continue

            if not os.path.exists(join(free_path, clean_json)) or not os.path.exists(join(curr_path, clean_json)):
                print('skipping experiment', exp_name)
                continue

            # get the results from the attacks
            for r in results_to_check:

                attack_name = clean_json if r == 'clean' else r + attack

                try:
                    attack_curr = get_data(curr_path, attack_name)
                    results['curr'][r]['acc'][e_idx, a_idx, m_idx] = attack_curr['accuracy_top/01'] * 100
                    results['curr'][r]['mistakes'][e_idx, a_idx, m_idx] = attack_curr['ilsvrc_dist_mistakes/avg01']
                except:
                    print(f'Skipping {r} in curr {exp_name}. No results available')
                    continue

                try:
                    attack_free = get_data(free_path, attack_name)
                    results['free'][r]['acc'][e_idx, a_idx, m_idx] = attack_free['accuracy_top/01'] * 100
                    results['free'][r]['mistakes'][e_idx, a_idx, m_idx] = attack_free['ilsvrc_dist_mistakes/avg01']
                except:
                    print(f'Skipping {r} in free {exp_name}. No results available')
                    continue


# save all results
# with open(f'{output_path}/tables-results-diff.json', 'w') as f:
#     f.write(json.dumps(results))

# plot everything
for r in tqdm(results_to_check):
    for e_idx, eps in enumerate(epses):
        f = plt.figure()

        # plot clean results
        ax = plt.subplot(1, 2, 1)
        # import pdb; pdb.set_trace()
        im = plot_info(results['curr'][r]['acc'][e_idx, ...],
                       results['free'][r]['acc'][e_idx, ...],
                       'acc', ax)
        insert_variables('Accuracy', ax, im)

        ax = plt.subplot(1, 2, 2)
        im = plot_info(results['curr'][r]['mistakes'][e_idx, ...],
                       results['free'][r]['mistakes'][e_idx, ...],
                       'mistakes', ax)
        insert_variables('Avg Mistake', ax, im)

        plt.tight_layout()
        # plt.suptitle(f'Eps = {eps} | r = {r}')
        plt.savefig(f'{output_path}/diff_{r}_eps{eps}.png')
        plt.close()
# import pdb; pdb.set_trace()
