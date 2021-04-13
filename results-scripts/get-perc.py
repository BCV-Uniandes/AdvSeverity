import os
import json
import itertools
import matplotlib
import numpy as np

from os.path import join
from matplotlib import pyplot as plt
matplotlib.use('Agg')

output_path = 'RESULTS/perc'
os.makedirs(output_path, exist_ok=True)

results_path = '/home/gjeanneret/RESULTS/robust-hierarchy'

fr_results = os.listdir(join(results_path, 'free'))
fc_results = os.listdir(join(results_path, 'free_curriculum'))

common_results = [i for i in fr_results if i in fc_results]

measures = ('accuracy_top/01', 'ilsvrc_dist_mistakes/avg01')

h_results = ['hPGD-u-intra-level', 'hPGD-u-extra_max-level', 'hPGD-u-extra_wo_h-level']
clean_json = 'clean-eval.json'
results_to_check = ['PGD-u']
results_to_check.extend([x + f'{y}' for (x, y) in itertools.product(h_results,  range(1, 7))])

epses = [4, 6, 8]
steps = [1, 2, 4, 6]
iters = [2, 4, 6, 8]
colors = ['r', 'g', 'b', 'y']


def get_data(path, json_name):
    path = join(path, json_name)
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return {k: v for k,v in data.items() if k in measures}


def plot_info(curr_data, free_data, r, metric):

    plt.plot(epses, [0 for _ in epses], '.k-')

    for idx, m in enumerate(iters):
        curr = np.array(curr_data[m][r][metric])
        free = np.array(free_data[m][r][metric])
        to_plot = 100 * (curr - free) / free
        plt.plot(epses, to_plot, '.' + colors[idx] + '-', label=f'm={m}')


def insert_variables(metric):
    plt.legend()
    plt.xlabel('Eps')
    plt.ylabel(metric)


results = {}
results['clean'] = get_data('/home/gjeanneret/RESULTS/better-mistakes/crossentropy_inaturalist19_v2/json/val', clean_json)
results['clean'] = get_data('/home/gjeanneret/RESULTS/better-mistakes/curriculum_inaturalist19/json/val', clean_json)

for a in steps:

    data = {'curr': {}, 'free': {}}

    for m in iters:

        data['free'][m] = {r: {'acc': [float('nan') for _ in epses],
                               'mistakes': [float('nan') for _ in epses]} for r in results_to_check}
        data['curr'][m] = {r: {'acc': [float('nan') for _ in epses],
                               'mistakes': [float('nan')for _ in epses]} for r in results_to_check}

        data['free'][m]['clean'] = {'acc': [float('nan') for _ in epses],
                                    'mistakes': [float('nan') for _ in epses]}
        data['curr'][m]['clean'] = {'acc': [float('nan') for _ in epses],
                                    'mistakes': [float('nan') for _ in epses]}

        for idx, eps in enumerate(epses):

            exp_name = f'eps{eps}_iter{m}_step{a}'

            free_path = join(results_path, 'free', exp_name, 'json/val')
            curr_path = join(results_path, 'free_curriculum', exp_name, 'json/val')

            attack = f'-iter10-eps{eps}-step1-eval.json'

            if not os.path.exists(free_path) or not os.path.exists(curr_path):
                print('skipping experiment', exp_name)
                continue

            if not os.path.exists(join(free_path, clean_json)) or not os.path.exists(join(curr_path, clean_json)):
                print('skipping experiment', exp_name)
                continue

            # clean evaluation
            try:
                clean_free = get_data(free_path, clean_json)

                data['free'][m]['clean']['acc'][idx] = clean_free['accuracy_top/01']
                data['free'][m]['clean']['mistakes'][idx] = clean_free['ilsvrc_dist_mistakes/avg01']

                clean_curr = get_data(curr_path, clean_json)

                data['curr'][m]['clean']['acc'][idx] = clean_curr['accuracy_top/01']
                data['curr'][m]['clean']['mistakes'][idx] = clean_curr['ilsvrc_dist_mistakes/avg01']
            except:
                print(f'Skipping {exp_name}. No clean results available')
                continue

            # get the results from the attacks
            for r in results_to_check:

                try:
                    attack_free = get_data(free_path, r + attack)

                    data['free'][m][r]['acc'][idx] = attack_free['accuracy_top/01']
                    data['free'][m][r]['mistakes'][idx] = attack_free['ilsvrc_dist_mistakes/avg01']

                    attack_curr = get_data(curr_path, r + attack)

                    data['curr'][m][r]['acc'][idx] = attack_curr['accuracy_top/01']
                    data['curr'][m][r]['mistakes'][idx] = attack_curr['ilsvrc_dist_mistakes/avg01']
                except:
                    print(f'Skipping {r} in {exp_name}. No results available')
                    continue

    results[a] = data

# save all results
with open(f'{output_path}/tables-results-diff.json', 'w') as f:
    f.write(json.dumps(results))

# plot everything
for a in steps:

    # f = plt.figure()

    # # plot clean results
    # plt.subplot(1, 2, 1)
    # plot_info(results[a]['curr'], results[a]['free'], 'clean', 'acc')
    # insert_variables('Accuracy')
    # plt.subplot(1, 2, 2)
    # plot_info(results[a]['curr'], results[a]['free'], 'clean', 'mistakes')
    # insert_variables('Avg Mistake')

    # plt.suptitle('Clean')
    # plt.savefig(f'{output_path}/clean.png')
    # plt.close()

    for r in results_to_check:
        f = plt.figure()

        # plot clean results
        plt.subplot(1, 2, 1)
        plot_info(results[a]['curr'], results[a]['free'], r, 'acc')
        insert_variables('Accuracy')
        plt.subplot(1, 2, 2)
        plot_info(results[a]['curr'], results[a]['free'], r, 'mistakes')
        insert_variables('Avg Mistake')
        plt.tight_layout()

        plt.suptitle(r)
        plt.savefig(f'{output_path}/perc_{r}_step{a}.png')
        plt.close()
