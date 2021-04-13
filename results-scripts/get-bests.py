import os
import json
import itertools
import numpy as np

from tqdm import tqdm
from os.path import join


output_path = 'RESULTS/diff-3D'
os.makedirs(output_path, exist_ok=True)

results_path = '/home/gjeanneret/RESULTS/newRH'

# fr_results = os.listdir(join(results_path, 'free'))
# fc_results = os.listdir(join(results_path, 'free_curriculum'))
fr_results = os.listdir(join(results_path, 'free'))
fc_results = os.listdir(join(results_path, 'curr'))

common_results = [i for i in fr_results if i in fc_results]

measures = ('accuracy_top/01', 'ilsvrc_dist_mistakes/avg01')

h_results = ['hPGD-u-intra-level', 'hPGD-u-extra_max-level', 'hPGD-u-extra_wo_h-level']
clean_json = 'clean-eval.json'
results_to_check = ['clean', 'PGD-u']
# results_to_check.extend([x + f'{y}' for (x, y) in itertools.product(h_results,  range(1, 7))])

epses = [4, 6, 8]
steps = [1, 2, 4, 6]
iters = [2, 4, 6, 8]

# we create a cube to store the results to easy plot any dimension
nanmat = np.zeros((len(epses), len(steps), len(iters)))
# nanmat[...] = 'nan'


def get_data(path, json_name):
    path = join(path, json_name)
    with open(path, 'r') as f:
        data = json.loads(f.read())
    return {k: v for k,v in data.items() if k in measures}


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

            # get the results from the attacks
            for r in results_to_check:

                attack_name = clean_json if r == 'clean' else r + attack

                try:
                    attack_free = get_data(free_path, attack_name)
                    results['free'][r]['acc'][e_idx, a_idx, m_idx] = attack_free['accuracy_top/01'] * 100
                    results['free'][r]['mistakes'][e_idx, a_idx, m_idx] = attack_free['ilsvrc_dist_mistakes/avg01']
                except:
                    print(f'Skipping {r} in free {exp_name}')

                try:
                    attack_curr = get_data(curr_path, attack_name)
                    results['curr'][r]['acc'][e_idx, a_idx, m_idx] = attack_curr['accuracy_top/01'] * 100
                    results['curr'][r]['mistakes'][e_idx, a_idx, m_idx] = attack_curr['ilsvrc_dist_mistakes/avg01']
                except:
                    print(f'Skipping {r} in curr {exp_name}')


# plot everything
r = 'PGD-u'
for e_idx, eps in enumerate(epses):

    # takes best for PGD
    max_free = np.unravel_index(results['free'][r]['acc'][e_idx, ...].argmax(),
                                results['free'][r]['acc'][e_idx, ...].shape)
    best_free = results['free'][r]['acc'][e_idx, max_free[0], max_free[1]]
    am_free = results['free'][r]['mistakes'][e_idx, max_free[0], max_free[1]]
    print(f'For {eps}, best model for free is: {best_free}/{am_free} with iter = {iters[max_free[1]]}, step = {steps[max_free[0]]}')

    max_curr = np.unravel_index(results['curr'][r]['acc'][e_idx, ...].argmax(),
                                results['curr'][r]['acc'][e_idx, ...].shape)
    best_curr = results['curr'][r]['acc'][e_idx, max_curr[0], max_curr[1]]
    am_curr = results['curr'][r]['mistakes'][e_idx, max_curr[0], max_curr[1]]
    print(f'For {eps}, best model for curr is: {best_curr}/{am_curr} with iter = {iters[max_curr[1]]}, step = {steps[max_curr[0]]}')

    # we extract the results of all best models



# import pdb; pdb.set_trace()
