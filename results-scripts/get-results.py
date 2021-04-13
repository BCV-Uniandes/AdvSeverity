import os
import json

from os.path import join

results_path = '/home/gjeanneret/RESULTS/robust-hierarchy'

fr_results = os.listdir(join(results_path, 'free'))
fc_results = os.listdir(join(results_path, 'free_curriculum'))

common_results = [i for i in fr_results if i in fc_results]

tops_to_consider = ('01', '05', '10')
measures = ('accuracy_top/', 'ilsvrc_dist_mistakes/avg')

results_to_check = ('clean', 'PGD-u-iter10-', 'hPGD-u-intra-level', 'hPGD-u-extra_max-level', 'hPGD-u-extra_wo_h-level')
levels = ['1', '2', '3', '4', '5', '6']

all_results = {}


def get_measures(path):
    with open(path, 'r') as f:
        data = json.loads(f.read())

    results = {}

    for measure in measures:
        results[measure] = {}
        mult = 100 if measure == 'accuracy_top/' else 1
        for t in tops_to_consider:
            results[measure][t] = data[measure + t] * mult

    return results


def get_difference(curr, free):
    results = {}
    for measure in measures:
        results[measure] = {}
        for t in tops_to_consider:
            results[measure][t] = curr[measure][t] - free[measure][t]

    return results


def get_difference_relative(curr, free):
    results = {}
    for measure in measures:
        results[measure] = {}
        for t in tops_to_consider:
            results[measure][t] = 100 * (curr[measure][t] - free[measure][t]) / free[measure][t]

    return results


def check_file(path):
    files = os.listdir(path)
    files_of_interest = []

    for r in results_to_check:
        files_of_interest += [f for f in files if r in f]

    return len(files_of_interest) == 0


for exp in common_results:
    print(exp)
    path_free = join(results_path, 'free', exp, 'json/val')
    path_curr = join(results_path, 'free_curriculum', exp, 'json/val')

    if check_file(path_free) or check_file(path_curr):
        print(f'Skipping experiemtn {exp}')
        continue

    concat_results = {}

    for m in results_to_check:

        if 'level' in m:

            for l in levels:
                json_free = [i for i in os.listdir(path_free) if m + l in i][0]
                json_curr = [i for i in os.listdir(path_free) if m + l in i][0]

                if json_curr != json_free:
                    print(f'Breaking loop with measure {m + l}')
                    break

                json_free = join(path_free, json_free)
                json_curr = join(path_curr, json_curr)

                try:
                    data_free = get_measures(json_free)
                except:
                    print(f'error on {json_free}')
                    break

                try:
                    data_curr = get_measures(json_curr)
                except:
                    print(f'error on {json_curr}')
                    break

                concat_results[m + l] = {'free': data_free,
                                         'free_curriculum': data_curr,
                                         'diff': get_difference(data_curr, data_free),
                                         'diff rel': get_difference_relative(data_curr, data_free)}

        else:
            json_free = [i for i in os.listdir(path_free) if m in i][0]
            json_curr = [i for i in os.listdir(path_free) if m in i][0]

            if json_curr != json_free:
                print(f'Breaking loop with measure {m}')
                break

            json_free = join(path_free, json_free)
            json_curr = join(path_curr, json_curr)

            try:
                data_free = get_measures(json_free)
            except:
                print(f'error on {json_free}')
                break

            try:
                data_curr = get_measures(json_curr)
            except:
                print(f'error on {json_curr}')
                break

            concat_results[m] = {'free': data_free,
                                 'free_curriculum': data_curr,
                                 'diff': get_difference(data_curr, data_free),
                                 'diff rel': get_difference_relative(data_curr, data_free)}

    all_results[exp] = concat_results

csv = ''
# process all information to create csv
for exp, result in all_results.items():
    csv += '\n' + exp + '\n'

    for metric in result.keys():
        csv += '\n;' + metric + '\n;'
        for m in measures:
            csv += ';' + m
            for t in tops_to_consider:
                csv += ';' + t
        csv += '\n'

        csv += ';free'
        for m in measures:
            csv += ';'
            for t in tops_to_consider:
                csv += ';' + str(result[metric]['free'][m][t])

        csv += '\n'

        csv += ';curr'
        for m in measures:
            csv += ';'
            for t in tops_to_consider:
                csv += ';' + str(result[metric]['free_curriculum'][m][t])
        csv += '\n'

        csv += ';diff'
        for m in measures:
            csv += ';'
            for t in tops_to_consider:
                csv += ';' + str(result[metric]['diff'][m][t])
        csv += '\n'

        csv += ';diff rel (%)'
        for m in measures:
            csv += ';'
            for t in tops_to_consider:
                csv += ';' + str(result[metric]['diff rel'][m][t])
        csv += '\n'

with open('results-summary.csv', 'w') as f:
    f.write(csv)
