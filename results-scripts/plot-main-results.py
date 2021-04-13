import matplotlib.pyplot as plt

# lha acc
lha_cc = [[16.32, 16.01, 15.36, 13.60, 13.36, 13.33], [10.71, 10.49, 9.92, 8.53, 8.36, 8.32], [7.65, 7.52, 7.05, 6.11, 5.92, 5.98]]
lha_cf = [[14.64, 14.39, 13.80, 12.39, 12.23, 12.25], [ 8.99,  8.78, 8.33, 7.29, 7.09, 7.08], [5.54, 5.45, 5.13, 4.31, 4.24, 4.27]]

# lha am
lha_mc = [[2.81, 2.82, 2.86, 2.97, 3.01, 3.02], [3.06, 3.07, 3.10, 3.20, 3.24, 3.26], [3.27, 3.28, 3.32, 3.41, 3.45, 3.46]]
lha_mf = [[2.98, 2.99, 3.02, 3.12, 3.15, 3.17], [3.23, 3.23, 3.27, 3.36, 3.39, 3.41], [3.53, 3.54, 3.57, 3.64, 3.67, 3.69]]


# gha acc
gha_cc = [[13.36, 15.97, 16.47, 17.01, 19.33, 20.39], [8.34, 10.20, 10.55, 10.89, 12.75, 13.79], [6.05, 7.28, 7.50, 7.81, 9.26, 10.10]]
gha_cf = [[12.33, 14.71, 15.07, 15.56, 17.52, 18.42], [7.11,  8.55,  8.82,  9.14, 10.69, 11.60], [4.31, 5.21, 5.35, 5.53, 6.60,  7.32]]

#gha am
gha_mc = [[3.05, 3.18, 3.19, 3.20, 3.21, 3.19], [3.29, 3.40, 3.41, 3.43, 3.44, 3.44], [3.49, 3.58, 3.58, 3.60, 3.62, 3.62]]
gha_mf = [[3.20, 3.32, 3.33, 3.34, 3.34, 3.33], [3.45, 3.54, 3.55, 3.56, 3.58, 3.58], [3.71, 3.79, 3.79, 3.80, 3.82, 3.82]]


# nha acc
nha_cc = [[16.28, 16.87, 17.58, 22.03, 23.92, 24.46], [10.29, 10.54, 11.09, 15.20, 16.96, 17.49], [7.21, 7.49, 7.84, 11.00, 12.92, 13.46]]
nha_cf = [[15.12, 15.59, 16.39, 20.56, 22.54, 23.12], [ 8.63,  8.88,  9.29, 12.85, 14.71, 15.24], [5.27, 5.39, 5.65,  8.06,  9.93, 10.27]]

# nha am
nha_mc = [[3.23,3.25,3.27,3.27,3.24,3.22], [3.45,3.47,3.49,3.52,3.51,3.49], [3.63,3.64,3.66,3.71,3.70,3.69]]
nha_mf = [[3.38,3.39,3.41,3.39,3.39,3.36], [3.59,3.60,3.62,3.66,3.65,3.63], [3.84,3.85,3.87,3.92,3.91,3.90]]

eps = [4, 6, 8]
color = ['orange', 'blue', 'green']

def plot_results(curr_ac, curr_am, free_ac, free_am):
    plt.plot([], [], 'k-s', label='Curriculum')
    plt.plot([], [], 'k:x', label='Cross-Entropy')
    for i in range(3):
        plt.plot([], [], color=color[i], label=r'$\epsilon$ = ' + str(eps[i]))
        plt.plot(curr_ac[i], curr_am[i], '-s', color=color[i])
        plt.plot(free_ac[i], free_am[i], ':x', color=color[i])
    plt.xlabel('Accuracy')
    plt.ylabel('Average Mistake')
    plt.grid()
    plt.legend()
    plt.show()


plot_results(lha_cc, lha_mc, lha_cf, lha_mf)
plot_results(gha_cc, gha_mc, gha_cf, gha_mf)
plot_results(nha_cc, nha_mc, nha_cf, nha_mf)