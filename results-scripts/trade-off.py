import matplotlib.pyplot as plt

pgd = [[3.25],
       [11.06]]

Ac = 29.64
nc = 100 - Ac
Mc = 3.25

lha = [[3.03,  3.04,  3.07,  3.17,  3.21,  3.23],
       [13.25, 13.03, 12.58, 11.20, 11.01, 10.97]]

gha = [[3.25,  3.36,  3.37,  3.38,  3.39,  3.38],
       [11.06, 13.27, 13.59, 14.11, 15.99, 16.78]]

nha = [[3.41,  3.43,  3.45,  3.46,  3.44,  3.42],
       [13.58, 13.97, 14.70, 18.55, 20.61, 21.03]]

data = [pgd, lha, gha, nha]
names = ['PGD', 'LHA', 'GHA', 'NHA']
color = ['k', '#ff7f0e', '#1f77b4', '#2ca02c']
marker = ['d', 's', 'p', '*']


f = plt.figure()
for idx in range(3):
    x = [Ac - Aa for Aa in data[idx+1][1]]
    y = [((100 - Aa) * Ma - nc * Mc) / (Ac - Aa) for Ma, Aa in zip(data[idx+1][0], data[idx+1][1])]
    plt.plot(x, y,
             marker[idx+1] + '-', color=color[idx+1],
             label=names[idx+1])

    for i in range(6):
        plt.annotate(f'{i+1}', (x[i], y[i]))
        # plt.annotate(f'{names[idx+1]}@{i+1}', (x[i], y[i]))

plt.legend(loc='lower left', fontsize='x-large')
plt.title('Trade Off')
plt.xlabel('Drop Accuracy')
plt.ylabel('Flipped Average Mistake')
plt.grid()
plt.show()
