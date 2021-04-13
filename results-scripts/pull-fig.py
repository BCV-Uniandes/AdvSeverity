import matplotlib.pyplot as plt

# adv_ac_xe = [11.07, 12.58, 13.59, 14.70]
# adv_ac_cu = [12.15, 14.49, 15.68, 17.06]

# adv_am_xe = [3.25, 3.07, 3.37, 3.45]
# adv_am_cu = [3.01, 2.78, 3.17, 3.25]

adv_ac_xe = [12.33, 13.80, 15.07, 16.39]
adv_ac_cu = [13.36, 15.36, 16.47, 17.58]

adv_am_xe = [3.20, 3.02, 3.33, 3.39]
adv_am_cu = [3.06, 2.86, 3.19, 3.27]

fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.subplot(1,2,1)
x = [0.5, 1.0, 1.5, 2.0]

plt.bar([i - 0.1 for i in x], adv_ac_xe, color='yellow', width=0.2, edgecolor='k', linewidth=1.5, label='Standard')
plt.bar([i + 0.1 for i in x], adv_ac_cu, color='royalblue', width=0.2, edgecolor='k', linewidth=1.5, label='Curriculum')
plt.ylim(10, 20)
plt.ylabel('Adversarial Accuracy')
plt.title('(a) Robustness')
plt.xticks(x, ['PGD50', 'LHA50@3', 'GHA50@3', 'NHA50@3'], rotation=45)
plt.legend()

plt.subplot(1,2,2)
plt.bar([i - 0.1 for i in x], adv_am_xe, color='yellow', width=0.2, edgecolor='k', linewidth=1.5, label='Standard')
plt.bar([i + 0.1 for i in x], adv_am_cu, color='royalblue', width=0.2, edgecolor='k', linewidth=1.5, label='Curriculum')
plt.ylim(2.6, 3.6)
plt.ylabel('Average Mistake')
plt.title('(b) Severity')
plt.xticks(x, ['PGD50', 'LHA50@3', 'GHA50@3', 'NHA50@3'], rotation=45)
plt.legend()

plt.show()
# plt.figure()
# name = ['PGD50', 'LHA50@3', 'GHA50@3', 'NHA50@3']
# colors = ['black', 'blue', 'green', 'orange']
# a = 0.05

# plt.plot([], [], 'x', color='k', label='Cross-Entropy')
# plt.plot([], [], 's', color='k', label='Curriculum')

# for i in range(4):
#     plt.plot([], [], color=colors[i], label=name[i])
#     plt.plot(adv_ac_xe[i], adv_am_xe[i], 'x', color=colors[i])
#     plt.plot(adv_ac_cu[i], adv_am_cu[i], 's', color=colors[i])
#     plt.arrow((1 - a) * adv_ac_xe[i] + a * adv_ac_cu[i],
#               (1 - a) * adv_am_xe[i] + a * adv_am_cu[i],
#               (1 - 2 * a) * (adv_ac_cu[i] - adv_ac_xe[i]),
#               (1 - 2 * a) * (adv_am_cu[i] - adv_am_xe[i]),
#               color=colors[i], length_includes_head=True,
#               head_width=0.01)

# plt.legend()
# plt.grid()
# plt.show()