import numpy as np
import matplotlib.pyplot as plt
from qmc_sse import simulate
from tqdm import tqdm

N = 20
T = np.linspace(0.1, 2, N)
beta = 1/T
res = []
err = []
for i in tqdm(range(N)):
    r, e = simulate(2,4,beta[i],10000,200000,need_cor=True)
    res.append(r)
    err.append(e)
res = np.array(res)
err = np.array(err)

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.errorbar(T, res[:, 0], yerr=err[:, 0], fmt='-o', lw=0.8, mew=0.01)
ax1.set_xlabel(r'$T/J$')
ax1.set_ylabel('E')

ax2.errorbar(T, res[:, 1], yerr=err[:, 1], fmt='-o', lw=0.8, mew=0.01)
ax2.set_xlabel(r'$T/J$')
ax2.set_ylabel('C')

ax3.errorbar(T, res[:, 2], yerr=err[:, 2], fmt='-o', lw=0.8, mew=0.01)
ax3.set_xlabel(r'$T/J$')
ax3.set_ylabel(r'$\langle m_s^2 \rangle$')

ax4.errorbar(T, res[:, -3], yerr=err[:, -3], fmt='-o', lw=0.8, mew=0.01)
ax4.set_xlabel(r'$T/J$')
ax4.set_ylabel('C(1)')

fig.savefig('2D_basic_property.pdf')
plt.close()

plt.plot(T, res[:, -2], '-o', label=r'$\chi$', lw=0.8, mew=0.01)
plt.plot(T, res[:, -1], '-o', label='free spin', lw=0.8, mew=0.01)
plt.xlabel(r'$T/J$')
plt.ylabel(r'$\chi$ and free spin')
plt.legend()
plt.savefig('chi_freespin.pdf')
plt.close()


plt.scatter(T, err[:, -4], label='standard estimator')
plt.scatter(T, err[:, -2], label='improved estimator')
plt.legend()
plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
plt.xlabel(r'$T/J$')
plt.ylabel(r'error of $\chi$')
plt.savefig('estimator_error.pdf')
plt.close()

