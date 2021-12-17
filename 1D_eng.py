import numpy as np
import matplotlib.pyplot as plt
from qmc_sse import simulate
from tqdm import tqdm

m3 = np.arange(6, 11, 1)
res3 = []
err3 = []
for i in tqdm(m3):
    r, e = simulate(1, 512, 2**i, 2000, 10000)
    res3.append(r)
    err3.append(e)
res3 = np.array(res3)
err3 = np.array(err3)

plt.errorbar(m3, -res3[:, 0], yerr=err3[:, 0], label=r'$N=512$')
plt.xlabel(r'$m$ ($\beta=2^m$)')
plt.ylabel(r'$-E/N$')
plt.savefig('1D_eng.pdf')
plt.show()