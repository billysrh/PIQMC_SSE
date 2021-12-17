import numpy as np
import matplotlib.pyplot as plt
from qmc_sse import simulate
from tqdm import tqdm

N5 = 8
beta5 = 2**10
T5 = 1/beta5
L5 = np.arange(N5, dtype=np.int32)*2 + 8
res5 = []
err5 = []
for i in tqdm(range(N5)):
    r, e = simulate(2, L5[i], beta5, 1000, 5000)
    res5.append(r)
    err5.append(e)
res5 = np.array(res5)
err5 = np.array(err5)

a, b = np.polyfit(1/L5, res5[:, 3], 1)
print(a, b)
print(err5[:, 3])

plt.errorbar(1/L5, res5[:, 3], yerr=err5[:, 3], fmt='o', lw=0.8, mew=0.01)
plt.plot(np.linspace(0, 0.15), a*np.linspace(0, 0.15)+b, label=rf'$b={b}$')
plt.xlabel(r'$1/L$')
plt.ylabel(r'$\langle m_s^2 \rangle$')
plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
plt.legend()
plt.savefig('2D_ms_L.pdf')
plt.show()