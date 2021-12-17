import numpy as np
import matplotlib.pyplot as plt
from qmc_sse import simulate
from tqdm import tqdm

def plot_sus(n, L, Tmin, Tmax, N):
    T = np.linspace(Tmin, Tmax, N)
    beta = 1/T
    sus = np.zeros(N)
    for i in tqdm(range(N)):
        sus[i] = simulate(n, L, beta[i], 10000, 100000)[0][-2]
    plt.plot(T, sus, label=f'L={L}')

for l in np.arange(4)+3:
    print(f"Simulating L={l}")
    plot_sus(2, l, 0.1, 2, 40)
plt.legend()
plt.xlabel(r'$T/J$')
plt.ylabel(r'$\chi$')
plt.savefig('chi_odd_even.pdf')
plt.show()