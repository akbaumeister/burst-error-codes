import numpy as np
import scipy.special
from utils import *
from lattice_algorithms import *
import matplotlib.pyplot as plt

qs = range(8)
ns = range(8)

#records = np.zeros((100, 100), dtype=int)
records = np.load('records_n_q.npy')

for q in qs:
    for n in ns:
        if records[n, q] != 0:
            continue
        print('processing q =', q, 'n =', n)
        highest = 0
        S = enumerate_burst_vectors(n, q)
        for i in range(q**n-1):
            vec = next(S)
            wt = burst_weight(q, vec)[0]
            if wt > highest:
                highest = wt
        records[n, q] = highest

np.save('records_n_q', records)

plt.imshow(records, cmap='hot', interpolation='nearest')
plt.show()

np.savetxt('records_textdump', records, fmt='%d', delimiter='&')