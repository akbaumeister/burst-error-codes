import scipy.special

from lattice_algorithms import *
import matplotlib.pyplot as plt

n = 2
q = 5


# c = [2,0,2,0]
# wt, elapsed = burst_weight(q, c)
# print('An_smallest_rep, Computed in [s]', round(elapsed), ', vector:', c, ', burst weight:', wt)
# wt2, elapsed = burst_weight(q, c, 'brute_force')
# print('Brute force, Computed in [s]', round(elapsed), ', vector:', c, ', burst weight:', wt2)

#
# mask = [-1, 1]
# for i in range(1000):
#     c = random_vector(q, n)
#     print('Random vector:', c, 'Convolved:', np.convolve(c, mask))
#     wt, elapsed = burst_weight(q, c)
#     print(wt)
#     #print('An_smalles_rep, Computed in [s]', round(elapsed), ', vector:', c, ', burst weight:', wt)
#     wt2, elapsed = burst_weight(q, c, 'brute_force')
#     print(wt2)
#     #print('Brute force, Computed in [s]', round(elapsed), ', vector:', c, ', burst weight:', wt2)
#     if wt != wt2:
#         print(c, np.convolve(c, mask), wt, wt2)
#         raise ValueError

# test uniform sampling
t = 3
n = 5

keys = [str(e) for e in enumerate_sphere(n, t)]
vecs = dict.fromkeys(keys, 0)

for i in range(10000000):
    v = sample_uniformly_An(n, t)
    vecs[str(v)] += 1

plt.bar(vecs.keys(), vecs.values())

#for export
plt.axis('off')
plt.savefig('sampling_nonuniform_n5_t3.png', bbox_inches='tight', pad_inches=0, dpi=1200)

y_pos = range(len(keys))
plt.xticks(y_pos, vecs.keys(), rotation=90)

# test enumerate
# print(list(enumerate_sphere(3, 3)))
# print(len(list(enumerate_sphere(3, 3))))

print(*(k for k, v in vecs.items() if v==0))
plt.show()