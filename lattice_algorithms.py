import time
import itertools
from sympy.utilities.iterables import multiset_permutations
from scipy.signal import deconvolve
from utils import *


def enumerate_sphere(n, d):
    # enumerate all points over the A_n lattice which have distance d to the central point [0]*(n+1)

    parts_pos = rule_asc_len(d, n)
    parts_neg = rule_asc_len(d, n)

    for l1 in parts_pos:
        for l2 in parts_neg:
            if len(l1) + len(l2) > (n + 1):
                continue
            cat = l1 + [-c for c in l2] + [0] * ((n + 1) - len(l1) - len(l2))

            vecs = multiset_permutations(cat)
            for vec in vecs:
                yield vec
        parts_neg = rule_asc_len(d, n)


def burst_to_an(v):
    return np.convolve(v, [-1, 1])


def an_to_burst(v):
    s, _ = deconvolve(v, [-1, 1])
    s = np.array(s, dtype=int)
    return s


def lee_weight(q_, v_):
    return sum([min(i, q_ - i) for i in v_ % q_])


def random_vector(q_, n_):
    v_ = np.random.randint(q_, size=n_)
    return v_


def sample_non_uniformly_An(n, t):
    # sample naively a vector of burst weight t and length n+1.
    lambda_p = random_partition_uniform(t, n)
    lambda_n = [-i for i in random_partition_uniform(t, (n + 1) - len(lambda_p))]
    zeros = [0] * ((n + 1) - len(lambda_p) - len(lambda_n))
    vec = lambda_p + lambda_n + zeros
    random.shuffle(vec)

    return vec


def sample_uniformly_An(n, t):
    # sample uniformly a vector of burst weight t and length n+1.
    l_squared = gen_lambda_squared()
    keys = [tuple(lp,ln) for lp,ln in l_squared]
    num_perms = dict.fromkeys(keys)
    for lp_ln in num_perms.keys():
        num_perms[lp_ln] = num_multiset_permutations(lp_ln[0]) * num_multiset_permutations(lp_ln[1])

    return vec


def closest_qlattice_point(q_, v):
    # given a point of the lattice An, find the closest point of the lattice qAn
    assert sum(v) == 0, 'vector {} is not in the lattice A_n'.format(v)

    # step 1: calculate f(v), where f(v_i) is the closest q-multiple:
    f = np.zeros(len(v), dtype=int)
    for i, v_i in enumerate(v):
        f[i] = q_ * round(v_i / q_)

    # calculate the deficiency delta = sum(f)
    delta = sum(f)
    if delta == 0:
        return f
    else:
        # gamma(v) = v - f(v), the distance of each component to a q-Lattice point
        gamma = v - f
        delta_norm = abs(delta // q_)  # delta can only be multiples of q

        c = f.copy()
        if delta > 0:
            # find indices of elements with largest deviation in the negative direction
            idx = gamma.argsort()[:delta_norm]
            c[idx] -= q_
        else:
            idx = (-gamma).argsort()[:delta_norm]
            c[idx] += q_

    assert (sum(c) == 0)
    return c


def burst_weight(q, v, algorithm='An_smallest_rep'):
    assert all(0 <= i < q for i in v), 'burst_weight(q, v): the vector can only consist of elements of the field Fq'
    start = time.time()
    n = len(v)

    if algorithm == 'brute_force':
        # create a list of all unit bursts
        basis_vectors = []
        for num_ones in range(1, n + 1):
            for offset in range(n - num_ones + 1):
                vector = np.zeros(n, dtype=int)
                vector[offset:offset + num_ones] = 1
                basis_vectors.append(np.array(vector))

        dim_basis = len(basis_vectors)
        assert dim_basis == ((n + 1) * n / 2)

        lowest_weight = math.inf

        coefficients = itertools.product([i for i in range(q)], repeat=dim_basis)
        for candidate in coefficients:
            candidate_weight = lee_weight(q, np.array(candidate))
            if candidate_weight >= lowest_weight:
                continue
            v_c = np.zeros(n)
            for i, c in enumerate(candidate):
                v_c += c * basis_vectors[i]
                v_c %= q
            if np.array_equal(v_c, v):
                lowest_weight = min(lowest_weight, candidate_weight)

    elif algorithm == 'An_smallest_rep':
        mask = [-1, 1]
        vec = np.convolve(v, mask)
        center = closest_qlattice_point(q, vec)
        lowest_weight = int(np.linalg.norm(center - vec, 1) // 2)

    end = time.time()
    return lowest_weight, end - start
