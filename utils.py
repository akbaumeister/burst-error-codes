import random
import numpy as np
import math


def rule_asc_len(n, l):
    # produce all partitions of the integer n into at most l parts

    a = [0] * (n + 1)
    a[1] = n

    k = 1
    while k:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while x <= y and k < l - 1:
            a[k] = x
            y -= x
            k += 1
        a[k] = x + y

        yield a[:k + 1]


def count_partitions(n, l):
    # count the number of partitions of the integer n into at most l parts
    gen = rule_asc_len(n, l)
    return sum(1 for _ in gen)


def random_partition_uniform(n, l):
    # produce a random partition of the integer n into at most l parts
    n_partitions = count_partitions(n, l)
    parts = rule_asc_len(n, l)
    which = random.randint(1, n_partitions)
    for i in range(which - 1):
        next(parts)
    return next(parts)


def num_multiset_permutations(vec):
    _, counts = np.unique(vec, return_counts=True)
    denom = np.prod([math.factorial(i) for i in counts])
    return math.factorial(len(vec))//denom


def gen_lambda_squared(n, t):
    gen_pos = rule_asc_len(t, n)
    gen_neg = rule_asc_len(t, n)

    for l_p in gen_pos:
        for l_n in gen_neg:
            if len(l_p) + len(l_n) <= n+1:
                yield l_p, [-i for i in l_n]
