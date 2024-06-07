import scipy


# sphere size
def sphere_size(n, m):
    # returns the number of length n vectors of distance exactly m to vec(0)n

    ub = min(m, n)
    return int(
        sum([scipy.special.binom(n + m - 1 - i, n - 1) * (scipy.special.binom(n, i)) ** 2 for i in range(ub + 1)]))
