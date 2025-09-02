import numpy as np
from numpy import convolve

def get_weights(d, size):
    # gets weights from w_0 = 1 up to but not including size

    factors = np.ones(size)
    k_array = np.arange(1, size)
    factors[1:] = - 1 *(d - k_array + 1 ) /k_array
    weights = factors.cumprod()

    return weights


def get_fractionally_differenced_series(d, X):
    # Fractional differencing using an expanding window with no cutoff
    # Some confusion about whether to drop the first entry.
    # To align exactly with .diff(1), we should.
    # However, this will prevent a difference of d and -d from being
    # true inverses.

    W = get_weights(d, X.shape[0])
    return convolve(W, X, mode="full")[:X.shape[0]]


def get_relative_weight_loss_cutoff(weights, tol):
    # Typos, tacit reversals, and bad indexing choices make this very annoying

    # Define lambda_l = sum_{T-l}^{T-1} |w_j| / sum_{all} |w_i|, is the missing weight from \tilde{X}_{T-l}.
    # lambda_l decreases as l increases.
    # We can use l up until up to and not including where lambda_l > tol --> \tilde{X}_{T-l} beyond this l

    lambda_l = np.abs(weights)[::-1].cumsum() / np.sum(np.abs(weights))

    # using i_star instead of l_start to break connection with book, which has lambda 1 indexed
    i_star = np.searchsorted(lambda_l, tol)

    # if Z = get_fractionally_differenced_series(W, X), can use Z[-i_star:]
    return int(i_star)


def get_fixed_window_weights(weights, tol):
    # Any weights falling below threshold, set them to zero.
    # note: for negative d, takes a while for weights to fall below threshold; slow decay
    # consider a higher threshold

    w = weights.copy()
    w[np.abs(w) <= tol] = 0

    return w[np.abs(w) > 0]


def get_frac_diff_series_FW(d, X, tol):
    W = get_weights(d, X.shape[0])

    fixed_window_weights = get_fixed_window_weights(W, tol)
    c = fixed_window_weights.shape[0]

    return convolve(fixed_window_weights, X, mode="full")[c - 1:X.shape[0]]


def get_frac_diff_series_EW(d, X, tol):
    W = get_weights(d, X.shape[0])

    frac_diff_series = get_fractionally_differenced_series(d, X)
    i_star = get_relative_weight_loss_cutoff(W, tol)

    return frac_diff_series[-i_star:]


