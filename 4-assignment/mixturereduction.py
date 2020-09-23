from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights, shape=(N,)
    mean: np.ndarray,  # the mixture means, shape(N, n)
    cov: np.ndarray,  # the mixture covariances, shape (N, n, n)
    ) -> Tuple[np.ndarray, np.ndarray]:
         # the mean and covariance of of the mixture, shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""
    # mean
    mean_bar = np.average(mean, weights=w, axis=0)

    # covariance
    # # internal covariance
    cov_int = np.average(cov, weights=w, axis=0)

    # # spread of means
    mean_diff = mean - mean_bar  # shape (N,n)
    # this implements (6.21), although probably more inefficient than using
    # np.average. However, the way mean_diff is calculated, it's shape doesn't
    # play too nice with the average routine.
    cov_ext = mean_diff.T @ np.diag(w) @ mean_diff
    # shapes:    (n, N)   @   (N, N)   @  (N, n)  = (n, n)

    # # total covariance
    cov_bar = cov_int + cov_ext

    return mean_bar, cov_bar
