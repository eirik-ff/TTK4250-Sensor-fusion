from typing import Tuple

import numpy as np


def discrete_bayes(
    # the prior: p(x), shape=(n,)
    pr: np.ndarray,
    # the conditional/likelihood: p(z | x), shape=(n, m)
    cond_pr: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the new marginal and conditional: shapes=((m,), (m, n))
    """Swap which discrete variable is the marginal and conditional."""

    # joint: p(x, z) = p(z | x) p(x), shape=(n, m)
    joint = cond_pr * pr.reshape((-1, 1))  # -1 is inferred dimension

    # marginal: p(z), shape=(m,)
    marginal = cond_pr.T @ pr

    # Take care of rare cases of degenerate zero marginal,
    # conditional/posterior: p(x | z), shape=(m, n)
    conditional = joint / marginal

    # flip axes?? (n, m) -> (m, n)
    conditional = conditional.T

    # optional DEBUG
    assert np.all(
        np.isfinite(conditional)
    ), "NaN or inf in conditional in discrete bayes"
    assert np.all(
        np.less_equal(0, conditional)
    ), "Negative values for conditional in discrete bayes"
    assert np.all(
        np.less_equal(conditional, 1)
    ), "Value more than on in discrete bayes"
    assert np.all(
        np.isfinite(marginal)
    ), "NaN or inf in marginal in discrete bayes"

    return marginal, conditional
