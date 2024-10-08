import numpy as np
from scipy import stats

from gp_mobo.acquisition_funcs.hypervolume import Hypervolume


def expected_improvement(pred_means: np.ndarray, pred_vars: np.ndarray, y_best: float) -> np.ndarray:
    """
    Calculate the expected improvement of a set of predictions.
    Args:
        pred_means: The predicted means of the model.
        pred_vars: The predicted variances of the model.
        y_best: The best value observed so far.

    Returns:
        np.ndarray: The expected improvement of the predictions.
    """
    std = np.sqrt(pred_vars)
    z = (pred_means - y_best) / std
    ei = (pred_means - y_best) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
    return np.maximum(ei, 1e-30)


def expected_hypervolume_improvement(
    pred_means: np.ndarray, pred_vars: np.ndarray, reference_point: np.ndarray, pareto_front: np.ndarray, N=1000
):
    num_points, _ = pred_means.shape
    ehvi_values = np.zeros(num_points)

    hv = Hypervolume(reference_point)
    current_hv = hv.compute(pareto_front)

    for i in range(num_points):
        mean = pred_means[i]
        var = pred_vars[i]
        cov = np.diag(var)

        # Monte Carlo integration
        samples = np.random.multivariate_normal(mean, cov, size=N)

        hvi = 0.0
        for sample in samples:
            augmented_pareto_front = np.vstack([pareto_front, sample])
            hv_sample = hv.compute(augmented_pareto_front)
            hvi += max(0, hv_sample - current_hv)

        ehvi_values[i] = hvi / N

    return ehvi_values
