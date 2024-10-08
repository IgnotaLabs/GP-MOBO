from typing import Callable, List, Union

import numpy as np

ORACLE_LIKE = Callable[[Union[str, List[str]]], Union[float, np.ndarray]]


def evaluate_objectives(smiles_list: list[str], oracles: List[ORACLE_LIKE]) -> np.ndarray:
    """
    Given a list of N smiles, return an NxK array A such that
    A_{ij} is the jth objective function on the ith SMILES.

    """
    # Initialise arrays for each objective
    predictions = [np.array(oracle(smiles_list)) for oracle in oracles]

    # Filter out NaN values from f1 and corresponding entries in other arrays
    f1 = predictions[0]
    valid_indices = ~np.isnan(f1)
    predictions = [prediction[valid_indices] for prediction in predictions]

    # Ensure all arrays have the same shape
    if not all(len(f1) == len(prediction) for prediction in predictions):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack(predictions)
    return out.T  # transpose, Nx(n)


# def expected_hypervolume_improvement(
#     known_smiles: list[str],
#     query_smiles: list[str],
#     known_Y: np.ndarray,
#     oracles: List[ORACLE_LIKE],
#     gp_means: np.ndarray,
#     gp_amplitudes: np.ndarray,
#     gp_noises: np.ndarray,
#     n_iterations: int = 20,
# ) -> OptimizationResult:
#     """

#     args:
#     known_smiles: list of smiles strings for which all objective values have been observed
#     query_smiles: list of smiles strings for which we want to evaluate the acquisition function
#     known_Y: NxK array of known objective values
#     oracles: list of K oracles, each of which takes a list of smiles and returns a list of objective values
#     gp_means: KxN array of GP means
#     gp_amplitudes: KxN array of GP amplitudes
#     gp_noises: KxN array of GP noises
#     n_iterations: number of iterations to run the Bayesian optimization loop
#     """

#     assert known_Y.shape == (len(known_smiles), len(oracles))
#     BO_known_Y = known_Y.copy()
#     BO_known_smiles = list(known_smiles)

#     for iteration in range(n_iterations):
#         print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

#         y_best = np.max(BO_known_Y, axis=0)  # best eval so far
