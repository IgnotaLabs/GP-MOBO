import logging
from typing import Callable, List, Literal, Optional, Union

import numpy as np

from gp_mobo.acquisition_funcs.expected_improvement import (
    expected_hypervolume_improvement,
)
from gp_mobo.acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from gp_mobo.acquisition_funcs.pareto import pareto_front
from gp_mobo.kern_gp.gp_model import independent_tanimoto_gp_predict

ORACLE_LIKE = Callable[[Union[str, List[str]]], Union[float, np.ndarray]]
ACQUISITION_FUNCTION = Literal["ehvi", "ei"]


def ehvi_acquisition(
    query_smiles: List[str],
    known_smiles: List[str],
    known_Y: np.ndarray,
    gp_means: np.ndarray,
    gp_amplitudes: np.ndarray,
    gp_noises: np.ndarray,
    reference_point: np.ndarray,
):
    pred_means, pred_vars = independent_tanimoto_gp_predict(
        query_smiles=query_smiles,
        known_smiles=known_smiles,
        known_Y=known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
    )

    pareto_mask = pareto_front(known_Y)
    pareto_Y = known_Y[pareto_mask]

    ehvi_values = expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_Y)

    return ehvi_values


def evaluate_objectives(smiles_list: list[str], oracles: List[ORACLE_LIKE]) -> np.ndarray:
    """
    Evaluate the objectives for a list of smiles.

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


def mobo(
    objective_functions: List[ORACLE_LIKE],
    known_smiles: list[str],
    known_Y: np.ndarray,
    query_smiles: list[str],
    gp_means: np.ndarray,
    gp_amplitudes: np.ndarray,
    gp_noises: np.ndarray,
    max_ref_point: np.ndarray = None,
    scale: float = 0.1,
    scale_max_ref_point: bool = False,
    n_iterations: int = 20,
) -> tuple[list[str], np.ndarray, list[float], list[float]]:
    """
    Multi-objective Bayesian optimisation using Expected Hypervolume Improvement (EHVI).

    Args:
    - objective_functions: List of K objective functions to be optimised.
    - known_smiles: list of smiles strings for which all objective values have been observed
    - known_Y: NxK array of known objective values
    - query_smiles: list of smiles strings for which we want to evaluate the acquisition function
    - gp_means: KxN array of GP means
    - gp_amplitudes: KxN array of GP amplitudes
    - gp_noises: KxN array of GP noises
    - n_iterations: number of iterations to run the Bayesian optimization loop


    Returns:
    - known_smiles: list of smiles strings for which all objective values have been observed
    - known_Y: All observed objective values
    - hypervolumes_bo: List of hypervolumes at each iteration
    - acquisition_values: List of maximum acquisition function values at each iteration
    """
    S_chosen = set()
    hypervolumes_bo = []
    acquisition_values = []

    # Initialise reference point
    for iteration in range(n_iterations):
        logging.info(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")
        reference_point = infer_reference_point(
            known_Y, max_ref_point=max_ref_point, scale=scale, scale_max_ref_point=scale_max_ref_point
        )

        max_acq = -np.inf
        best_smiles = None

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            ehvi_values = ehvi_acquisition(
                query_smiles=[smiles],
                known_smiles=known_smiles,
                known_Y=known_Y,
                gp_means=gp_means,
                gp_amplitudes=gp_amplitudes,
                gp_noises=gp_noises,
                reference_point=reference_point,
            )
            ehvi_value = ehvi_values[0]
            if ehvi_value > max_acq:
                max_acq = ehvi_value
                best_smiles = smiles

        acquisition_values.append(max_acq)
        logging.debug(f"Max acquisition value: {max_acq}")
        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_objectives([best_smiles], objective_functions)
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])
            logging.info(f"Chosen SMILES: {best_smiles} with acquisition function value = {max_acq}")
            logging.info(f"Value of chosen SMILES: {new_Y}")
            logging.info(f"Updated dataset size: {known_Y.shape}")
        else:
            logging.info("No new SMILES selected.")

        hv = Hypervolume(reference_point)
        current_hypervolume = hv.compute(known_Y)
        hypervolumes_bo.append(current_hypervolume)
        logging.debug(f"Hypervolume: {current_hypervolume}")

    return known_smiles, known_Y, hypervolumes_bo, acquisition_values
