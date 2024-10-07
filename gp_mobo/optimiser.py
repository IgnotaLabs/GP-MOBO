from typing import Callable, List, Union

import numpy as np

ORACLE_LIKE = Callable[[Union[str, List[str]]], Union[float, np.ndarray]]


def evaluate_objectives(smiles_list: list[str], oracles: List[ORACLE_LIKE]) -> np.ndarray:
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
