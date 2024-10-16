import numpy as np
import pytest

from gp_mobo.optimiser import evaluate_objectives


class Test_evaluate_objectives:
    def test_it_correctly_filters_nans_from_first_function(self):
        smiles_list = ["A"] * 3
        objective_functions = [lambda x: [1, np.nan, 3], lambda x: [1, 2, 3]]

        output = evaluate_objectives(smiles_list, objective_functions)
        expected = np.array([[1, 1], [3, 3]])

        np.testing.assert_array_equal(output, expected)

    @pytest.mark.parametrize(
        "n, n_funcs",
        [(0, 2), (1, 2), (2, 3), (11, 117)],
    )
    def test_it_correctly_filters_nans_from_nth_function(self, n: int, n_funcs: int):
        smiles_list = ["A"] * 3

        def nth_func(x):
            return [1, np.nan, 3]

        def objective(x):
            return [1, 2, 3]

        objective_functions = [objective] * n + [nth_func] + [objective] * (n_funcs - n - 1)

        output = evaluate_objectives(smiles_list, objective_functions)

        expected = np.array([[1] * n_funcs, [3] * n_funcs], dtype=np.float64)

        np.testing.assert_array_equal(output, expected)
