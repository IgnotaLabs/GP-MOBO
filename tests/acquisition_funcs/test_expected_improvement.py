import numpy as np

from gp_mobo.acquisition_funcs.expected_improvement import expected_improvement


class Test_expected_improvement:
    def test_it_handles_zero_variences(self):
        means, vars = np.zeros((2,)), np.zeros((2,))
        y_best = 0

        output = expected_improvement(means, vars, y_best)
        assert not np.any(np.isnan(output))
