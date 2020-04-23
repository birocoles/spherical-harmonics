import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import associated_legendre as al


def test_comparison_fully_norm_X_analytical():
    'Compare the analytical expressions and those obtained by recurrence'
    colatitudes = np.linspace(5, 175, 60)
    maximum_degree = 4
    for colatitude in colatitudes:
        Pnm_recurrence = al.fully_norm_Plm(colatitude, maximum_degree)
        Pnm_analytical = al.fully_norm_analytical(colatitude)
        aae(Pnm_recurrence, Pnm_analytical, decimal=10)
