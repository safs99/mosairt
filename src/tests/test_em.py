import sys
sys.path.insert(0, '.')

import numpy as np
from irt.estimation.em_algo import(
    gauss_hermite_quadrature,
    e_step,
    m_step_1pl,
    m_step_2pl,
    m_step_3pl,
    m_step_4pl,
    compute_eap,
    marginal_log_likelihood,
    run_em
)

np.random.seed(42)
N_PERSONS, N_ITEMS = 200, 8

B_TRUE = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
A_TRUE = np.array([1.2, 0.8, 1.5, 1.0, 1.3, 0.9, 1.1, 1.4])
C_TRUE = np.array([0.2, 0.15, 0.25, 0.2, 0.1, 0.2, 0.15, 0.2])
D_TRUE = np.array([0.9, 0.95, 0.9, 0.92, 0.88, 0.95, 0.9, 0.93])
THETA_TRUE = np.random.normal(0, 1, N_PERSONS)



