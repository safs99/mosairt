import sys
sys.path.insert(0, '.')

import numpy as np
from irt.models.model_1pl import Rasch
from irt.utils.evaluation import (cross_validate, mae_b, rmse_b,
                                   r2_theta, correlation_theta)

# Données simulées
np.random.seed(42)
n_persons, n_items = 200, 8
b_true = np.linspace(-2, 2, n_items)
theta_true = np.random.normal(0, 1, n_persons)

P_true = 1 / (1 + np.exp(-(theta_true[:, None] - b_true[None, :])))
X = (np.random.rand(n_persons, n_items) < P_true).astype(int)


# Test du modèle
model = Rasch(verbose=True)
model.fit(X)


#test données simulés 
print(f"MAE sur b  : {mae_b(model.b_, b_true):.4f}")
print(f"RMSE sur b : {rmse_b(model.b_, b_true):.4f}")
print(f"R² sur θ   : {r2_theta(model.theta_, theta_true):.4f}")
print(f"Corr θ     : {correlation_theta(model.theta_, theta_true):.4f}")
print(f"\nb estimés : {model.b_.round(2)}")
print(f"b vrais   : {b_true.round(2)}")
print(f"Convergé  : {model.converged_}")


results = cross_validate(Rasch, X, k=5)