# experiments/run_experiments.py

import sys
sys.path.insert(0, '.')

import numpy as np
from irt.models.model_1pl import Rasch

#Les données (du notebook d'Anais)
X = np.array([
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
], dtype=float)


# Entrainement

model = Rasch()
model.fit(X)

# Prédiction vs realité

# On aplatit la matrice en une liste de paires (étudiant, item)
# C'est ce que le prof appelle "pour chaque paire (e, a)"
predicted_performances = model.predict().flatten().tolist()
expected_performances  = X.flatten().astype(int).tolist()

print("Prédictions :", predicted_performances)
print("Réalité     :", expected_performances)
print()

#métriques

predicted = np.array(predicted_performances)
expected  = np.array(expected_performances)

VP = int(np.sum((predicted == 1) & (expected == 1)))
VN = int(np.sum((predicted == 0) & (expected == 0)))
FP = int(np.sum((predicted == 1) & (expected == 0)))
FN = int(np.sum((predicted == 0) & (expected == 1)))

total     = VP + VN + FP + FN
accuracy  = (VP + VN) / total
precision = VP / (VP + FP) if (VP + FP) > 0 else 0
recall    = VP / (VP + FN) if (VP + FN) > 0 else 0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0)

print(f"VP={VP}, VN={VN}, FP={FP}, FN={FN}")
print(f"Accuracy  : {VP+VN}/{total} = {accuracy:.4f}")
print(f"Precision : {VP}/{VP+FP} = {precision:.4f}")
print(f"Recall    : {VP}/{VP+FN} = {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")