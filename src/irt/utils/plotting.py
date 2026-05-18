"""Fichier pour stocker les fonctions de dessin"""

import numpy as np
import matplotlib.pyplot as plt

def plot_icc(model):
    """Fonction de dessin des ICC"""

    if not hasattr(model, 'b_'):
        print("Erreur : le modèle doit être entrainé avant de pouvoir visualiser les ICC.")
        return

    theta_range = np.linspace(-4, 4, 100)
    plt.figure(figsize=(10, 6))

    for i, b_j in enumerate (model.b_):
        p = 1.0 / (1.0 + np.exp(-(theta_range - b_j)))
        plt.plot(theta_range, p, label=f'Item {i+1} (b={b_j:.2f})')

    plt.axhline(0.5, color='red', linestyle='--', alpha = 0.3)
    plt.title("Courbes ICC")
    plt.xlabel("Capacité de l'individu (thêta)")
    plt.ylabel("Probabilité de réussite P(X=1)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha = 0.2)
    plt.tight_layout()

    print("Fenêtre graphique en cours d'ouverture")
    plt.show()
