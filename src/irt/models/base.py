"""
base.py
-------
Classe abstraite dont héritent tous les modèles IRT.
Elle définit l'interface commune : fit(), probability(),
log_likelihood() et information().
"""

import numpy as np
from abc import ABC, abstractmethod


class IRTModel(ABC):
    """
    Classe de base pour tous les modèles IRT.

    Toutes les sous-classes DOIVENT implémenter :
        - probability(theta)  → courbe ICC
        - fit(X, mask)        → estimation des paramètres

    Toutes les sous-classes HÉRITENT automatiquement de :
        - log_likelihood(X, mask)
        - information(theta)
    """

    # ------------------------------------------------------------------
    # Méthodes abstraites — chaque sous-classe doit les implémenter
    # ------------------------------------------------------------------

    @abstractmethod
    def probability(self, theta):
        """
        Calcule P(réussite | theta) pour chaque item.

        Paramètres
        ----------
        theta : float ou array (n_persons,)
            Compétence(s) du ou des candidats.

        Retourne
        --------
        P : array (n_persons, n_items) ou (n_items,)
            Probabilité de réussite.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, mask=None):
        """
        Estime les paramètres du modèle à partir des données.

        Paramètres
        ----------
        X    : array (n_persons, n_items) de 0/1
            Matrice de réponses binaires.
        mask : array (n_persons, n_items) de 0/1, optionnel
            1 = réponse présente, 0 = donnée manquante.
            Si None, toutes les réponses sont considérées présentes.

        Retourne
        --------
        self : pour chaîner model.fit(X).score(X)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Méthodes communes — héritées par toutes les sous-classes
    # ------------------------------------------------------------------

    def log_likelihood(self, X, mask=None):
        """
        Calcule la log-vraisemblance des données étant donné
        les paramètres actuels du modèle.

        On utilise le logarithme pour éviter les underflows numériques
        (multiplier des milliers de petits nombres donne 0 en float).

        Paramètres
        ----------
        X    : array (n_persons, n_items)
        mask : array (n_persons, n_items), optionnel

        Retourne
        --------
        ll : float — log-vraisemblance totale
        """
        if not hasattr(self, 'theta_'):
            raise RuntimeError("Appelle fit() avant log_likelihood().")

        X = np.asarray(X, dtype=float)
        n_persons, n_items = X.shape

        if mask is None:
            mask = np.ones_like(X)
        mask = np.asarray(mask, dtype=float)

        # P a shape (n_persons, n_items)
        P = self.probability(self.theta_)
        P = np.clip(P, 1e-9, 1 - 1e-9)  # éviter log(0)

        # log-vraisemblance item par item, pondérée par le masque
        ll_matrix = (
            X * np.log(P) + (1 - X) * np.log(1 - P)
        ) * mask

        return float(np.sum(ll_matrix))

    def information(self, theta):
        """
        Fonction d'information de l'item (IIF).

        Pour le 1PL et 2PL : I(theta) = a² × P(theta) × (1 - P(theta))
        Le 1PL a a=1 donc I(theta) = P × (1-P).

        Paramètres
        ----------
        theta : float ou array

        Retourne
        --------
        I : array — information à chaque niveau theta
        """
        P = self.probability(theta)
        return P * (1 - P)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _validate_input(self, X, mask=None):
        """Vérifie et convertit les entrées."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X doit être 2D (n_persons, n_items), reçu shape {X.shape}")
        if not np.all(np.isin(X[mask.astype(bool) if mask is not None else np.ones_like(X, dtype=bool)], [0, 1])):
            raise ValueError("X doit contenir uniquement des 0 et des 1.")
        if mask is not None:
            mask = np.asarray(mask, dtype=float)
            if mask.shape != X.shape:
                raise ValueError("mask doit avoir la même shape que X.")
        return X, mask

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"

    def get_params(self):
        """Retourne les hyperparamètres du modèle (avant fit)."""
        return {}
