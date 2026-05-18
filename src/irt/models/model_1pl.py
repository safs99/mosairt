import numpy as np
from .base import IRTModel
from scipy.optimize import minimize
from ..estimation.em_algo import run_em, m_step_1pl, gauss_hermite_quadrature, compute_eap
from ..estimation.gradient_descent import fit_gradient_descent


class Rasch(IRTModel):

    def __init__(self, n_iter=100, tol=1e-4, n_quad=21, 
                 verbose=False, method='em', 
                 learning_rate=0.1 ):
        
        self.n_iter   = n_iter
        self.tol      = tol
        self.n_quad   = n_quad
        self.verbose  = verbose
        self.method = method
        self.learning_rate = learning_rate

    def _prob(self, theta, b):
        """Formule du modèle de Rasch : sigmoid(theta - b)."""
        return 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))
    
    def _prob_func(self, quad_pts, params):
        b = params['b']
        return self._prob(quad_pts, b)

    def fit(self, X, mask=None):
        """Estime b et thta par l'algo EM"""

        X = np.asarray(X, dtype=float)
        n_persons, n_items = X.shape

        #Si il y a pas de masque, on suppose que les rep sont presentes
        if mask is None :
            mask = np.ones_like(X, dtype=float)
        else :
            mask = np.asarray(mask, dtype=float)

        if self.method == 'em':
            return self._fit_em(X, mask)
        elif self.method == 'gd':
            return self._fit_gd(X, mask)
        else :
            raise ValueError("La méthode doit être 'em' ou 'gd'.")
        #return self
    def _fit_em(self, X, mask):
        """Estimation par l'algo EM"""
    #Calcul de Gauss hermitze on transforme une integrale complexe de la vrai semblance
    # marginale en une somme pondérée en 21 points stratégiques

        quad_pts, quad_wts = gauss_hermite_quadrature(self.n_quad)

    #Initialisation de b par le taux de succès pour que l'algo converge plus vite

        """Un item réussi par + de 80% des personnes donne un b négatif (car b facile)
        Un item réussi par 20% des personnes donne un b positif (car b difficile) """
        success_rate = (np.sum(X*mask, axis = 0) / np.maximum(np.sum(mask, axis = 0), 1))
        success_rate = np.clip(success_rate,0.01, 0.99)
        b_init = -np.log((success_rate / (1 - success_rate)))

        if self.verbose:
            print(f"Init b ∈ [{b_init.min():.2f}, {b_init.max():.2f}]")

        #etape M
        def m_step_wrapper(r, f, params, quad_pts):
            b_new = m_step_1pl(r, f, params['b'], quad_pts)
            return {'b': b_new}
        
        #on delegue la boucle EM à run_em
        # le concept est le mm, seul le modèle change a chaque fois

        final_params, converged, n_iter_done, ll = run_em(
            X           = X,
            mask        = mask,
            prob_func   = self._prob_func,
            m_step_func = m_step_wrapper,
            init_params = {'b': b_init},
            quad_pts    = quad_pts,
            quad_wts    = quad_wts,
            n_iter      = self.n_iter,
            tol         = self.tol,
            verbose     = self.verbose,
        )

        #estimation EAP pour theta 
        """Une fois b connue, on estime theta pour chaque individu"""

        theta_eap, theta_se, _ = compute_eap(
            X, mask, self._prob_func, final_params, quad_pts, quad_wts
        )

        #stockage des resulatts

        self.b_              = final_params['b']
        self.theta_          = theta_eap
        self.theta_se_       = theta_se
        self.log_likelihood_ = ll
        self.converged_      = converged
        self.n_iter_         = n_iter_done
        self._quad_pts       = quad_pts
        self._quad_wts       = quad_wts
        self.loss_history = None
 
        return self
    
    
    
    def _fit_gd(self, X, mask):
        """Estimation de la descente de gradient"""

        n_epochs = self.n_iter if self.n_iter != 100 else 1000
        if self.verbose:
            print(f"[GD] Entraînement : {X.shape[0]} personnes, "
                  f"{X.shape[1]} items, {n_epochs} epochs, "
                  f"lr={self.learning_rate}")
 
        theta, b, loss_history = fit_gradient_descent(
            X             = X,
            mask          = mask,
            n_epochs      = n_epochs,
            learning_rate = self.learning_rate,
            verbose       = self.verbose,
            verbose_every = max(1, n_epochs // 10),
        )

        # Stockage des résultats — même convention que l'EM
        self.b_              = b
        self.theta_          = theta
        self.theta_se_       = None   # pas disponible en GD
        self.log_likelihood_ = -loss_history[-1]
        self.converged_      = True
        self.n_iter_         = n_epochs
        # Pour que evaluation.py fonctionne avec le GD aussi
        quad_pts, quad_wts   = gauss_hermite_quadrature(self.n_quad)
        self._quad_pts       = quad_pts
        self._quad_wts       = quad_wts
        self.loss_history_   = loss_history
    
    def probability(self, theta = None):

        """Calcule P(réussite | theta, b) pour chaque item.
        Méthode publique, vérifie que fit() a été appelé.
        """
        if not hasattr(self, 'b_') or self.b_ is None: #hasattr vérifie si un objet possède un attribut !!
            raise RuntimeError(
                "Le modèle n'est pas encore entraîné. "
                "Appelle model.fit(X) d'abord."
            )
        if theta is None: #si theta est none, on utilise le theta estimé dans le fit() pour generer des proba pour les etudiants
            theta = self.theta_
 
        theta = np.asarray(theta, dtype=float)

        if theta.ndim == 0: #si theta est un seul nombre
            return self._prob(np.array([float(theta)]), self.b_)[0]
        else:
            return self._prob(theta, self.b_)
        
    def predict(self, theta=None):
        """
        Prédit les réponses pour chaque candidat en 0/1
        Si P(réussite >= 0.5) → prédit 1, sinon 0

        But : calculer accuracy et F1 score
        """
        P = self.probability(theta)
        return (P >= 0.5).astype(int)
    
    def information(self, theta):
        """Ici cest la fonction d'iinformation de l'itmet
         sur theta 
         On voit quel item est le plus utile pour estimer theta
         L'info est maximale quand theta = b"""
        P = self.probability(theta)
        return P * (1 - P)
    
    def get_params(self):
        """
        Retourne les hyperparamètres du modèle.
        Utile pour la sérialisation et la validation croisée.
        Suit la convention scikit-learn.
        """
        return {
            'n_iter': self.n_iter,
            'tol':    self.tol,
            'n_quad': self.n_quad,
            'verbose': self.verbose,
        }
    
    def summary(self):
        """Intérprétation des résultats obtenus """
        if not hasattr(self, 'b') or self.b_ is None: 
            print("Appeler fit()")
            return
        print("=" * 45)
        print("RÉSUMÉ — Modèle de Rasch (1PL)")
        print("=" * 45)
        print(f"Items    : {len(self.b_)}")
        print(f"Candidats: {len(self.theta_)}")
        print(f"Etat Convergence : {self.converged_} ({self.n_iter_} itérations)")
        print(f"Fiabilité (Log-lik)  : {self.log_likelihood_:.4f}")
        print()

        #Affichage des difficultés b 
        print(f"Difficultés b :")
        for i, b in enumerate(self.b_): #si b est positif -> difficile, si b est négatif -> facile
            print(f"  Item {i+1:2d} : b = {b:+.3f}")
        print()

        #Statistiques sur theta
        print(f"Compétences theta :")
        print(f"  Moyenne  : {self.theta_.mean():.3f}")
        print(f"  Std      : {self.theta_.std():.3f}")
        print(f"  Min/Max  : {self.theta_.min():.3f} / "
              f"{self.theta_.max():.3f}")
        

    def __repr__(self):
     """ Représentation textuelle de l'objet.
        Affiché quand on tape model dans le terminal Python.
    """
     return (f"Rasch(method='{self.method}', n_iter={self.n_iter}, "
                f"tol={self.tol}, n_quad={self.n_quad})")
 
