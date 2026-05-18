import numpy as np
from .base import IRTModel
from scipy.optimize import minimize
from ..estimation.em_algo import run_em, m_step_2pl, gauss_hermite_quadrature, compute_eap
from ..estimation.gradient_descent import fit_gradient_descent
class TwoPL(IRTModel):
    def __init__(self, n_iter=100, tol=1e-4, n_quad=21, verbose=False, 
                 method='em', learning_rate=0.1 ):
        
        self.n_iter = n_iter
        self.tol = tol
        self.n_quad = n_quad
        self.verbose = verbose
        self.method = method
        self.learning_rate = learning_rate
        
        #paramètres estimés
        self.a_ = None
        self.b_ = None
        self.theta_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None

    def _prob(self, theta, a, b):
        """Formule du modèle à 2 paramètres : sigmïd(-a(theta - b))"""
        return 1.0/(1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    
    def _prob_func(self, quad_pts, params):
        a, b = params
        return self._prob(quad_pts, a, b)
    

    def fit(self, X, mask = None) : 
        
        
        X = np.asarray(X, dtype = float)
        n_persons, n_items = X.shape
        if mask is None :
            mask = np.ones_like(X, dtype=float)
        else :
            mask = np.asarray(mask, dtype=float)

        if self.method == 'em':
            return self._fit_em(X, mask)
        elif self.method == 'gd':
            return self._fit_gd(X, mask)
        else :
            raise ValueError("Method must be 'em' or 'gd'.")

        #return self
        
    def _fit_em(self, X, mask):

        quad_pts, quad_wts = gauss_hermite_quadrature()

        """Un item réussi par + de 80% des personnes donne un b négatif (car b facile)
        Un item réussi par 20% des personnes donne un b positif (car b difficile)"""
        success_rate = (np.sum(X*mask, axis = 0) / np.maximum(np.sum(mask, axis = 0), 1))
        success_rate = np.clip(success_rate,0.01, 0.99)
        b_init = -np.log((success_rate / (1 - success_rate)))
        a_init = np.ones(n_items) #on initialise a = 1
        init_params = [a_init, b_init]

        if self.verbose:
            print(f"Init a ∈ [{a_init.min():.2f}, {a_init.max():.2f}]")
            print(f"Init b ∈ [{b_init.min():.2f}, {b_init.max():.2f}]") 

        #on def la fonctioin M propre au model 2pl

        def m_step_wrapper(r, f, params, quad_pts):
            #on recupère a et b maj
            a_new, b_new = m_step_2pl(r, f, params['a','b'],quad_pts)
            return{'a': a_new, 'b': b_new}
        
        #on lance la boucle em

        final_params, converged, n_iter_done, ll = run_em(
            X           = X,
            mask        = mask,
            prob_func   = self._prob_func,
            m_step_func = m_step_wrapper,
            init_params = {'a': a_init, 'b': b_init},
            quad_pts    = quad_pts,
            quad_wts    = quad_wts,
            n_iter      = self.n_iter,
            tol         = self.tol,
            verbose     = self.verbose,
        )


        theta_eap, theta_se, _ = compute_eap(
            X, mask, self._prob_func, final_params, quad_pts, quad_wts
        )
        self.a_, self.b_ = final_params['a'], final_params['b']
        self.theta_ = theta_eap
        self.theta_se_ = theta_se
        self.log_likelihood_ = ll
        self.converged_ = converged
        self.n_iter_ = n_iter_done
        self.quad_pts = quad_pts
        self.quad_wts = quad_wts
        self.loss_history_ = None

        return self
    
    def _fit_gr(self, X, mask):
        """Estimation de la descente de gradient"""

        n_epochs = self.n_iter if self.n_iter != 100 else 1000
        if self.verbose:
            print(f"[GD] Entraînement : {X.shape[0]} personnes, "
                  f"{X.shape[1]} items, {n_epochs} epochs, "
                  f"lr={self.learning_rate}")

        theta, a, b, loss_history = fit_gradient_descent(
            X             = X,
            mask          = mask,
            n_epochs      = n_epochs,
            learning_rate = self.learning_rate,
            verbose       = self.verbose,
            verbose_every = max(1, n_epochs // 10),
        )

        #Stockage resultat
        self.a_ = a
        self.b_ = b
        self.theta_ = theta
        self.theta_se_ = None
        self.log_likelihood_ = -loss_history[-1]
        self.converged_ = True
        self.n_iter_ = n_epochs
        #Pour que evaluation.py fonctionne avec le GD aussi
        quad_pts, quad_wts = gauss_hermite_quadrature(self.n_quad)
        self._quad_pts = quad_pts
        self._quad_wts = quad_wts
        self.loss_history_ = loss_history

    def probability(self, theta):
        
        """Calcule P(réussite |theta,a,b) pour chaque item.
        Méthode publique qui vérifie que fit() a été appelé.
        """
        if not hasattr(self, 'a_', 'b_') or self.a_ and self.b_ is None:
            raise RuntimeError(
                "Le modèle n'est pas encore entraîné. "
                "Appelle model.fit(X) d'abord."
            )
        if theta is None:
            theta = self.theta_

        theta = np.asarray(theta, dtype=float)

        if theta.ndim == 0:
            return self._prob(np.array([float(theta)]), self.a_, self.b_)[0]
        else:
            return self._prob(theta, self.a_, self.b_)
        
    def predict(self, theta=None):
        
        """Calcule P(réussite |theta,a,b) pour chaque item.
        Méthode publique qui vérifie que fit() a été appelé.
        """
        P = self.probability(theta)
        return self.probability(theta)
    
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
        print(f"Discriminant  : {len(self.a_)}")
        print(f"Items    : {len(self.b_)}")
        print(f"Candidats: {len(self.theta_)}")
        print(f"Etat Convergence : {self.converged_} ({self.n_iter_} itérations)")
        print(f"Fiabilité (Log-lik)  : {self.log_likelihood_:.4f}")
        print()
    
    #Affichage du discriminant a
        print(f"Discriminants a (Pente) :")
        for i, a in enumerate(self.a_): # Plus a est élevé, plus l'item est discriminant (courbe raide)
            print(f"  Item {i+1:2d} : a = {a:.3f}")
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
        return (f"Mondel 2PL(method='{self.method}', n_iter={self.n_iter}, "
                f"tol={self.tol}, n_quad={self.n_quad})")
