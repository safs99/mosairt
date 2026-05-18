import numpy as np
from .base import IRTModel
from ..estimation.em_algo import run_em, m_step_3pl, gauss_hermite_quadrature

class ThreePL(IRTModel):
    def __init__(self, n_iter=100, tol=1e-4, n_quad=21, verbose = False):
        self.n_iter = n_iter
        self.tol = tol
        self.n_quad = n_quad
        self.verbose = verbose
        self.a = None
        self.b = None
        self.c = None
    
    def prob(self, theta, a, b, c):
        """Formule du modèle à 3 paramètres : c + (1-c) * Sigmoïd(theta - a - b)"""
        base_prob = 1.0/(1.0 + (np.exp(-a * (theta - b))))
        return c + (1-c) * base_prob
    
    def prob_wrapped(self, theta, params):
        a, b, c = params
        return self.prob(theta, a, b, c)

    def fit(self, X, mask = None):
        """Estime a, b, c à partir de l'algo EM"""
        n_items = X.shape[1]
        quad_pts, quad_wts = gauss_hermite_quadrature(self.n_quad)
        
        #initialisation (discrimination à 1, difficulté à 0 et chance à 0.2 (une chance sur 5))

        init_a = np.ones(n_items)
        init_b = np.zeros(n_items)
        init_c = np.full(n_items, 0.2)
        init_params = (init_a, init_b, init_c)

        #etape M propre au modele 3pl
        def m_step_wrapper(r, f, pts, wts):
            #on recupère a et b maj
            a_new, b_new, c_new = m_step_3pl(r, f, self.a if self.a is not None else init_a, 
                                      self.b if self.b is not None else init_b, 
                                      self.c if self.c is not None else init_c, pts)
            return(a_new, b_new, c_new)
        
        #on lance la boucle EM
        
        final_params, converged, iters, ll = run_em(
            X, mask, self._prob_wrapper, m_step_wrapper, 
            init_params, quad_pts, quad_wts, 
            n_iter=self.n_iter, tol=self.tol, verbose=self.verbose
        )

        #on stocke les resultats

        self.a_, self.b_, self.c_ = final_params
        self.converged_ = converged
        self.iters_ = iters
        self.log_likelihood_ = ll

        return self



