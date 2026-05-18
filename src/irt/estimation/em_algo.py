import numpy as np
from scipy.optimize import minimize

"""L'algo fait tourner l'EM pour trouver a, b, c, d"""


def gauss_hermite_quadrature(n_squad=21):
    """calcule les points et les poids de quadrature pour intégrer sur theta
    
    Dans l'IRT, theta est inconnu — c'est une variable latente.
        Pour calculer la vraisemblance, on doit intégrer sur TOUTES
        les valeurs possibles de theta :
            L(b) = ∫ P(X | theta, b) * P(theta) dtheta
 
        Cette intégrale n'a pas de solution analytique (on ne peut
        pas la calculer à la main). On l'approche numériquement.
 
        La quadrature de Gauss-Hermite choisit 21 points et poids
        OPTIMAUX pour approcher cette intégrale gaussienne.
        C'est bien mieux qu'une grille uniforme — 21 points bien
        choisis > 100 points au hasard en termes de précision.
 
    TRANSFORMATION :
        hermgauss() donne des points pour exp(-x²).
        On transforme pour N(0,1) — le prior sur theta :
            theta = sqrt(2) * x
            poids /= sqrt(pi)"""
    raw_pts, raw_wts = np.polynomial.hermite.hermgauss(n_squad)
    quad_pts = np.sqrt(2) * raw_pts #transformation vers N(0,1) ///// quad_pts retourne les 21 valeurs de theta à tester
    quad_wts = raw_wts / np.sqrt(np.pi) #normalisation ///// quad_wts retourne le poid de chaque val (somme = 1)
    return quad_pts, quad_wts

#ETAPE E
# Le but est de répondre à : Si les paramètres actuels (b, a...) sont corrects,
#quelle compétence theta est-ce qu'on ATTEND pour chaque
# individu, et avec quelle probabilité
def e_step(X, mask, prob_func, params, quad_pts, quad_wts):
    """
    Étape E de l'algorithme EM.
 
    Pour chaque individu j et chaque point de quadrature k,
        elle calcule la probabilité que theta_j = theta_k étant
        donné ses réponses observées. C'est le POSTÉRIEUR.

    Puis calcule les counts attendus r et f qui seront consommés par l'étape M.

    Paramètres : 

    X = matrice (n_persons, n_items) de réponses 0/1
    mask = idem, 1 = reponse existante, 0 = reponse manquante
    prob_func = formule P(théta|params) du modèle (change pour chaque model)
    quad_pts = 21 points de quadrature theta
    quad_wts = poids de quadrature
    """
    eps = 1e-9 #pour eviter log(0)

    # P_grid[k, j] = P(réussir item j | theta = theta_k, params)

    P_grid   = prob_func(quad_pts, params)
    P_grid = np.clip(P_grid, eps, 1 - eps)  #on clip log(0) pour eviter de crash
    log_P = np.log(P_grid) #log de la proba
    log_1mP = np.log(1 - P_grid) #log de 1 - proba
    log_lik  = (np.einsum('ji,ki,ji->jk', X,     log_P,   mask) + 
                np.einsum('ji,ki,ji->jk', 1 - X, log_1mP, mask)) #np.einsum c une opération matricielle qui remplace 3 boucles imbriqqués pour i, j , k
    log_post = log_lik + np.log(quad_wts[None, :])

    #stabilité piur eviter de diviser par 0 (car np.exp aurait des val nég qui serait = à 0)
    log_post -= log_post.max(axis=1, keepdims=True)
    posterior = np.exp(log_post)
    posterior /= posterior.sum(axis=1, keepdims=True)
    # "combien de bonnes réponses attend-on à l'item j au niveau theta_k ?"
    r = np.einsum('ik,ij,ij->kj', posterior, X, mask)
    # "combien de passages attend-on à l'item j au niveau theta_k ?"
    f = np.einsum('ik,ij->kj', posterior, mask)

    return posterior, r, f #retourne posterior, r, f ( pour l'individu j, la proba que son theta soit le point k)

#ETAPE M

def m_step_1pl(r, f, b_current, quad_pts):
    """
    Étape M pour le modèle 1PL (Rasch).
 
    Trouve le b_j qui maximise la vraisemblance marginale
    pour chaque item j, en fixant theta (via les counts r et f).
    """
 
    eps = 1e-9 
    n_items = r.shape[1]
    b_new = np.zeros(n_items)

    for j in range(n_items):
        rj = r[:, j] #bonnesz réponses attendues pour l'item j
        fj = f[:, j] #le nb de réponses  attendus pour l'item j

        def neg_ll(b_j, rj=rj, fj=fj):#neg_ll fonction a minimiser pour optimiser b_j
            """ 1PL  : P = sigmoid(theta - b)"""
            P = 1.0/ (1.0 + np.exp(-(quad_pts - b_j[0])))
            P = np.clip(P, eps, 1 - eps)
            return -np.sum(rj * np.log(P) + (fj - rj) * np.log(1 - P))
        res = minimize(neg_ll, x0=[b_current[j]],
                   method='L-BFGS-B', bounds=[(-6, 6)])
        b_new[j] = res.x[0]

    return b_new
def m_step_2pl(r, f, a_current, b_current, quad_pts):
    """
    Étape M pour le modèle 2PL (2PL).
 
    Trouve le b_j et a_j qui maximise la vraisemblance marginale
    pour chaque item j, en fixant theta (via les counts r et f).
    """
 
    eps = 1e-9
    n_items = r.shape[1]
    b_new = np.zeros(n_items)
    a_new = np.zeros(n_items)

    for j in range(n_items):
        rj = r[:, j]
        fj = f[:, j]

        def neg_ll(params, rj=rj, fj=fj):
            a_j, b_j = params
            """ 2PL : P = sigmoid( a * (theta - b))"""
            P = 1.0 / (1.0 + np.exp(-a_j * (quad_pts - b_j)))
            P = np.clip(P, eps, 1 - eps)
            return -np.sum(rj * np.log(P) + (fj - rj) * np.log(1 - P))
 
        res = minimize(neg_ll,
                            x0=[a_current[j], b_current[j]],
                            method='L-BFGS-B',
                            bounds=[(0.01, 5), (-6, 6)])  # a > 0 obligatoire
        a_new[j] = res.x[0]
        b_new[j] = res.x[1]
 
    return a_new, b_new

def m_step_3pl(r, f, a_current, b_current, c_current, quad_pts):
    """Etape M pour le modèle 3PL (3PL).
 
    Trouve le b_j, a_j et c_j qui maximise la vraisemblance marginale
    pour chaque item j, en fixant theta (via les counts r et f).
    """

    eps = 1e-9
    n_items = r.shape[1]
    a_new = np.zeros(n_items)
    b_new = np.zeros(n_items)
    c_new = np.zeros(n_items)

    for j in range(n_items):
        rj = r[:, j]
        fj = f[:, j]

        def neg_ll(params, rj=rj, fj=fj):
            a_j, b_j, c_j = params
            """ 3PL : P = sigmoid(theta - a - b - c)"""
            base = 1.0 / (1.0 + np.exp(-a_j * (quad_pts - b_j)))
            P = c_j + (1 - c_j) * base
            P = np.clip(P, eps, 1 - eps)
            return -np.sum(rj * np.log(P) + (fj - rj) * np.log(1 - P))
        
        res = minimize(neg_ll, x0=[a_current[j], b_current[j], c_current[j]],
                   method='L-BFGS-B', bounds=[(0.01, 5), (-6, 6), (0, 1)]) #a>0 et b appartint a [-6,6] 
        a_new[j] = res.x[0]
        b_new[j] = res.x[1]
        c_new[j] = res.x[2]

    return b_new, a_new, c_new

def m_step_4pl(r, f, a_current, b_current, c_current, d_current, quad_pts):
    """Etape M pour le modèle 4PL (4PL).
 
    Trouve le b_j, a_j, c_j et d_j qui maximise la vraisemblance marginale
    pour chaque item j, en fixant theta (via les counts r et f).s"""

    eps = 1e-9
    n_items = r.shape[1]
    a_new = np.zeros(n_items)
    b_new = np.zeros(n_items)
    c_new = np.zeros(n_items)
    d_new = np.zeros(n_items)

    for j in range(n_items):
        rj = r[:, j]
        fj = f[:, j]

        def neg_ll(params, rj=rj, fj=fj):
            a_j, b_j, c_j, d_j = params
            """ 4PL : P = sigmoid(theta - a - b - c - d)"""
            base = 1.0 / (1.0 + np.exp(-a_j * (quad_pts - b_j)))
            P    = c_j + (d_j - c_j) * base   # plancher c, plafond d
            P    = np.clip(P, eps, 1 - eps)
            return -np.sum(rj * np.log(P) + (fj - rj) * np.log(1 - P))
 
        res = minimize(neg_ll, x0=[a_current[j], b_current[j],
                                c_current[j], d_current[j]],
                                method='L-BFGS-B',
                                bounds=[(0.01, 5), (-6, 6),
                                (0.0, 0.4), (0.6, 1.0)])
        a_new[j] = res.x[0]
        b_new[j] = res.x[1]
        c_new[j] = res.x[2]
        d_new[j] = res.x[3]
 
    return a_new, b_new, c_new, d_new

def compute_eap(X, mask, prob_func, params, quad_pts, quad_wts):
    """
    Calcule l'estimation EAP de theta pour chaque individu avec les paramètres 
    finaux du modèle.

    EAP = moyenne de la distribtion postérieure de theta
    Calcule l'estimation EAP (Expected A Posteriori) de theta.
        L'EAP prend la MOYENNE de la distribution postérieure —
        c'est plus stable et c'est l'estimateur optimal au sens
        de l'erreur quadratique moyenne.
 
    CE QUE ÇA FAIT :
        Avec les paramètres b finaux estimés par l'EM, on fait
        une dernière passe de l'étape E pour avoir le postérieur
        de theta pour chaque individu.
        Puis on calcule la moyenne et l'écart-type de ce postérieur.
 
    RETOURNE :
        theta_eap : array (n_persons,)
                    → compétence estimée de chaque candidat
                    → c'est la MOYENNE du postérieur de theta
 
        theta_se  : array (n_persons,)
                    → erreur standard sur l'estimation de theta
                    → c'est l'ÉCART-TYPE du postérieur
                    → plus c'est petit, plus on est certain
 
        posterior : array (n_persons, 21)
                    → distribution complète (utile pour visualisation)


    """

    #Réutilisation e_step avc les paramètres finaux
    posterior, _, _ = e_step(X, mask, prob_func, params, quad_pts, quad_wts)

    #EAP = moyenne pondérée des points de quadrature
    theta_eap = posterior @ quad_pts #ft le calcul en une opération matricielle
    
    #Erreur standard : Ecart-type du postrieur
    theta_se = np.sqrt(posterior@(quad_pts**2) - theta_eap**2)

    return theta_eap, theta_se, posterior

# log vraisemblance marginale

def marginal_log_likelihood(X, mask, prob_func, params, quad_pts, quad_wts):

    """Calcule la log-vraisemblance marginale
    
    Theta est inconnu, on l'intégre pour avoir une vraisemblance qui dépend que des params
    Utilisée pour : 
    - le critère de convergence de l'algorithme EM
    - comparer des modèles 
    """

    eps = 1e-9
    P_grid = prob_func(quad_pts, params)
    P_grid = np.clip(P_grid, eps, 1 - eps)
    log_P = np.log(P_grid)
    log_1mP = np.log(1 - P_grid)
    log_lik = (np.einsum('ji,ki,ji->jk', X, log_P, mask) +
                np.einsum('ji,ki,ji->jk', 1 - X, log_1mP, mask))

    log_wts = np.log(quad_wts)
    log_lik_w = log_lik + log_wts[None, :]
    max_val = log_lik_w.max(axis=1, keepdims=True)
    ll_per_person = (np.log(np.exp(log_lik_w - max_val).sum(axis=1)) + max_val.squeeze())
    return float(ll_per_person.sum())

def run_em(X, mask, prob_func, m_step_func, init_params, quad_pts, quad_wts, n_iter=100, tol=1e-4, verbose=False):
    """ Boucle EM utilisée pour tous les modèles IRT
    Etapes: 
    1 Initialisation des paramètres
    2 Répetition jusqu'à convergence
        2.1 E-Step: Calcule le postérior de theta pour chaque individu
        2.2 M-Step: Met à jour les nouveaux paramètres
        2.3 Calcul de la log-vraisemblance marginale
        2.4 Test de convergence, si la ll n'a pas bougé de + de tol on arrete
    3 Retourne les paramètres finaux et la log-vraisemblance marginale
    """
    params = init_params.copy()
    ll_prev = -np.inf

    for iteration in range(n_iter):
        _, r, f = e_step(X, mask, prob_func, params, quad_pts, quad_wts)
        params = m_step_func(r, f, params, quad_pts)
        ll = marginal_log_likelihood(X, mask, prob_func, params, quad_pts, quad_wts)

        if verbose:
            print(f"Iteration {iteration+1:3d}: log-likelihood = {ll:.4f} | delta = {ll-ll_prev:.6f}" )

        if abs(ll - ll_prev) < tol:
            if verbose:
                print(f"Convergence à l'itération {iteration+1}.")
            return params, True, iteration+1, ll

        ll_prev = ll
    if verbose:
        print("Attention : max itérations atteint sans cenvergences")

    return params, False, n_iter, ll