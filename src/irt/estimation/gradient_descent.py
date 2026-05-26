import numpy as np

RANDOM_SEED = 42

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def predict_proba(theta, b):
    return sigmoid(theta[:, None] - b[None, :])

def compute_loss(X, P, mask):
    """ On calcule le loss (log likelihood) de notre modélisation
    Le mask permet d'ignorer les observations manquantes et de calculer sur 
    des données observées.
    """
    eps = 1e-9
    P = np.clip(P, eps, 1 - eps)
    loss_matrix = -(X * np.log(P) + (1 - X) * np.log(1 - P))

    #on applique le mask et on moyenne sur les données observées

    n_observed = np.sum(mask)
    if n_observed == 0 :
        return 0
    n_observed = np.maximum(np.sum(mask), 1)
    return float(np.sum(loss_matrix * mask) / n_observed)

def compute_gradients(X, P, mask):
    """Calcul des gradients de la loss par rapport à theta et b"""
    n_observed = np.maximum(np.sum(mask), 1)
 
    # Erreur pour chaque pairs  
    # On applique le masque pour ignorer les données manquantes
    error = (P - X) * mask   # shape (n_persons, n_items)
 
    # grad_theta : moyenne des erreurs de l'étudiant i sur ses items
    grad_theta = np.sum(error, axis=1) / n_observed  # (n_persons,)
 
    # grad_b : moins la moyenne des erreurs de l'item j sur ses étudiants
    grad_b = -np.sum(error, axis=0) / n_observed     # (n_items,)
 
    return grad_theta, grad_b

def fit_gradient_descent(X, mask=None, n_epochs=5000, learning_rate=0.1,
                          verbose=False, verbose_every=100):
    """
    Estime theta et b par descente de gradient.
 
    ALGORITHME :
        1. Initialisation de theta (aléatoire) et b (zéros)
        2. Pour chaque epoch :
            a. Calcul des probabilités prédites P
            b. Calcul de la loss
            c. Calcul des gradients
            d. Mise à jour : theta -= lr * grad_theta
                             b     -= lr * grad_b
        3. Retourne les paramètres estimés

    """
    X = np.asarray(X, dtype=float)
    n_persons, n_items = X.shape
 
    if mask is None:
        mask = np.ones_like(X, dtype=float)
    mask = np.asarray(mask, dtype=float)
 
    # 1. Initialisation
    np.random.seed(42)  # reproductibilité
    theta = np.random.random(n_persons) #valeurs aléatoire entre 0 et 1 
    b     = np.zeros(n_items) #0
 
    loss_history = []
 
    #2. Boucle d'entraînement
    for epoch in range(n_epochs):
 
        #2.a Calcul de la probabilité
        P = predict_proba(theta, b)   # (n_persons, n_items)
 
        #2.b calcul du loss
        loss = compute_loss(X, P, mask)
        loss_history.append(loss)
 
        #2.c Calcul des gradients
        grad_theta, grad_b = compute_gradients(X, P, mask)
 
        # 3. Mise à jour des params
        theta -= learning_rate * grad_theta
        b     -= learning_rate * grad_b
 
        if verbose and (epoch % verbose_every == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:4d}/{n_epochs} | Loss = {loss:.6f}")
 
    if verbose:
        print(f"\nEntraînement terminé. Loss finale = {loss_history[-1]:.6f}")
 
    return theta, b, loss_history

def search_best_lr(X, mask=None, b_em=None,
                   lrs=(0.01, 0.05, 0.1, 0.5, 1.0),#on cherche le meilleur taux d'apprentissage pour que em et gd se ressemblent le +
                   epochs_list=(1000, 3000, 5000)):#on cherche le meilleur nb d'epoch pour savoir quel est le meilleur
    if mask is None:
        mask = np.ones_like(X, dtype=float)
 
    results = []
    best_rmse = np.inf
    best_lr, best_epochs = 0.1, 5000 #val par defaut
 
    print(f"\n{'lr':>8} {'epochs':>8} {'RMSE vs EM':>12} {'loss finale':>12}")
    print("-" * 46)
 
    for lr in lrs:
        for epochs in epochs_list:
            theta, b_gd, loss_hist = fit_gradient_descent(
                X, mask=mask, n_epochs=epochs,
                learning_rate=lr, verbose=False
            )
            loss_finale = loss_hist[-1]
 
            if b_em is not None:
                #rmse entre b_em et b_gd (calcul l'ecart )
                rmse = float(np.sqrt(np.mean((b_gd - b_em) ** 2)))
                print(f"{lr:>8.3f} {epochs:>8d} {rmse:>12.4f} {loss_finale:>12.6f}")
                if rmse < best_rmse: #on prend le plus ptit rmse
                    best_rmse   = rmse
                    best_lr     = lr
                    best_epochs = epochs
            else:
                print(f"{lr:>8.3f} {epochs:>8d} {'N/A':>12} {loss_finale:>12.6f}")
 
            results.append({
                'lr': lr, 'epochs': epochs,
                'rmse_vs_em': rmse if b_em is not None else None,
                'loss_finale': loss_finale
            })
 
    if b_em is not None:
        print(f"\nMeilleur : lr={best_lr}, epochs={best_epochs}, "
              f"RMSE={best_rmse:.4f}")
 
    return {
        'best_lr':     best_lr,
        'best_epochs': best_epochs,
        'best_rmse':   best_rmse,
        'all_results': results
    }

def predict(theta, b, threshold = 0.5):
        """Predit 1 ou 0  pour une personne et un item (theta, b)
        Si P(réussite) > threshold, on predit 1 sinon 0 

        threshols : est un seuil permettant de transformer une valeur continues
        en une classification binaire (0 ou 1).

        Si P(réussite) >= threshold, la condition devient True et on predit 1,
        Si P(réussite) < threshold, la condition devient False et on predit 0 
        """
        theta = np.asarray(theta, dtype = float)
        b = np.asarray(b, dtype = float)
        if theta.ndim == 0 and b.ndim ==0:
            P = sigmoid(float(theta)-float(b))
            return int(P> threshold)
        
def evaluate(X, theta, b, mask = None):
        
        if mask is None:
            mask = np.ones_like(X, dtype = float)
        mask = mask.astype(bool)

        #Ici threshold (seuil) fixé à 0.5
        P= predict_proba(theta, b)
        X_pred = (P>=0.5).astype(int)

        #Comparaison sur les données observées
        y_true = X[mask].astype(int)
        y_pred = X_pred[mask].astype(int)

        VP = np.sum((y_pred == 1) & (y_true == 1))
        VN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        total     = VP + VN + FP + FN
        accuracy  = (VP + VN) / total if total > 0 else 0.0
        precision = VP / (VP + FP) if (VP + FP) > 0 else 0.0
        recall    = VP / (VP + FN) if (VP + FN) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)
    
        loss = compute_loss(X.astype(float), P, mask.astype(float))
    
        return {
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "loss":      round(loss, 6),
            "VP": int(VP), "VN": int(VN),
            "FP": int(FP), "FN": int(FN),
        }