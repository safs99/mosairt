import numpy as np

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

def predict(theta, b, threshold = 0.5):
        theta = np.asarray(theta, dtype = float)
        b = np.asarray(b, dtype = float)
        if theta.ndim == 0 and b.ndim ==0:
            P = sigmoid(float(theta)-float(b))
            return int(P> threshold)
        
def evaluate(X, theta, b, mask = None):
        if mask is None:
            mask = np.ones_like(X, dtype = float)
        mask = mask.astype(bool)
        P= predict_proba(theta, b)
        X_pred = (P>=0.5).astype(int)

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