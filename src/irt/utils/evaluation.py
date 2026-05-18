import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

#métriques de base

def mae_b(b_estimated, b_true):
    """MAE : Erreur absolue Moyenne
    MAE = (1/N) * sum(b estimé - b vrai) : Ce sont les erreurs moyenne en unités de theta
    """
    return float(np.mean(np.abs(b_estimated-b_true)))

def mse_b(b_estimated, b_true):
    """ MSE : Erreur quadratique Moyenne
    MSE = (1/N) * sum(b estime - b vrai) :
    """
    return float(np.mean((b_estimated - b_true)**2))

def rmse_b(b_estimated, b_true):
    """ RMSE : Racine carrée de l'Erreur Quadratique Moyenne
    RME = sqrt(MSE)
    """
    return float(np.sqrt(mse_b(b_estimated, b_true)))

def mae_theta(theta_estimated, theta_true):
    """MAE sur les compétences theta"""
    return float(np.mean(np.abs(theta_estimated - theta_true)))

def rmse_theta(theta_estimated, theta_true):
    """RMSE sur les compétences theta"""
    return float(np.sqrt(np.mean((theta_estimated - theta_true)**2)))

def r2_theta(theta_estimated, theta_true):
    """R² : Coefficient de Détermination"""
    ss_res = np.sum((theta_true - theta_estimated) ** 2)
    ss_tot = np.sum((theta_true - np.mean(theta_true)) ** 2 )
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res/ss_tot))

def correlation_theta(theta_estimated, theta_true):
    """Correlation entre theta estime et theta vrai"""
    return float(np.corrcoef(theta_estimated, theta_true)[0,1])

def accuracy(X, X_pred, mask = None):
    """ Accuracy : justesse des prédictions
    Accuracy = (VP+VN) / (VP+VN+FP+FN)
    
    X : réponses réelles
    X_pred : réponses prédites
    mask : réponse présente """

    if mask is None:
        mask = np.ones_like(X, dtype=bool) 
    mask = mask.astype(bool)
    return float(np.mean(X[mask] == X_pred[mask]))

def classification_report_irt(X, X_pred, mask=None):
    """
    Rapport complet : Accuracy, Précision, Rappel, F1.
 
    Accuracy  = (VP + VN) / total
    Précision = VP / (VP + FP)  → parmi les 1 prédits, combien vrais ?
    Rappel    = VP / (VP + FN)  → parmi les vrais 1, combien détectés ?
    F1        = 2 * P * R / (P + R) → équilibre précision/rappel
 
    En IRT :
        VP = réussite initialement prédite
        VN = échec initialement prédit
        FP = échec prédit comme réussite (surestimation)
        FN = réussite prédite comme échec (sous-estimation)
    """
    if mask is None:
        mask = np.ones_like(X, dtype=bool)
    mask = mask.astype(bool)
    y_true = X[mask].astype(int).flatten()
    y_pred = X_pred[mask].astype(int).flatten()

    return{
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

def brier_score(X, P_pred, mask=None):
    """Score de Brier : erreur quadratique entre les probabilités prédites
      et réponses réelles
    Brier score = (1/N) * sum((P_pred - X)**2)

    0.0 -> prédiction parfaite
    0.5 -> prédiction nulle
    1.0 -> prédiction inverse

    """
    if mask is None:
        mask = np.ones_like(X, dtype=bool)
    mask = mask.astype(bool)
    return float(np.mean((X[mask] - P_pred[mask])**2))

def log_loss_irt(X, P_pref, mask=None):
    """
    Log loss : pénalise les erreur de logistique
    Log loss = -(1/N) * sum[ X*log(P) + (1-X)*log(1-P) ]

    Si le modèle dit P = 0.99 mais X=0, pénalité énorme
    Si le modèle dit P = 0.5 et se trompe, pénalité modérée


    """
    if mask is None:
        mask = np.ones_like(X, dtype=bool)
    mask = mask.astype(bool)
    P = np.clip(P_pref[mask], 1e-9, 1 - 1e-9)
    X_obs = X[mask]
    return float(-np.mean(X_obs * np.log(P) + (1 - X_obs) * np.log(1 - P)))


def evaluate(model, X_train, X_test, mask_train=None, mask_test=None):
    """Evalue le modèle sur les données de train et de test.
    1. Estimation de b sur X_train
    2. Réestimation de theta sur X_test avec b fixés
    3. Calcul de toutes les métriques sur X_test

    Retourne le dictionnaire avec accuracy, precision, recall, f1, brier, log_loss
    """
    model.fit(X_train, mask=mask_train)

    #réestimation de theta sur le test avec b fixés    

    theta_test = _estimate_theta_eap(
        X_test, mask_test, model.b_, 
        model._quad_pts, model._quad_wts, model
    )


    if mask_test is None:
        mask_test = np.ones_like(X_test)
    mask_test = np.asarray(mask_test, dtype=float)

    # proba et prédictions
    #P_test = model._prob(theta_test[:, None], model.b_[None, :])
    P_test = model.probability(theta_test)
    X_pred = (P_test > 0.5).astype(int) 

    #metriques
    clf = classification_report_irt(X_test, X_pred, mask_test) 

    return{
        "accuracy": clf["accuracy"],
        "precision": clf["precision"],
        "recall": clf["recall"],
        "f1": clf["f1"],
        "brier_score": brier_score(X_test, P_test, mask_test),
        "log_loss": log_loss_irt(X_test, P_test, mask_test),
        "n_iter": model.n_iter_,
        "converged": model.converged_,
        "ll_train": model.log_likelihood_,
    }

def _estimate_theta_eap(X, mask, b, quad_pts, quad_wts, model):
    """
    Réestime theta par EAP pour de nouveaux individus
    avec des paramètres b déjà connus (fixés depuis le train).
    """
    X = np.asarray(X, dtype=float)
    if mask is None:
        mask = np.ones_like(X)
    mask = np.asarray(mask, dtype=float)
    eps  = 1e-9
 
    P_grid  = model._prob(quad_pts, b)
    log_P   = np.log(np.clip(P_grid,     eps, 1 - eps))
    log_1mP = np.log(np.clip(1 - P_grid, eps, 1 - eps))
 
    log_lik  = (np.einsum('ji,ki,ji->jk', X,     log_P,   mask) +
                np.einsum('ji,ki,ji->jk', 1 - X, log_1mP, mask))
    log_post  = log_lik + np.log(quad_wts[None, :])
    log_post -= log_post.max(axis=1, keepdims=True)
    posterior  = np.exp(log_post)
    posterior /= posterior.sum(axis=1, keepdims=True)
 
    return posterior @ quad_pts

#Validation Croisé kfold
 
def cross_validate(model_class, X, mask=None, k=5,
                   model_params=None, random_state=42, verbose=True):
    """
    Validation croisée KFold sur les personnes (lignes de X).
 
    Principe :
        On divise les N candidats en k groupes.
        Pour chaque groupe :
            - Entraînement sur les k-1 autres groupes
            - Évaluation sur ce groupe
        On moyenne les métriques sur les k folds.
 
    Retourne
    --------
    dict avec moyenne ± écart-type de chaque métrique
 
    Exemple : (a taper dans -m python)

    >>> from irt.models.model_1pl import Rasch
    >>> from irt.utils.evaluation import cross_validate
    >>> results = cross_validate(Rasch, X, k=5)
    >>> print(f"F1 moyen : {results['f1_mean']:.4f}")
    """
    if model_params is None:
        model_params = {}
 
    X = np.asarray(X, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=float)
 
    kf           = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_results = []
 
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        if verbose:
            print(f"Fold {fold_idx+1}/{k} "
                  f"— train: {len(train_idx)} | test: {len(test_idx)}")
 
        X_train    = X[train_idx]
        X_test     = X[test_idx]
        mask_train = mask[train_idx] if mask is not None else None
        mask_test  = mask[test_idx]  if mask is not None else None
 
        model   = model_class(**model_params)
        metrics = evaluate(model, X_train, X_test, mask_train, mask_test)
        fold_results.append(metrics)
 
        if verbose:
            print(f"  accuracy={metrics['accuracy']:.4f} | "
                  f"f1={metrics['f1']:.4f} | "
                  f"brier={metrics['brier_score']:.4f} | "
                  f"converged={metrics['converged']}")
            
    """ Réponses du terminal : 

Fold 1/5 — train: 8 | test: 2
  accuracy=0.4000 | f1=0.0000 | brier=0.3596 | converged=True
Fold 2/5 — train: 8 | test: 2
  accuracy=0.7000 | f1=0.6667 | brier=0.2521 | converged=True
Fold 3/5 — train: 8 | test: 2
  accuracy=0.5000 | f1=0.4444 | brier=0.2933 | converged=True
Fold 4/5 — train: 8 | test: 2
  accuracy=0.5000 | f1=0.6154 | brier=0.2492 | converged=True
Fold 5/5 — train: 8 | test: 2
  accuracy=0.7000 | f1=0.5714 | brier=0.1945 | converged=True

=======================================================
Résultats KFold (5 folds) :
  Accuracy  : 0.5600 ± 0.1200
  Précision : 0.4833 ± 0.2603
  Rappel    : 0.4476 ± 0.2405
  F1        : 0.4596 ± 0.2413
  Brier     : 0.2697 ± 0.0548
  Log-loss  : 0.7403 ± 0.1248
  Convergés : 5/5 folds
>>> print(f"F1 moyen : {results['f1_mean']:.4f}")  
F1 moyen : 0.4596

"""
    # Agrégation
    summary      = {}
    numeric_keys = ['accuracy', 'precision', 'recall',
                    'f1', 'brier_score', 'log_loss', 'll_train']
 
    for key in numeric_keys:
        vals = [r[key] for r in fold_results]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"]  = float(np.std(vals))
 
    summary['all_folds']   = fold_results
    summary['n_folds']     = k
    summary['n_converged'] = sum(r['converged'] for r in fold_results)
 
    if verbose:
        print(f"\n{'='*55}")
        print(f"Résultats KFold ({k} folds) :")
        print(f"  Accuracy  : {summary['accuracy_mean']:.4f}"
              f" ± {summary['accuracy_std']:.4f}")
        print(f"  Précision : {summary['precision_mean']:.4f}"
              f" ± {summary['precision_std']:.4f}")
        print(f"  Rappel    : {summary['recall_mean']:.4f}"
              f" ± {summary['recall_std']:.4f}")
        print(f"  F1        : {summary['f1_mean']:.4f}"
              f" ± {summary['f1_std']:.4f}")
        print(f"  Brier     : {summary['brier_score_mean']:.4f}"
              f" ± {summary['brier_score_std']:.4f}")
        print(f"  Log-loss  : {summary['log_loss_mean']:.4f}"
              f" ± {summary['log_loss_std']:.4f}")
        print(f"  Convergés : {summary['n_converged']}/{k} folds")
 
    return summary
