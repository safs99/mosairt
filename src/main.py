import numpy as np
from irt.models.model_1pl import Rasch
# from irt.models.model_2pl import TwoPL     ← décommente quand prêt
# from irt.models.model_3pl import ThreePL   ← décommente quand prêt
# from irt.models.model_4pl import FourPL    ← décommente quand prêt
from irt.utils.evaluation import cross_validate


def compute_metrics(predicted, expected):
    """
    Calcule accuracy, precision, recall, F1 à partir de deux listes.

    Exemple du prof :
        predicted = [1, 0, 0, 0, 1, 1]
        expected  = [0, 0, 0, 1, 1, 1]
        precision = 2/3
        recall    = 2/3
        accuracy  = 4/6
    """
    predicted = np.array(predicted)
    expected  = np.array(expected)

    VP = int(np.sum((predicted == 1) & (expected == 1)))
    VN = int(np.sum((predicted == 0) & (expected == 0)))
    FP = int(np.sum((predicted == 1) & (expected == 0)))
    FN = int(np.sum((predicted == 0) & (expected == 1)))
    total = VP + VN + FP + FN

    accuracy  = (VP + VN) / total if total > 0 else 0
    precision = VP / (VP + FP)    if (VP + FP) > 0 else 0
    recall    = VP / (VP + FN)    if (VP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        "VP": VP, "VN": VN, "FP": FP, "FN": FN,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "total": total
    }


def run_experiment(model_class, name):
    print(f"\n" + "="*40)
    print(f" EXPÉRIENCE : Modèle {name}")
    print("="*40)

    # Données simulées — 100 personnes, 10 items
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(100, 10))

    #Validation croisée

    print(f"Lancement de la validation croisée pour {name}...")
    results = cross_validate(model_class, X, k=5, verbose=False)

    print(f"\n--- RÉSULTATS VALIDATION CROISÉE {name} ---")
    print(f"Accuracy moyenne : {results['accuracy_mean']:.4f}")
    print(f"F1-Score moyen   : {results['f1_mean']:.4f}")
    print(f"Brier Score      : {results['brier_score_mean']:.4f}")

    
    #Predicted vs Expected

    print(f"\n--- PREDICTED vs EXPECTED ({name}) ---")

    model = model_class()
    model.fit(X)

    # Pour chaque paire (étudiant e, activité a) → predicted = predict(e, a)
    predicted_performances = model.predict().flatten().tolist()
    expected_performances  = X.flatten().astype(int).tolist()

    print(f"Prédictions (20 premières) : {predicted_performances[:20]}")
    print(f"Réalité     (20 premières) : {expected_performances[:20]}")

    # Métriques
    m = compute_metrics(predicted_performances, expected_performances)

    print(f"\nVP={m['VP']}, VN={m['VN']}, FP={m['FP']}, FN={m['FN']}")
    print(f"Accuracy  : {m['VP']+m['VN']}/{m['total']} = {m['accuracy']:.4f}")
    print(f"Precision : {m['VP']}/{m['VP']+m['FP']} = {m['precision']:.4f}")
    print(f"Recall    : {m['VP']}/{m['VP']+m['FN']} = {m['recall']:.4f}")
    print(f"F1-Score  : {m['f1']:.4f}")

    
    #Comparaison EM et GD
    if name == "1PL / Rasch":
        _compare_em_gd(X)


def _compare_em_gd(X):
    """Compare les résultats EM vs Gradient Descent sur le modèle 1PL."""
    print(f"\n--- COMPARAISON EM vs GRADIENT DESCENT (1PL) ---")

    expected = X.flatten().astype(int).tolist()

    # EM
    model_em = Rasch(method='em', verbose=False)
    model_em.fit(X)
    pred_em = model_em.predict().flatten().tolist()
    m_em    = compute_metrics(pred_em, expected)

    # Gradient Descent
    model_gd = Rasch(method='gd', n_iter=1000, learning_rate=0.1, verbose=False)
    model_gd.fit(X)
    pred_gd = model_gd.predict().flatten().tolist()
    m_gd    = compute_metrics(pred_gd, expected)

    print(f"\n{'':>15} {'EM':>10} {'GD':>10}")
    print("-" * 37)
    print(f"  Accuracy   {m_em['accuracy']:>10.4f} {m_gd['accuracy']:>10.4f}")
    print(f"  Precision  {m_em['precision']:>10.4f} {m_gd['precision']:>10.4f}")
    print(f"  Recall     {m_em['recall']:>10.4f} {m_gd['recall']:>10.4f}")
    print(f"  F1-Score   {m_em['f1']:>10.4f} {m_gd['f1']:>10.4f}")

    print(f"\nDifficultés b estimées :")
    print(f"  {'Item':>6} {'b_EM':>10} {'b_GD':>10}")
    print("  " + "-" * 28)
    for j in range(len(model_em.b_)):
        print(f"  {j+1:>6} {model_em.b_[j]:>+10.3f} {model_gd.b_[j]:>+10.3f}")


if __name__ == "__main__":
    run_experiment(Rasch, "1PL / Rasch")
    # run_experiment(TwoPL,   "2PL")  
    # run_experiment(ThreePL, "3PL")   
    # run_experiment(FourPL,  "4PL") 