import numpy as np
from irt.models.model_1pl import Rasch
# from irt.models.model_2pl import TwoPL     ← décommente quand prêt
# from irt.models.model_3pl import ThreePL   ← décommente quand prêt
# from irt.models.model_4pl import FourPL    ← décommente quand prêt
from irt.utils.evaluation import cross_validate
from irt.estimation.gradient_descent import search_best_lr
from irt.estimation.gradient_descent import fit_gradient_descent
from irt.data_loader import OuladConverter


#Pour que le tirage au sort soit reproductible, on utilise le nb 42 
#pour avoir tourjours le même résultat
np.random.seed(42) 
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

    accuracy  = (VP + VN) / total if total > 0 else 0 #> à 0 pour éviter de diviser par 0
    precision = VP / (VP + FP)    if (VP + FP) > 0 else 0
    recall    = VP / (VP + FN)    if (VP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        "VP": VP, "VN": VN, "FP": FP, "FN": FN,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "total": total
    }


def run_experiment(model_class, name, X):
    print(f"\n" + "="*40)
    print(f" EXPÉRIENCE : Modèle {name}")
    print("="*40)

    # Données simulées — 100 personnes, 10 items
    #X = np.random.randint(0, 2, size=(100, 10)) 

    #Valisation croisée 

    """Le KFold divise les 100 candidtas en 5 groupes dits folds
    Pour chaque fold:
        - On entraine les 80 candidats (qui restent)
        - On calcule la performance sur les 20 candidats restants

    On fait la moyenne métrique des 5 folds
    Cela permet de mesurer si le modèle généralise bien à de nouvaux candidats    
    
    """

    print(f"Lancement de la validation croisée pour {name}...")
    results = cross_validate(model_class, X, k=5, verbose=False)#on affiche pas le détail de chaque fold


    print(f"\nValidation Croisée (k=5)")
    results = cross_validate(model_class, X, k=5, verbose=False)

    #Resultats de la Validation croisée
    """ On a ajouté l'écart type pour mesurer la stabilité entre les folds
    Si l'écart type est petit, cela veut dire que le modèle est stable et cohérent
    Autrement, si l'écart est grand, le modèle n'est pas stable
    """

    print(f"\n--- RÉSULTATS VALIDATION CROISÉE ({name}) ---")
    print(f"Accuracy moyenne : {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"F1-Score moyen   : {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    print(f"Brier Score      : {results['brier_score_mean']:.4f} ± {results['brier_score_std']:.4f}")



    #Predicted vs Expected


    print(f"\n--- PREDICTED vs EXPECTED ({name}) ---")

    model = model_class() #crée une instance avc les hyperparamètres par defaut
    model.fit(X) #estime b et theta sur tt les données


    
    # model.predict() retourne une matrice (100 personnes × 10 items) de 0/1.
    #flatten() retourne une matrice 1D de 1000 valeurs
    # Pour chaque paire (étudiant e, activité a) → predicted = predict(e, a)
    predicted_performances = model.predict().flatten().tolist()
    expected_performances  = X.flatten().astype(int).tolist()

    # On affiche les 20 premiers
    print(f"Prédictions (20 premières) : {predicted_performances[:20]}")
    print(f"Réalité     (20 premières) : {expected_performances[:20]}")

    # Métriques calculés sur les 1000 paires (theta, b)
    m = compute_metrics(predicted_performances, expected_performances)

    print(f"\nVP={m['VP']}, VN={m['VN']}, FP={m['FP']}, FN={m['FN']}")
    print(f"Accuracy  : {m['VP']+m['VN']}/{m['total']} = {m['accuracy']:.4f}")
    print(f"Precision : {m['VP']}/{m['VP']+m['FP']} = {m['precision']:.4f}")
    print(f"Recall    : {m['VP']}/{m['VP']+m['FN']} = {m['recall']:.4f}")
    print(f"F1-Score  : {m['f1']:.4f}")

    
    #Comparaison EM et GD sur le modèle de rasch
    if name == "1PL / Rasch":
        _compare_em_gd(X)


def _compare_em_gd(X):
    """Compare les résultats EM vs Gradient Descent sur le modèle 1PL.
    Structure: 
        1. Entrainement EM en b_em de réference    
        2. Recherche meilleur learning rate
        3. Entrainement GD avec le meilleur learning rate
        4. Comparaison des résultats
        5. Test minimum local avec 5 seeds différents    
    
    
    """
    print(f"\n--- COMPARAISON EM vs GRADIENT DESCENT (1PL / Rasch) ---")

    expected = X.flatten().astype(int).tolist()

    # EM
    np.random.seed(42)
    model_em = Rasch(method='em', verbose=False)
    model_em.fit(X)
    pred_em = model_em.predict().flatten().tolist()
    m_em    = compute_metrics(pred_em, expected)

    #Recherche meilleur learning rate
    #plus rmse est petit, plus  gd et em se resseblent donc lr bon
    print(f"\n--- RECHERCHE MEILLEUR LEARNING RATE (1PL / Rasch) ---")
    search_results = search_best_lr(
        X, 
        b_em=model_em.b_, #b de reference
        lrs=(0.01, 0.05, 0.1, 0.5), #on test tel taux d'apprentissage
        epochs_list=(1000, 5000, 10000, 20000) #on teste tel nombre d'epoch
        )

    # on prend la meilleure combinaisoin
    best_lr = search_results['best_lr']
    best_epochs = search_results['best_epochs']

    #GD avec le meilleur learning rate
    #reproductibilité existante car seed = 42 (theta tjr initialisé de la même manière)

    print(f"\n--- ENTRAINEMENT GD lr={best_lr}, epochs={best_epochs} ---")
    np.random.seed(42)
    model_gd = Rasch(
        method='gd', 
        n_iter=best_epochs, 
        learning_rate=best_lr, 
        verbose=False
        )
    model_gd.fit(X)
    pred_gd = model_gd.predict().flatten().tolist()
    m_gd    = compute_metrics(pred_gd, expected)

    #Résultats

    print(f"\n{'':>15} {'EM':>10} {'GD':>10}")
    print("-" * 37)
    print(f"  Accuracy   {m_em['accuracy']:>10.4f} {m_gd['accuracy']:>10.4f}") #si se ressemblent alors font les mm prédictions
    print(f"  Precision  {m_em['precision']:>10.4f} {m_gd['precision']:>10.4f}")
    print(f"  Recall     {m_em['recall']:>10.4f} {m_gd['recall']:>10.4f}")
    print(f"  F1-Score   {m_em['f1']:>10.4f} {m_gd['f1']:>10.4f}")

    print(f"\n--- DIFFICULTES b ESTIMEES ---")
    print(f"  {'Item':>6} {'b_EM':>10} {'b_GD':>10}") #si se ressemblent alors ont estimés les mm difficultés
    print("  " + "-" * 28)
    for j in range(len(model_em.b_)):
        print(f"  {j+1:>6} {model_em.b_[j]:>+10.3f} "
              f"{model_gd.b_[j]:>+10.3f}")
    
    rsme = np.sqrt(np.mean((model_em.b_ - model_gd.b_) ** 2))
    print(f"\n Rmse entre b_EM et b_GD : {rsme:.4f}")
 
    print(f"\n--- MINIMUM LOCAL ---")
    print("--- TEST AVEC 5 INITIALISATIONS DIFFERENTES ---")
    seeds  = [0, 1, 7, 21, 99] #chaque seed initialise les params de theta à des val. différentes
    b_results = []


    for seed in seeds:
        np.random.seed(seed) #change de point de départs 
        _, b_tmp, loss_hist = fit_gradient_descent(
            X, n_epochs=best_epochs, learning_rate=best_lr, verbose=False
        )
        b_results.append(b_tmp) #stock où on arrive 
        print(f"  seed={seed:>3} | loss finale={loss_hist[-1]:.6f} "
              f"| b[0]={b_tmp[0]:+.3f}") #à chaque point de départ, on regarde où on termine
 
    b_std = np.std(b_results, axis=0) #calcule la disperstion (l'ecart type calcule a quel poiints les resultats sont diff les un des autres)
    print(f"\n Écart-type des b selon la seed : {b_std.mean():.4f}")

    # Si les résultats sont similaires, on peut dire que le minimum local n'existe pas
    # Si les résultats sont diff, on peut dire que le minimum local existe
    if b_std.mean() < 0.1:
        print("Pas de minimum local détecté : GD converge "
              "vers la même solution quelle que soit l'initialisation.")
    else:
        print("Minimum local possible : les résultats varient "
              "selon l'initialisation.") #il y a plusieurs creux 

def run_experiment_entry():
    """Fonction appelée automatiquement par la commande run-mosairt"""
    print("\n--- CHARGEMENT DE DONNEES OULAD ---")

    #chargement des données
    df_oulad = OuladConverter.to_irt_matrix("data/studentAssessment.csv", score_threshold=40)

    #conversion en matrice Numpy
    X_real = df_oulad.values

    #on prend seulement les 1500 premiers exemples
    X_real = X_real[:150, :]

    run_experiment(Rasch, "1PL / Rasch", X_real)
    # run_experiment(TwoPL,   "2PL")  
    # run_experiment(ThreePL, "3PL")   
    # run_experiment(FourPL,  "4PL") 

if __name__ == "__main__":
    run_experiment_entry()