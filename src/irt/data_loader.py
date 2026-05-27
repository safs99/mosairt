import numpy as np
import pandas as pd

class OuladConverter:
    """
    Module pour cherger et convertir les données brutes de l'OULAD
    au format attendu par nos modèles IRT
    """
    @staticmethod
    
    def to_irt_matrix(student_assessment_path, score_threshold=40):
        """
        Prend les données brutes de l'OULAD et les transfome en une matrice binaire (0/1)
        Lignes = etudiants (id_student), colonnes = items(id_assessment)
        """
        print(f"Chargement du fichier : {student_assessment_path}")
        #1. Charge les données brutes
        df =pd.read_csv(student_assessment_path)
        #2. Nettoyer les données manquantes
        df = df.dropna(subset=['score'])
        #3. Convertir le score en success (0/1)
        df['success'] = (df['score'] > score_threshold).astype(int)

        print("Pivotement des données pour créer la matrice IRT ...")

        # 4. Pivoter la table : index = étudiants, colonnes = évaluations, valeurs = succès
        # On remplit par 0 (fillna) si un étudiant n'a pas passé une évaluation
        df_pivot = df.pivot(index='id_student', columns='id_assessment', values='success').fillna(0)
        df_pivot = df_pivot.astype(int)
        print(f"Matrice créée : {df_pivot.shape[0]} étudiants et {df_pivot.shape[1]} items.")
        
        # On retourne le DataFrame complet (pratique pour garder les IDs des étudiants)
        return df_pivot