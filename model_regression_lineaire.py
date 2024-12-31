import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importer le fichier CSV en spécifiant le séparateur
data = pd.read_csv("entrainement1.csv", encoding="ISO-8859-1", sep=";")  # Utilisation du point-virgule comme séparateur

# Vérifier les colonnes pour s'assurer que les noms sont corrects
print(data.columns)

# Extraire les colonnes nécessaires (en s'assurant que les noms sont exacts)
PIB_par_habitant = data["PIB_par_habitant"].values
Satisfaction_vie = data["Satisfaction_vie"].values

# Fonction pour calculer les paramètres de la régression linéaire
def calculer_regression_lineaire(x, y):
    # Moyennes
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Covariance et Variance
    covariance = np.sum((x - mean_x) * (y - mean_y))
    variance = np.sum((x - mean_x) ** 2)

    # Calcul des coefficients
    theta1 = covariance / variance
    theta0 = mean_y - theta1 * mean_x

    return theta0, theta1

# Entraînement
theta0, theta1 = calculer_regression_lineaire(PIB_par_habitant, Satisfaction_vie)

# Affichage des résultats
print(f"Intercept (θ0) : {theta0}")
print(f"Pente (θ1) : {theta1}")

# Visualisation des données et de la régression
plt.scatter(PIB_par_habitant, Satisfaction_vie, color="blue", label="Données")
plt.plot(PIB_par_habitant, theta0 + theta1 * PIB_par_habitant, color="red", label="Modèle linéaire")
plt.xlabel("PIB par habitant")
plt.ylabel("Satisfaction à l'égard de la vie")
plt.legend()
plt.show()
