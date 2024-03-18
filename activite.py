import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def activite_actuelle(date_initiale, radioactivite_initiale_uc, demi_vie, unite_demi_vie='jours'):
    """
    Calcule l'activité actuelle d'une source radioactive.

    Arguments :
    date_initiale : date de mesure initiale de l'élément radioactive (format : 'YYYY-MM-DD')
    radioactivite_initiale_uc : radioactivité initiale de l'élément en microcuries
    demi_vie : demi-vie de l'élément (en jours ou en années)
    unite_demi_vie : unité de la demi-vie ('jours' ou 'annees')

    Retourne :
    activite_actuelle : activité actuelle de la source radioactive en microcuries
    """

    # Convertir la date initiale en objet datetime
    date_initiale = datetime.datetime.strptime(date_initiale, '%Y-%m-%d')

    # Calculer le nombre de jours ou d'années écoulés depuis la date initiale jusqu'à la date actuelle
    date_actuelle = datetime.datetime.strptime("2024-03-11", '%Y-%m-%d')
    duree_ecoulee = (date_actuelle - date_initiale).days if unite_demi_vie == 'jours' else (date_actuelle - date_initiale).days / 365.25

    # Calculer l'activité actuelle en microcuries
    activite_actuelle = radioactivite_initiale_uc * np.exp(-0.693 * duree_ecoulee / demi_vie)

    return activite_actuelle

# Exemple d'utilisation de la fonction
date_initiale = '2023-01-31'
radioactivite_initiale_uc = .25  # microcuries
demi_vie = 30.08  # demi-vie en années ou jours
unite_demi_vie = 'annees'  # 'jours' ou 'annees'

activite = activite_actuelle(date_initiale, radioactivite_initiale_uc, demi_vie, unite_demi_vie)
activite *= 10**-6*3.7*10**10*.5*0.8  # changer le facteur géométrique 0.5
activite = int(activite)
print(f"L'activité actuelle est de {activite} désintégrations par seconde.")

nbc = 10919
t = 30.3
activite = 4512
eff_rel = round(nbc/t/activite*100, 2)
print('efficacité relative', eff_rel, '%')


# Données l'énergie gamma en keV et l'efficacité de détection relative en %
energie_gamma_keV = np.array([122, 511, 662, 835, 1173, 1275, 1332])  # énergie gamma en keV
efficacite_detection = np.array([55.53, 25.2, 7.99, 8.15, 5.66, 4.53, 3.78])  # efficacité de détection en %
counts = np.array([16522, 191828, 10919, 9334, 68005, 34494, 45481])
incertitude = np.array([562, 1623, 31, 140, 1638, 1125, 1407])
incertitude = np.sqrt(counts) + incertitude
inc_rel = incertitude/counts*efficacite_detection
print(inc_rel)

# Définition de la fonction logarithmique à ajuster
def fonction_logarithmique(x, a, b, c):
    return np.exp(a * np.log(x+c)+b)


# Ajustement de la courbe logarithmique aux données
# parametres_optimaux, covariance = curve_fit(fonction_logarithmique, np.delete(energie_gamma_keV, [2, 3]), np.delete(efficacite_detection, [2, 3]))
parametres_optimaux, covariance = curve_fit(fonction_logarithmique, energie_gamma_keV, efficacite_detection)
# Extraction des paramètres ajustés
a, b, c = parametres_optimaux

# Génération de données pour la courbe ajustée
x_courbe = np.arange(0, 1500, 100)  # Création d'un ensemble de points pour tracer la courbe ajustée
y_courbe = fonction_logarithmique(x_courbe, a, b, c)

# Tracer le graphique
plt.figure()
plt.plot(energie_gamma_keV, efficacite_detection, marker='.', markersize=5, color='b', linestyle='', label='Données expérimentales')
plt.plot(x_courbe, y_courbe, color='r', linestyle='-', label='Efficacité ajustée')


# Ajout des barres d'incertitude en T aux valeurs en y
plt.errorbar(energie_gamma_keV, efficacite_detection, yerr=inc_rel, fmt='', color='b', linestyle='')


# Configuration de l'axe des x en logarithmique
plt.xscale('log')
plt.yscale('log')

# Affichage de l'équation de la courbe ajustée
equation_courbe = f'Efficacité = {a:.2f} * log(Énergie) + {b:.2f}'
# plt.text(20, 20, equation_courbe, fontsize=12, color='r')
print(equation_courbe)

# Étiquettes des axes et titre du graphique
plt.xlabel('Énergie du rayonnement gamma (keV)')
plt.ylabel('Efficacité de détection relative (%)')

# Affichage de la légende
plt.legend()

# Affichage de la grille
plt.grid(True, which="both", ls="--")

# Affichage du graphique
plt.show()