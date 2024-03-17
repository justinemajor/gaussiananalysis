import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import find_peaks

y = []
element = "cs137"
f = open(f"etalon_{element}.txt", "r")
for i, line in enumerate(f):
    if 12 <= i < 4108:
        y.append(int(line))
y = np.array(y)
print(len(y))
f.close()

x = np.arange(len(y))

# Seuil
if element == "mn54":
    seuil = 50
else:
    seuil = 80

# Trouver les indices où y dépasse le seuil
indices_depassement = np.where(y > seuil)[0]

# Lissage des données pour trouver les bords de la ROI de manière plus précise
y_smoothed = np.convolve(y, np.ones(50)/50, mode='same')

# Trouver les pics dans les données lissées
peaks, prop = find_peaks(y_smoothed, height=seuil, width=50)
print(peaks)
left = [int(n) for n in prop['left_ips']]
right = [int(n) for n in prop['right_ips']]
print(left, right)


for i, peak in enumerate(peaks):
    indice_pic_max = peak

    debut_roi = max(left[i]-100, 0)  # Début de la ROI, 50 au lieu de 100 pour le co60 et 2e du na22, 30 pour le co57
    fin_roi = min(right[i]+100, max(x))  # Fin de la ROI


    # Région d'intérêt
    roi_x = x[debut_roi:fin_roi]
    roi_y = y[debut_roi:fin_roi]
    marge_lr = y[debut_roi-10:fin_roi+10]
    print(sum(roi_y), sum(marge_lr)-sum(roi_y))


    # Ajustement d'une courbe gaussienne à la région d'intérêt
    def gaussienne(x, A, mu, sigma):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


    # Estimation des paramètres initiaux pour l'ajustement gaussien
    A_guess = np.max(roi_y)
    mu_guess = np.mean(roi_x)
    sigma_guess = np.std(roi_x)

    # Ajustement de la courbe gaussienne aux données
    parametres_optimaux, covariance = curve_fit(gaussienne, roi_x, roi_y, p0=[A_guess, mu_guess, sigma_guess])

    # Extraction des paramètres ajustés
    A_optimal, mu_optimal, sigma_optimal = parametres_optimaux
    mu_err = np.sqrt(np.diag(covariance))[1]
    sigma_err = np.sqrt(np.diag(covariance))[2]
    facteur = 2*np.sqrt(2*np.log(2))
    fwhm_err = sigma_err*facteur
    fwhm = sigma_optimal*facteur
    print('mu', round(mu_optimal,3), round(mu_err,3))
    print('FWHM', round(fwhm,3), round(fwhm_err,3))

    # Affichage des résultats
    plt.plot(x, y, label='Données')
    plt.plot(roi_x, roi_y, 'r', label='Région d\'intérêt')
    plt.plot(x, gaussienne(x, A_optimal, mu_optimal, sigma_optimal), 'g--', label='Courbe gaussienne ajustée')
    plt.xlabel('Énergie [channel]')
    plt.ylabel('Nombre de comptes')
    plt.legend()
    # plt.title(f'Spectre de rayonnement gamma de l\'élément {element}, dont une région d\'intérêt à été identifiée et ajustée à une courbe gaussienne')
    plt.show()