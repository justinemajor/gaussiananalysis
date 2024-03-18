import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from lab_analysis import Data_analysis

# Supposons que vous ayez des valeurs connues et leurs indices correspondants
# indices_connu = np.array([311.151, 1509.071, 1886.314, 2612.035, 2961.651])
indices_connu = np.array([89.18195171624912, 225.22477063739933, 907.3673017230627, 332.80225678111793, 1630.5176612913885, 1276.4063097242545])
valeurs_calibrees = np.array([31, 81, 356, 122, 662, 511])

# Fonction pour ajuster les données connues
def fonction_calibration(x, a, b):
    return a * x + b

# Ajustement de la fonction aux données connues
parametres_optimaux, covariance = curve_fit(fonction_calibration, indices_connu, valeurs_calibrees)

# Extraction des paramètres ajustés
a, b = parametres_optimaux
a_err = np.sqrt(np.diag(covariance))[0]
b_err = np.sqrt(np.diag(covariance))[1]

# Définition de la fonction de calibration
def calibrer_axe_x(indice):
    return a * indice + b

plt.plot(indices_connu, valeurs_calibrees, label='Résultats expérimentaux de centroïde des photopics observés')
plt.plot(indices_connu, calibrer_axe_x(indices_connu), label='Étalonnage en énergie estimée par une équation linéaire')
plt.xlabel('Canaux d\'énergie connus')
plt.ylabel("Énergie équivalente [keV]")
plt.legend()
plt.show()


# application
data_analysis = Data_analysis()
folder_path = 'Donnees1/spec_ang/txt/'
file_directories = data_analysis.list_files_in_folder(folder_path)
print(file_directories)

for file in file_directories:
    print(file)
    # Read raw data from a file
    raw_data = data_analysis.read_raw_data(file)
    # print(raw_data)

    # Get measuring time for a specific file
    measuring_time = data_analysis.get_measuring_time(file_directories[0])
    print("temps d'acquisition : ", measuring_time)

    # Find ROI limits
    roi_limits = data_analysis.find_roi_limits(raw_data)
    print("nombre de ROI : ", len(roi_limits[0]))

    for ind, lower in enumerate(roi_limits[0]):
        print("Indice de la ROI : ", ind)

        try :
            # Fit Gaussian to ROI
            gaussian_params = data_analysis.fit_gaussian_to_roi(raw_data, lower, roi_limits[1,ind])
            print("    ", gaussian_params)

            # Plot spectrum with ROI(s) and fitted Gaussian curve
            file = file.replace('.txt', '').replace('txt', 'fig')
            data_analysis.plot_spectrum_with_roi([file, ind], raw_data, [lower,roi_limits[1,ind]], gaussian_params)

            # Calculate total count within fitted Gaussian
            total_count = data_analysis.calculate_total_count(raw_data, gaussian_params)
            print("    nombre de comptes total : ", total_count)

        except:
            0







x_ch = np.arange(len(y))
# Calibration des indices des données
x = calibrer_axe_x(x_ch)

print('pente', a, a_err)
print('absisse', b, b_err)

# pour le na22
cen = 2839.301  # 1184.363
cen_err = 0.687  # 0.221
cen = calibrer_axe_x(cen)
# cen_err = a*cen*np.sqrt((a_err/a)**2+(cen_err/cen)**2)+b_err
cen_err = a*cen*(a_err/a+cen_err/cen)+b_err
print("centroïde Na22", cen, cen_err)

# Seuil
seuil = 80  # 50 pour mn (très faible)

# Trouver les indices où y dépasse le seuil
indices_depassement = np.where(y > seuil)[0]

# Lissage des données pour trouver les bords de la ROI de manière plus précise
y_smoothed = np.convolve(y, np.ones(50)/50, mode='same')

# Trouver les pics dans les données lissées
peaks, prop = find_peaks(y_smoothed, height=seuil, width=50)
print(peaks)
left = [int(n) for n in prop['left_ips']]
right = [int(n) for n in prop['right_ips']]


for i, peak in enumerate(peaks):
    indice_pic_max = peak

    debut_roi = left[i]-50  # Début de la ROI
    fin_roi = right[i]+50  # Fin de la ROI

    # Région d'intérêt
    roi_x = x[debut_roi:fin_roi]
    roi_y = y[debut_roi:fin_roi]


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
    plt.axvline(511, color='grey', linestyle=':', label=f'photopics attendus à 511 keV et 1275 keV')
    plt.axvline(1275, color='grey', linestyle=':')
    plt.xlabel('Énergie [keV]')
    plt.ylabel('Nombre de comptes')
    plt.legend()
    # plt.title(f'Ajustement de courbe gaussienne à une région d\'intérêt du {element}')
    plt.show()