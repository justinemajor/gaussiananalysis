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
folder_path = 'Donnees1/spec_diffuseurs/txt/'
file_directories = data_analysis.list_files_in_folder(folder_path)

for file in file_directories:
    print(file)
    # Read raw data from a file
    raw_data = data_analysis.read_raw_data(file)
    # print(raw_data)

    # Get measuring time for a specific file
    measuring_time = data_analysis.get_measuring_time(file)
    print("temps d'acquisition : ", measuring_time)

    # Find ROI limits
    roi_limits = data_analysis.find_roi_limits(raw_data)
    # print("nombre de ROI : ", len(roi_limits[0]))

    x_ch = np.arange(len(raw_data))
    # Calibration des indices des données
    x = calibrer_axe_x(x_ch)

    for ind, lower in enumerate(roi_limits[0]):
        print("Indice de la ROI : ", ind)

        try :
            # Fit Gaussian to ROI
            gaussian_params = data_analysis.fit_gaussian_to_roi(raw_data, lower, roi_limits[1,ind])
            # print("    ", gaussian_params)
            # pour le na22
            cen = gaussian_params['mu']
            cen_err = gaussian_params['mu_error']
            cen = calibrer_axe_x(cen)
            # cen_err = a*cen*np.sqrt((a_err/a)**2+(cen_err/cen)**2)+b_err
            cen_err = a*cen*(a_err/a+cen_err/cen)+b_err
            print("    centroïde : ", cen, cen_err, ' keV')

            # Extraction des paramètres ajustés
            facteur = 2*np.sqrt(2*np.log(2))
            fwhm_err = gaussian_params['sigma_error']*facteur
            fwhm = gaussian_params['sigma']*facteur
            print('    FWHM : ', round(fwhm,3), round(fwhm_err,3))

            # Calculate total count within fitted Gaussian
            total_count = data_analysis.calculate_total_count(raw_data, gaussian_params)
            print("    nombre de comptes normalisé : ", total_count/measuring_time)

        except:
            0
