import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from lab_analysis import Data_analysis


valeurs_calibrees = np.array([31, 81, 356, 122, 662, 511])

# calibration pour Donnees1
data_type = ".txt"
indices_connu = np.array([89.18195171624912, 225.22477063739933, 907.3673017230627, 332.80225678111793, 1630.5176612913885, 1276.4063097242545])

# calibration pour Donnees2 - fixe
# data_type = "fixe"
# indices_connu = np.array([296.0978926149062, 921.2635968006008, 3936.4071297508494, 1387.5355513300294, 7000.837661871095, 5507.799309382738])

# calibration pour Donnees2 - mobile
# data_type = "mobile"
# indices_connu = np.array([182.53096167711857, 481.73852791480573, 1903.007495986019, 708.9043529372805, 3367.1005554765425, 2659.326003318573])


indices_connu = np.sort(indices_connu)
valeurs_calibrees = np.sort(valeurs_calibrees)


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
folder_path = 'Donnees1/mesures_init/txt/'
file_directories = data_analysis.list_files_in_folder(folder_path)

for file in file_directories:
    print(file)

    if data_type in file:
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
                bruit = False
                upper = roi_limits[1,ind]

                if "mobile" in file:
                    bruit = True
                elif "fixe" in file:
                    bruit = False
                
                if "45_mobile" in file and ind == 0:
                    lower += 800
                    upper -= 900
                elif "_fixe" in file:
                    lower -= 200
                    upper += 300

                gaussian_params = data_analysis.fit_gaussian_to_roi(raw_data, lower, upper, bruit=bruit)
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
                total_count = data_analysis.calculate_total_count(raw_data, gaussian_params, upper)
                tc_err = (np.sqrt(total_count)/total_count+1/measuring_time)*total_count/measuring_time
                print("    nombre de comptes normalisé : ", total_count/measuring_time, tc_err)

            except Exception as e:
                raise(e)
