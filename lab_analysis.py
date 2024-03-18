import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt, peak_widths


class Data_analysis:
    def __init__(self):
        pass

    def list_files_in_folder(self, folder_path):
        """
        Parse all files in a folder and create a list of their directories.

        Args:
        folder_path (str): Path to the folder.

        Returns:
        list: List of file directories.
        """
        file_directories = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_directories.append(os.path.join(root, file))
        return file_directories

    import re

    def read_raw_data(self, file_path):
        """
        Read the raw data from a file.

        Args:
        file_path (str): Path to the file.

        Returns:
        numpy.ndarray: Array of numbers.
        """
        with open(file_path, 'r') as file:
            content = file.read()

            # Find the starting and ending indices
            start_match = re.search(r'\d+\s+\d+\n', content)
            if start_match:
                start_index = start_match.end()
            else:
                raise ValueError("Could not find start of data.")

            end_index = content.find('$ROI:\n')
            if end_index == -1:
                raise ValueError("Could not find end of data.")

            # Extract data between start and end indices
            data_content = content[start_index:end_index]

            # Find all integers in the extracted data
            data = re.findall(r'\d+', data_content)

            # Convert integers to integers
            data = [int(num) for num in data]

            return np.array(data[2:])


    def get_measuring_time(self, file_path):
        """
        Return the measuring time for a specific acquisition file.

        Args:
        file_path (str): Path to the file.

        Returns:
        float: Measuring time.
        """
        with open(file_path, 'r') as file:
            content = file.read()
            match = re.search(r'\$MEAS_TIM:\s*(\d+)', content)
            if match:
                return int(match.group(1))
            else:
                return None

    def find_roi_limits(self, spectrum):
        """
        Return indices corresponding to the lower and upper limits of an ROI representing a peak.

        Args:
        spectrum (numpy.ndarray): Raw spectrum data.
        threshold (int): Threshold for peak detection.

        Returns:
        tuple: Lower and upper limits of the ROI.
        """

        threshold = np.mean(spectrum)*1.8  # or 1.5 for mesures_init and etalonnage
        higher = np.where(spectrum > threshold)[0]

        #apply a Savitzky-Golay filter
        # smooth = savgol_filter(spectrum, window_length = 100, polyorder = 5)
        y_smoothed = np.convolve(spectrum, np.ones(50)/50, mode='same')
        # threshold = np.mean(y_smoothed)+20

        #find the maximums
        peaks_idx_max, _ = find_peaks(y_smoothed, height=threshold, width=30)
        # print('max', peaks_idx_max)
        widths = peak_widths(y_smoothed, peaks_idx_max, rel_height=0.7)
        widths = np.array(widths[0])
        # print('widths', widths)
        lower_limits = peaks_idx_max-widths/2
        lower_limits[0] = max(lower_limits[0], 0)
        upper_limits = peaks_idx_max+widths/2
        upper_limits[-1] = min(upper_limits[-1], len(spectrum))
        peaks_limits = np.vstack([lower_limits.astype(int), upper_limits.astype(int)])

        return peaks_limits

    def gaussian_function(self, x, A, mu, sigma):
        """
        Gaussian function.

        Args:
        x (numpy.ndarray): Input array.
        A (float): Amplitude.
        mu (float): Mean.
        sigma (float): Standard deviation.

        Returns:
        numpy.ndarray: Gaussian curve.
        """
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    def fit_gaussian_to_roi(self, spectrum, lower_limit, upper_limit):
        """
        Fit a Gaussian curve to an ROI in the spectrum.

        Args:
        spectrum (numpy.ndarray): Raw spectrum data.
        lower_limit (int): Lower limit of the ROI.
        upper_limit (int): Upper limit of the ROI.

        Returns:
        dict: Dictionary with optimal mu, sigma, and their errors.
        """
        x_data = np.arange(len(spectrum))
        y_data = spectrum

        roi_x = x_data[lower_limit:upper_limit+1]
        roi_y = y_data[lower_limit:upper_limit+1]

        # Guess initial parameters
        A_guess = np.max(roi_y)
        mu_guess = np.mean(roi_x)
        sigma_guess = np.std(roi_x)

        # Perform curve fitting
        popt, pcov = curve_fit(self.gaussian_function, roi_x, roi_y, p0=[A_guess, mu_guess, sigma_guess])

        # Calculate errors
        perr = np.sqrt(np.diag(pcov))

        return {'mu': popt[1], 'mu_error': perr[1], 'sigma': popt[2], 'sigma_error': perr[2], 'A':popt[0]}

    def plot_spectrum_with_roi(self, file, spectrum, roi_limits, gaussian_params=None):
        """
        Plot the raw spectrum with its ROI(s) and the fitted Gaussian curve.

        Args:
        file (list): List containing name of file and index of ROI studied.
        spectrum (numpy.ndarray): Raw spectrum data.
        roi_limits (list): List containing lower and upper limits of ROIs.
        gaussian_params (dict): Dictionary containing Gaussian parameters.
        """
        plt.plot(spectrum, label='Données expérimentales')

        roi_x = np.arange(len(spectrum))[roi_limits[0]:roi_limits[1]]
        roi_y = spectrum[roi_limits[0]:roi_limits[1]]

        plt.plot(roi_x, roi_y, 'r', label='Région d\'intérêt')
        # plt.axvline(x=roi_limits[0], color='r', linestyle='--')
        # plt.axvline(x=roi_limits[1], color='r', linestyle='--')

        if gaussian_params:
            x_data = np.arange(len(spectrum))
            gaussian_curve = self.gaussian_function(x_data, gaussian_params['A'], gaussian_params['mu'], gaussian_params['sigma'])
            plt.plot(gaussian_curve, 'g--', label='Courbe gaussienne ajustée')

        plt.xlabel('Énergie [channel]')
        plt.ylabel('Nombre de comptes')
        plt.legend()
        plt.savefig(f"{file[0]}_{file[1]}.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def calculate_total_count(self, spectrum, gaussian_params):
        """
        Calculate the total count number contained within reasonable limits of a fitted Gaussian.

        Args:
        spectrum (numpy.ndarray): Raw spectrum data.
        gaussian_params (dict): Dictionary containing Gaussian parameters.

        Returns:
        float: Total count number.
        """
        x_data = np.arange(len(spectrum))
        gaussian_curve = self.gaussian_function(x_data, gaussian_params['A'], gaussian_params['mu'], gaussian_params['sigma'])
        return int(np.sum(gaussian_curve))

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
