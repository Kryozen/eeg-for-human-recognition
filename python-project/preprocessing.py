# Import module PyWavelets for Wavelet Transform
import pywt

# Import module NumPy for general purpose
import numpy

# Import module tqdm for progress bar utility
from tqdm import tqdm

# Import class Measurement from module measurement
from measurement import Measurement

# Import signal from module scipy for Power Spectral Density
from scipy import signal

# Import module scikit-learn for Independent Component Analysis
from sklearn.decomposition import FastICA

# Import module scikit-learn for Principal Component Analysis
from sklearn.decomposition import PCA


def bandpass_filter(measurement, low_value=0.5, high_value=50, sampling_frequency=256, order=5):
    """
    Filters the values of a list of values given a low_value and a high_value
    :param measurement: the Measurement object containing each measurement of the same subject
    :param low_value: the lowest bound
    :param high_value: the highest bound
    :param sampling_frequency: the frequency of samples per second acquired. Default 256
    :param order: the order of the filter. Default 5
    :return: a Measurement object containing the filtered measurements
    """

    # Get values from the Measurement object
    values = measurement.values

    # Create a new list that will contain the filtered values
    filtered = []

    # Do not apply the filter on the following columns: UNIX_TIMESTAMP (16)
    to_skip = [16]

    # Apply the filter to the values of each sensor
    for i in tqdm(range(len(values))):

        # Skip defined sensors
        if i in to_skip:
            continue

        # Create the filter
        num, den = signal.butter(N=order, Wn=(low_value, high_value), btype='bandpass', analog=False, fs=sampling_frequency)

        # Apply the filter
        filtered.append(signal.filtfilt(num, den, values[i]).tolist())

    return Measurement(filtered, measurement.subject_id, measurement.sessions)

def ica_processing(measurement, n_components=None):
    """
    Applies the ICA processing method to the measurements in order to reduce the number of components
    :param measurement: the Measurement object containing each measurement
    :param n_components: the number of independent components to extract
    :return: the values extracted
    """

    # Create an instance of FastICA from scikit-learn to apply the Independent Component Analysis
    ica = FastICA(n_components=n_components, whiten='unit-variance')

    # Apply the ICA to the Measurement object values
    ica_values = ica.fit_transform(numpy.transpose(measurement.values))

    return Measurement(ica_values, measurement.subject_id, measurement.sessions)


def pca_processing(measurement):
    """
    Applies the PCA processing method to the measurements in order to reduce the number of components
    :param measurement: the Measurement object containing each measurement
    :return: the values extracted
    """
    # Create an instance of FastICA from scikit-learn to apply the Principal Component Analysis
    pca = PCA(n_components='mle', copy=True)

    # Apply the ICA to the Measurement object values
    pca_values = pca.fit_transform(numpy.transpose(measurement.values))

    return Measurement(pca_values, measurement.subject_id, measurement.sessions)


def compute_spectral_features(measurement):
    """
    Computes the spectral features for the raw data structure provided
    :param measurement: the Measurement object containing each measurement
    :return: the spectral features and the names of the extracted features
    """

    # Compute the Power Spectral Density using the scipy.signal module
    f, psd = signal.periodogram(measurement.values)

    return psd


def compute_wavelet_transform(measurement):
    """
    Computes the wavelet transform for the raw data structure provided
    :param measurement: the Measurement object containing each measurement
    :return: the wavelet features extracted and their names
    """

    # Compute the Wavelet Transform using the PyWavelets module
    dwt = pywt.dwt(measurement.values, 'db1')

    return dwt
