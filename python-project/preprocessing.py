import pywt
import numpy
from measurement import Measurement
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA


def bandpass_filter(measurement, low_value=0.5, high_value=50, sampling_frequency=256, order=5):
    """
    Filters the values of a dataframe given a low_value and a high_value
    :param measurement: the Measurement object containing each measurement
    :param low_value: the lowest bound
    :param high_value: the highest bound
    :return: the filtered dataframe
    """
    # Get values
    values = measurement.values
    # Create a new DataFrame for containing the filtered values
    filtered = []

    # Do not apply the filter on the following columns
    # 16 is UNIX_TIMESTAMP
    to_skip = [16]

    for i in range(len(values)):
        # Skip these columns
        if i in to_skip:
            continue

        # Create the butterworth filter
        num, den = signal.butter(N=order, Wn=(low_value, high_value), btype='bandpass', analog=False, fs=sampling_frequency)

        # Apply the filter
        filtered.append(signal.filtfilt(num, den, values[i]).tolist())

    return Measurement(filtered, measurement.subject_id)

def ica_processing(measurement, n_components=None):
    """
    Applies the ICA processing method to the measurements in order to reduce the number of components
    :param measurement: the Measurement object containing each measurement
    :param n_components: the number of independent components to extract
    :return: the values extracted
    """
    ica = FastICA(n_components=n_components, whiten='unit-variance')
    ica_values = ica.fit_transform(numpy.transpose(measurement.values))

    return Measurement(ica_values, measurement.subject_id)

def pca_processing(measurement):
    """
    Applies the PCA processing method to the measurements in order to reduce the number of components
    :param measurement: the Measurement object containing each measurement
    :param n_components: the number of principal components to extract
    :return: the values extracted
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components='mle', copy=True)
    pca_values = pca.fit_transform(numpy.transpose(measurement.values))

    return Measurement(pca_values, measurement.subject_id)


def compute_spectral_features(measurement):
    """
    Computes the spectral features for the raw data structure provided
    :param measurement: the Measurement object containing each measurement
    :return: the spectral features and the names of the extracted features
    """
    f, psd = signal.periodogram(measurement.values)

    return psd


def compute_wavelet_transform(measurement):
    """
    Computes the wavelet transform for the raw data structure provided
    :param measurement: the Measurement object containing each measurement
    :return: the wavelet features extracted and their names
    """
    dwt = pywt.dwt(measurement.values, 'db1')

    return dwt
