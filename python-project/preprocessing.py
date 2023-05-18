import pandas as pd
import pywt
from scipy import signal
from sklearn.decomposition import FastICA


def bandpass_filter(df, low_value=0.5, high_value=50, sampling_frequency=256, order=5):
    """
    Filters the values of a dataframe given a low_value and a high_value
    :param df: the raw datastructure
    :param low_value: the lowest bound
    :param high_value: the highest bound
    :return: the filtered dataframe
    """
    # Create a new DataFrame for containing the filtered values
    filtered = pd.DataFrame()

    # Do not apply the filter on the following columns
    to_skip = ["UNIX_TIMESTAMP"]

    for column in to_skip:
        filtered[column] = df[column]

    for column in df.columns:
        # Skip these columns
        if column in to_skip:
            continue

        # Create the butterworth filter
        num, den = signal.butter(N=order, Wn=(low_value, high_value), btype='bandpass', analog=False, fs=sampling_frequency)

        # Apply the filter
        filtered[column] = signal.filtfilt(num, den, df[column])

    return filtered

def ica_processing(df, n_components=None):
    """
    Applies the ICA processing method to the raw data structure in order to reduce the number of components
    :param df: the raw data structure
    :param n_components: the number of indipendent components to extract
    :return: the values extracted
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    ICA = FastICA(n_components=n_components, whiten='unit-variance')
    ica_values = ICA.fit_transform(df.values)

    return ica_values


def compute_spectral_features(df):
    """
    Computes the spectral features for the raw data structure provided
    :param df: the dataframe data structure
    :return: the spectral features and the names of the extracted features
    """
    f, psd = signal.periodogram(df.values)

    return psd


def compute_wavelet_transform(df):
    """
    Computes the wavelet transform for the raw data structure provided
    :param df: the data structure providing the measurements to compute the wavelet transform
    :return: the wavelet features extracted and their names
    """
    dwt = pywt.dwt(df.values, 'db1')

    return dwt
