import scipy
import pandas as pd
import statsmodels.api as sm
from scipy import signal
from sklearn.decomposition import FastICA


# Setting up
PATH_TO_DS_1 = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_2 = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
# Set the limit of users' data to populate the pandas dataframe. Choose values between 1 and 21
_users_limit = 1
# Set the limit of sessions to use for the pandas dataframe. Choose values between 1 and 3
_session_limit = 1
# Set True if you want to show logs
_logging = True

def load(path, samp_freq=256):
    """
    Loads the matlib dataset into a pandas data stracture
    :param path: the path to the dataset
    :return: the raw data structure
    """
    # Loading 18 labels as given in the instructions

    labels = [
        'COUNTER',         # this field should be ignored
        'INTERPOLATED',    # this field should be ignored
        'F3',
        'FC5',
        'AF3',
        'F7',
        'T7',
        'P7',
        'O1',
        'O2',
        'P8',
        'T8',
        'F8',
        'AF4',
        'FC6',
        'F4',
        'UNIX_TIMESTAMP'
    ]

    recordings = []

    for i in range(1, _users_limit + 1):
        for j in range(1, _session_limit + 1):
            # Reading .mat file
            dataset_path = "{0}s{1}_s{2}.mat".format(path, i, j)

            # Loading .mat file
            dataset = scipy.io.loadmat(dataset_path)

            # Extracting recordings
            table = dataset['recording']

            # Gathering recordings
            recordings.extend(table)

    # Logging
    if _logging:
        print("INFO: loading {} measurements in pandas dataframe.".format(len(recordings)))

    # Create the dataframe
    df = pd.DataFrame(recordings, columns=labels)

    return df

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
    ICA = FastICA(n_components=n_components, whiten='unit-variance')
    ica_values = ICA.fit_transform(df.values)

    return ica_values

def compute_arrc(df, segment_len=3.0):
    """
    Compute Auto Regressive Coefficients
    :param df: the raw data structure
    :return: the calculated coefficients
    """

    # Convert DataFrame to a numpy array
    data = df.values.T

    """ Qui parte la prova """
    # For each sensor
    for sensor in data:
        # Estimate the optimal AR order using information criteria
        order = sm.tsa.arma_order_select_ic(sensor, ic='aic')['aic_min_order']

        print(order)

        # Fit the Vector of AutoRegressive coefficents
        var_model = sm.tsa.VAR(sensor)

        # Estimate VAR coefficients using the optimal order (Akaike Information Criterion)
        var_result = var_model.fit(maxlags=order, ic='aic')
        var_coeffs = var_result.coefs

    # Extract the AR reflection coefficients from the VAR coefficients
    arrc_coeffs = var_coeffs[:, 1:, :]  # Exclude the intercept coefficient

    print(arrc_coeffs)
    """ Qui finisce la prova """
    """
    # Estimate the optimal AR order using information criteria
    order = sm.tsa.arma_order_select_ic(data, ic='aic')['aic_min_order']
    
    # Fit a VAR model to the data
    var_model = sm.tsa.VAR(data)
    
    # Estimate VAR coefficients using the optimal order (Akaike Information Criterion)
    var_result = var_model.fit(maxlags=order, ic='aic')
    var_coeffs = var_result.coefs

    # Extract the AR reflection coefficients from the VAR coefficients
    arrc_coeffs = var_coeffs[:, 1:, :]  # Exclude the intercept coefficient
    """

    return arrc_coeffs

def compute_spectral_features(df):
    """
    Computes the spectral features for the raw data structure provided
    :param df: the dataframe data structure
    :return: the spectral features and the names of the extracted features
    """


def compute_wavelet_transform(raw):
    """
    Computes the wavelet transform for the raw data structure provided
    :param raw: the mne data structure providing the measurements to compute the wavelet transform
    :return: the wavelet features extracted and their names
    """

