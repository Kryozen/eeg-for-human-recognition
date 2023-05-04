import scipy
import mne
from mne.preprocessing import ICA
import pandas as pd

# Setting up

PATH_TO_DS_1 = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_2 = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"

def load(path):
    """
    Loads the matlib dataset into mne RAW data structures
    :param path: the path to the dataset
    :return: the mne RAW data structure
    """
    dataset = scipy.io.loadmat(path+"s1_s1.mat")

    # Loading 18 labels as given in the instructions

    labels = ['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP']
    recordings = dataset['recording']
    df = pd.DataFrame(recordings, columns=labels)

    # Creating mne RAW data structure
    mne_info = mne.create_info(ch_names=labels, sfreq=1000, ch_types="eeg")
    raw = mne.io.RawArray(df.values.T, mne_info)
    return raw

def bandpass_filter(raw, l_freq=0.5, h_freq=50):
    """
    Applies the band pass filter to the raw dataset given
    :param raw: the mne RAW data structure to filter
    :param l_freq: the low limit for the filter
    :param h_freq: the high limit for the filter
    :return: returns the filtered data structure
    """
    return raw.filter(l_freq=4, h_freq=50)

def ica_processing(raw, excluded_list=None):
    """
    Applies the ICA processing method to the raw data structure ignoring the component given as excluded_list
    :param raw: the mne RAW data structure
    :param excluded_list: the list of sensors to exclude
    :return: the raw data structure after applying the filter
    """
    ica = ICA(method='infomax', n_components=16, random_state=0)
    ica.fit(raw)
    ica.exclude = excluded_list
    ica.apply(raw)
    return raw
