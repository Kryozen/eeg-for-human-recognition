import scipy
import mne
import pandas as pd
import numpy as np
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_multitaper, morlet

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

def ica_processing(raw, excluded_list=None):
    """
    Applies the ICA processing method to the raw data structure ignoring the component given as excluded_list
    :param raw: the mne RAW data structure
    :param excluded_list: the list of sensors to exclude
    :return: the raw data structure after applying the filter
    """
    ica = ICA(n_components=14, random_state=0)
    ica.fit(raw)
    ica.exclude = excluded_list
    ica.apply(raw)
    return raw

# W.I.P.
def compute_arcc(raw, segment_len=3.0):
    """
    Compute Auto Regressive C Coefficients
    :param raw:
    :return: the calculated coefficients
    """
    # ar = AutoReject()
    # segment_len = 3.0
    # n_segments = int(np.floor(raw.times[-1] / segment_len))     #calcola il numero di segmenti
    # segments = np.array_split(raw.get_data(), n_segments, axis=1)   #suddivisione del segnale
    #
    # # Crea una lista di oggetti RawArray
    # raw_arrays = []
    # for i, segment_data in enumerate(segments):
    #     info = mne.create_info(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    #     raw_arrays.append(mne.io.RawArray(segment_data, info))
    #
    # raw_arrays = raw_arrays.real()
    # raw = mne.io.RawArray(raw_arrays, info)
    # # Crea gli oggetti Epochs
    # epochs = mne.Epochs(raw_arrays, events=None, tmin=0, tmax=segment_len, baseline=None)
    # #epochs = mne.make_fixed_length_epochs(raw, duration=3.0)    #conversione da raw a epoch, per dare in input a trasform
    # ar.fit(raw)
    # clean_raw, _ = ar.transform(raw, return_log=True)
    # raw.plot()
    # clean_raw.plot()
    # print(_)
    # return clean_raw

def compute_spectral_features(raw):
    """
    Computes the spectral features for the raw object provided
    :param raw: the raw data structure
    :return: the spectral features and the names of the extracted features
    """

    # Compute PSD using multitaper method from mne
    psd, freqs = psd_array_multitaper(raw.get_data(), sfreq=raw.info['sfreq'])

    # Compute spectral features
    spectral_centroid = np.sum(psd * freqs[:, np.newaxis].T, axis=0) / np.sum(psd, axis=0)
    spectral_entropy = -np.sum(psd * np.log2(psd), axis=0)
    spectral_edge_frequency = np.sum(np.cumsum(psd, axis=0) <= 0.5, axis=0) / psd.shape[0] * raw.info['sfreq'] / 2.
    spectral_flatness = np.exp(np.mean(np.log(psd), axis=0)) / np.mean(psd, axis=0)

    # Concatenate results
    spectral_features = np.concatenate((spectral_centroid, spectral_entropy, spectral_edge_frequency, spectral_flatness))
    spectral_features_names = np.array(("Spectral centroid", "Spectral entropy", "Spectral edge frequency", "Spectral flatness"))

    return spectral_features, spectral_features_names

def compute_wavelet_transform(raw):
    """
    Computes the wavelet transform for the raw structure provided
    :param raw: the mne data structure providing the measurements to compute the wavelet transform
    :return: the wavelet features extracted and their names
    """