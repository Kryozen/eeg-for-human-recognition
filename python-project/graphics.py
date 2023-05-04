import mne
import numpy as np
from mne.preprocessing import ICA
from autoreject import AutoReject
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

# Setting up
PATH_TO_DS = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
#matplotlib.use('TkAgg')

# Load matlab files into pandas DataFrame
dataset = scipy.io.loadmat(PATH_TO_DS+"s1_s1.mat")
labels = ['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP']
recordings = dataset['recording']
df = pd.DataFrame(recordings, columns=labels)

# Creating mne RAW data structure
mne_info = mne.create_info(ch_names=labels, sfreq=1000, ch_types="eeg")
raw = mne.io.RawArray(df.values.T, mne_info)

#applico filtro passa-banda
raw.filter(l_freq=4, h_freq=50)


#ICA
ica = ICA(method='infomax', n_components=16, random_state=0)
ica.fit(raw)
ica.exclude = [0, 1, 16]
ica.apply(raw)

#ARCC
"""
ar = AutoReject()
segment_len = 3.0
n_segments = int(np.floor(raw.times[-1] / segment_len))     #calcola il numero di segmenti
segments = np.array_split(raw.get_data(), n_segments, axis=1)   #suddivisione del segnale
# Crea una lista di oggetti RawArray
raw_arrays = []
for i, segment_data in enumerate(segments):
    info = mne.create_info(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
    raw_arrays.append(mne.io.RawArray(segment_data, info))

raw_arrays = raw_arrays.real()
raw = mne.io.RawArray(raw_arrays, info)
# Crea gli oggetti Epochs
epochs = mne.Epochs(raw_arrays, events=None, tmin=0, tmax=segment_len, baseline=None)
#epochs = mne.make_fixed_length_epochs(raw, duration=3.0)    #conversione da raw a epoch, per dare in input a trasform
ar.fit(raw)
clean_raw, _ = ar.transform(raw, return_log=True)
raw.plot()
clean_raw.plot()
print(_)
"""

# Show graphics
labels_to_show = raw.ch_names[2:10]
to_show = mne.pick_channels(raw.ch_names, labels_to_show)
start, stop = raw.time_as_index([raw.times[0], raw.times[-1]]) # Select 100 seconds of signals
data, times = raw[to_show, start:stop]

fig, ax = plt.subplots()
ax.plot(times, data.T)
for idx, pick in enumerate(to_show):
    ax.plot(raw.times, raw._data[pick], label=labels_to_show[idx])
ax.set(xlabel="Time (s)", ylabel = "Frequency (mV)", title="EEG Data (F3)")
ax.legend()
plt.show()
