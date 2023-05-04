import mne
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

# Setting up
PATH_TO_DS = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
matplotlib.use('TkAgg')

# Load matlab files into pandas DataFrame
dataset = scipy.io.loadmat(PATH_TO_DS+"s1_s1.mat")
labels = ['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP']
recordings = dataset['recording']
df = pd.DataFrame(recordings, columns=labels)

# Creating mne RAW data structure
mne_info = mne.create_info(ch_names=labels, sfreq=1000, ch_types="eeg")
raw = mne.io.RawArray(df.values.T, mne_info)

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