import mne
import preprocessing as pr
import matplotlib.pyplot as plt

def show_graphic(raw, column_list = None, min_age=0, max_age=-1):
    """
    Shows the graphic of the input mne RAW DataStructure
    :param raw: the mne raw data structure containing the measurements
    :param column_list: the list of columns to be put in the graphics
    :param min_age: the starting time to show the measurements
    :param max_age: the last time to show the measurements
    :return: None
    """
    # If no column list is provided show the first 7 columns (in order to not create a messy graphic)

    if column_list is None:
        column_list = raw.ch_names[2:10]

    to_show = mne.pick_channels(raw.ch_names, column_list)

    # Select the sub-interval of measurements
    start, stop = raw.time_as_index([raw.times[min_age], raw.times[max_age]])
    data, times = raw[to_show, start:stop]

    # Add legend to the graphic

    fig, ax = plt.subplots()
    ax.plot(times, data.T)

    for idx, pick in enumerate(to_show):
        ax.plot(raw.times, raw._data[pick], label=column_list[idx])

    # Show the graphic

    ax.set(xlabel="Time (s)", ylabel = "Frequency (mV)", title="EEG Data")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    raw = pr.load(pr.PATH_TO_DS_1)
    #matplotlib.use('TkAgg')    # Uncomment if you want to set the display controller for matplotlib
    show_graphic(raw, column_list=["F3", "FC6"])
    raw = pr.bandpass_filter(raw, l_freq=0.5)
    show_graphic(raw, column_list=["F3", "FC6"])

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

