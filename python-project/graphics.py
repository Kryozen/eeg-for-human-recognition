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

    # Remove 50Hz frequency (electrical noise)
    raw = raw.notch_filter(50, picks='eeg')

    raw = raw.filter(l_freq= 0.5, h_freq= 120)

    # Remove eye blinking and muscle movement
    raw = pr.ica_processing(raw, excluded_list=[0])

    # Compute spectral features
    spectral_features, spectral_features_names = pr.compute_spectral_features(raw)

    # Compute wavelet features
    # wavelet_features, wavelet_features_names = pr.compute_wavelet_transform(raw)

    show_graphic(raw)
    plt.plot(spectral_features)
    plt.show()