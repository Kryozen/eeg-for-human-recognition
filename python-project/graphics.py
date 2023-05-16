import preprocessing as pr
import matplotlib.pyplot as plt


def show_graphic(df_struct):
    """
    Shows the graphic of the input raw data structure
    :param df_struct: the pandas dataframe
    :return: None
    """
    # Close all plots if there are any
    plt.close("all")

    # Show the graphic
    df_struct.plot(x="UNIX_TIMESTAMP")
    plt.show()


if __name__ == '__main__':
    df = pr.load(pr.PATH_TO_DS_1)
    # matplotlib.use('TkAgg')    # Uncomment if you want to set the display controller for matplotlib

    # Apply a butterworth bandpass filter
    df = pr.bandpass_filter(df)

    # Apply indipendent component analysis to reduce the complexity of the signal
    # ica_values = pr.ica_processing(df, 16)

    # Compute the AutoRegressive Reflection Coefficients
    arrc_values = pr.compute_arrc(df)
