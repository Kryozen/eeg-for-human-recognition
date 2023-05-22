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
