import loading
import preprocessing as pr
import classification as cl
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
    users_measurements = loading.load(loading.PATH_TO_DS_1)
    # matplotlib.use('TkAgg')    # Uncomment if you want to set the display controller for matplotlib

    print("## INFO: starting preprocessing...")

    # Apply a butterworth bandpass filter
    new_users_measurements = []
    for user_measurements in users_measurements:
        new_user_measurements = pr.bandpass_filter(user_measurements)
        new_users_measurements.append(new_user_measurements)

    users_measurements = new_users_measurements

    # Apply indipendent component analysis to reduce the complexity of the signal
    # ica_values = []
    # for user_measurement in users_measurements:
    #     n_components = 16
    #     current_user_ica_values = pr.ica_processing(user_measurement, n_components)
    #     ica_values.append(current_user_ica_values)

    print("## INFO: starting classification...")

    users_names = [i for i in range(loading._users_limit)]
    gathered_df = loading.gather(users_measurements, users_names)
    meas_train, meas_test, names_train, names_test = cl.train_test_split(gathered_df.drop(["ID"], axis=1), gathered_df["ID"])
    print("## INFO: starting training...")
    svm = cl.train(meas_train, names_train)
    print("## INFO: starting testing...")
    predictions, acc = cl.test(svm, meas_test, names_test)

    print("Accuracy: {}%".format(acc))
