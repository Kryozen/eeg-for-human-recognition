import loading
import preprocessing
import classification

if __name__ == '__main__':
    # Loading the measurements for different users
    users_measurements = loading.load(loading.PATH_TO_DS_1)
    # matplotlib.use('TkAgg')    # Uncomment if you want to set the display controller for matplotlib

    print("## INFO: starting preprocessing...")

    print("## INFO: applying butterworth bandpass filter...")

    # Apply a butterworth bandpass filter
    new_users_measurements = []
    for user_measurements in users_measurements:
        new_user_measurements = preprocessing.bandpass_filter(user_measurements)
        new_users_measurements.append(new_user_measurements)

    users_measurements = new_users_measurements

    print("## INFO: butterworth bandpass filter applied successfully!")

    # Apply independent component analysis to reduce the complexity of the signal

    # print("## INFO: applying Independent Component Analysis...")
    #
    # ica_values = []
    # for user_measurements in users_measurements:
    #     n_components = 15
    #     current_user_ica_values = preprocessing.ica_processing(user_measurements, n_components)
    #     ica_values.append(current_user_ica_values)
    #
    # # Reduce the number of measurements
    # users_measurements = ica_values

    # print("## INFO: Independent Component Analysis applied successfully!")

    # Apply principal component analysis to reduce the complexity of the signal

    print("## INFO: applying Principal Component Analysis...")

    pca_values = []
    for user_measurements in users_measurements:
        current_user_pca_values = preprocessing.pca_processing(user_measurements)
        pca_values.append(current_user_pca_values)

    # Reduce the number of measurements
    users_measurements = pca_values

    print("## INFO: Principal Component Analysis applied successfully!")

    # Compute Power Spectral Density

    # print("## INFO: computing Power Spectral Density...")
    #
    # psd_values = []
    # for user_measurements in users_measurements:
    #     current_user_psd = preprocessing.compute_spectral_features(user_measurements)
    #     psd_values.append(current_user_psd)
    #
    # print("## INFO: Power Spectral Density computed successfully!")
    # # @todo add psd mean etc
    #
    # # Compute Wavelet Transform
    #
    # print("## INFO: computing Wavelet transform...")
    #
    # wavelet_values = []
    # for user_measurements in users_measurements:
    #     current_user_wavelet = preprocessing.compute_wavelet_transform(user_measurements)
    #     wavelet_values.append(current_user_wavelet)
    #
    # print("## INFO: Wavelet transform computed successfully!")

    # Starting Classification
    print("## INFO: starting classification...")

    x_train, y_train, x_test, y_test = classification.train_test_split_random(users_measurements, perc_train=75)
    exit(0)
    model = classification.classification_by_lstm(x_train, y_train)

    classification.prediction_by_lstm(model, x_test, y_test)
