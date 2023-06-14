import loading
import preprocessing
import classification

if __name__ == '__main__':
    # Loading the measurements for different users
    users_measurements = loading.load(loading.PATH_TO_DS_1)

    print("## INFO: starting preprocessing...")

    print("## INFO: applying butterworth bandpass filter...")

    # Apply a butterworth bandpass filter
    new_users_measurements = []
    for user_measurements in users_measurements:
        new_user_measurements = preprocessing.bandpass_filter(user_measurements)
        new_users_measurements.append(new_user_measurements)

    users_measurements = new_users_measurements

    print("## INFO: butterworth bandpass filter applied successfully!")

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

    # Split data in train and test
    x_train, y_train, x_test, y_test = classification.train_test_split(users_measurements, perc_train=70)

    # Compute a model for machine learning
    model = classification.classification_by_random_forest(x_train, y_train)

    # Make predictions using the computed model
    predictions = classification.prediction_by_random_forest(model, x_test)

    # Compute the confusion matrix based on the predictions
    confusion_matrix = classification.compute_confusion_matrix(predictions, y_test)

    # Compute the metrics
    metrics = classification.compute_metrics(confusion_matrix)

    # Print output
    for k, v in metrics.items():
        print(" =====\n Class {0}\n=====".format(k))
        print("Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nF-Score: {3}".format(*[metric for metric in v]))