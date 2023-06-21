# Module loading defines the functions to read the dataset
import loading

# Module preprocessing defines the functions for preprocessing data and extracting features
import preprocessing

# Module classification defines the functions for attempting Human Recognition using ML/DL algorithms
import classification

if __name__ == '__main__':
    # Loading the measurements for different users
    users_measurements = loading.load(loading.PATH_TO_DS_1)

    # Apply a butterworth bandpass filter
    new_users_measurements = []
    for user_measurements in users_measurements:
        new_user_measurements = preprocessing.bandpass_filter(user_measurements)
        new_users_measurements.append(new_user_measurements)

    users_measurements = new_users_measurements

    # Apply principal component analysis to reduce the complexity of the signal

    pca_values = []
    for user_measurements in users_measurements:
        current_user_pca_values = preprocessing.pca_processing(user_measurements)
        pca_values.append(current_user_pca_values)

    users_measurements = pca_values

    # Split data in train and test via 70-30 method
    x_train, y_train, x_test, y_test = classification.train_test_split(users_measurements, perc_train=70)

    # Compute a model for machine learning
    model = classification.classification_by_random_forest(x_train, y_train, grid_search=False)
    # model = classification.classification_by_random_forest(x_train, y_train, grid_search=True)
    # model = classification.classification_by_xgboost(x_train, y_train)

    # Make predictions using the computed model
    predictions = classification.prediction_by_random_forest(x_test)
    # predictions = classification.prediction_by_xgboost(model, x_test, y_test)

    # Compute the confusion matrix based on the predictions
    confusion_matrix = classification.compute_confusion_matrix(predictions, y_test)

    # Compute accuracy, precision, recall and f1 score
    metrics = classification.compute_metrics(confusion_matrix)

    # Print output
    for k, v in metrics.items():
        print("=====\n\tUser ID: {0}\n=====".format(k))
        print("Accuracy: {0}%\nPrecision: {1}%\nRecall: {2}%\nF1-Score: {3}".format(*[metric for metric in v]))