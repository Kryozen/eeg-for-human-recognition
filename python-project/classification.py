import sklearn.model_selection


def train_test_split(users_measurements, users_names, perc_train=70, rand_state=42):
    """

    :param users_measurements:
    :param users_names:
    :param perc_train:
    :param rand_state:
    :return:
    """

    # Uses the sklearn method to split the dataset with a random factor
    meas_train, meas_test, names_train, names_test = sklearn.model_selection.train_test_split(
        users_measurements, users_names, test_size= perc_train/100, random_state=rand_state
    )

    return meas_train, meas_test, names_train, names_test

def train(meas_train, names_train):
    """
    Trains a support vector machine.
    :param meas_train: The measurements on which the svm will be trained
    :param names_train: the names of the subjects associated to the measurements
    :return: the trained svm
    """

    # Creates a support vector machine
    svm = sklearn.svm.SVC()
    # Trains the support vector machine
    svm.fit(meas_train, names_train)

    return svm


def test(svm, meas_test, names_test):
    """
    Tests an already trained svm
    :param svm: a trained svm
    :param meas_test: the subset of measurements to test the svm with
    :param names_test: the names that should've been predicted
    :return: the predicted values and the accuracy of the prediction
    """

    # Predicts the names of the subjects for which are given the measurements
    names_predicted = svm.predict(meas_test)
    # Calculates the accuracy of the predictions
    accuracy = sklearn.metrics.accuracy_score(names_test, names_predicted)

    return names_predicted, accuracy
