import sklearn.model_selection



def train_test_split_random(users_measurements, users_names, perc_train=70, rand_state=42):
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


def train_test_split_sequence(users_measurements, perc_train=70):
    """
    Splits the set of data into data for training and data for testing based on the sequence of measurements.
    This can be used if we want to use an LSTM to achieve classification since it takes count of the sequence of values.
    :param users_measurements: the array-like object of Measurement sets containing user_id and sessions
    :param perc_train: the percentage of records used for training
    :return: an array-like object of Measurement objects for training and an array-like object for testing
    """

def train_test_split_session(users_measurements, n_session_train = 2):
    """
    Splits the set of data into data for training and data for testing based on the session id contained in the Measuremnet object.
    :param users_measurements: the array-like object of Measurement sets containing user_id and sessions
    :param n_session_train: the number of sessions used for training (the rest will be used for testing)
    :return: an array-like object of Measurement objects for training and an array-like object for testing
    """

    train_set, test_set = [], []
    for user_measurement in users_measurements:
        # The first n_measurements records will be used for training
        n_measurements = user_measurement.sessions[n_session_train]

        # Append the slice of measurements to the train_set
        # @todo train_set.append(user_measurement) ADD SLICE BUILT-IN FUNC TO MEASUREMENT CLASS
