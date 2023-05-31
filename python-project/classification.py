import math
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def train_test_split_random(users_measurements, perc_train=70):
    """

    :param users_measurements:
    :param users_names:
    :param perc_train:
    :param rand_state:
    :return:
    """

    # # Gather sessions
    # new_users_measurements = []
    #
    # for i, user_measurement in enumerate(users_measurements):
    #     meas_0 = user_measurement.values[0]
    #
    #     new_users_measurements[i] = Measurement(meas_0, user_measurement.subject_id, user_measurement.sessions)
    #     print("## INFO: merged user {0} measurements...".format(i))
    #
    # users_measurements = new_users_measurements

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale all the data to be values between 0 and 1
    for user_measurement in users_measurements:
        new_values = scaler.fit_transform(user_measurement.values)
        user_measurement.values = new_values

    # Train data and test data are lists of tuples.
    # The first element of the tuple is the numpy array containing the measurements
    # The second element of the tuple is the id of the subject
    train_data = []
    test_data = []
    for user_measurement in users_measurements:
        training_dataset_length = math.ceil(len(user_measurement) * perc_train / 100)
        train_data.append((user_measurement.values[0:training_dataset_length], user_measurement.subject_id))
        test_data.append((user_measurement.values[training_dataset_length:], user_measurement.subject_id))

    # Splitting the data
    x_train = train_data[0][0][0:13]
    y_train = [train_data[0][1],]
    for single_user in train_data[1:]:
        single_user = (np.delete(single_user[0], np.s_[-1:], axis=1), single_user[1])
        np.concatenate((x_train, single_user[0]))
        y_train.append(single_user[1])

    # Convert to numpy arrays
    y_train = np.array(y_train)

    # Reshape the data into 3-D array
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # splitting the x_test and y_test data sets
    x_test = []
    y_test = []
    for single_user in test_data:
        x_test.extend(single_user[0])
        y_test.append(single_user[1])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into 3-D array
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, x_test, y_train, y_test


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


def classification_by_lstm(x_train, y_train):
    # Initialising the RNN
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    print("## INFO: model created ({0}%)...".format(1/4 * 100))

    # Adding a second LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    print("## INFO: added second layer ({0}%)...".format(2/4 * 100))

    # Adding a third LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    print("## INFO: added third layer ({0}%)...".format(3/4 * 100))

    # Adding a fourth LSTM layer and Dropout layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    print("## INFO: added fourth layer ({0}%)...".format(4 / 4 * 100))


    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D - we use unit=1
    model.add(Dense(units=1))

    print("## INFO: added output layer")

    # compile and fit the model on 30 epochs
    model.compile(optimizer='adam', loss='mean_squared_error')

    # !!!ERRORE QUI!!!
    new_x = []
    for i in range(len(x_train)):
        new_x.append(tf.convert_to_tensor(x_train[i]))
    x_train = new_x

    model.fit(x_train, y_train, epochs=30, batch_size=50)

    return model


def prediction_by_lstm(model, x_test, y_test):
    # Check predicted values
    predictions = model.predict(x_test)

    # Undo scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = scaler.inverse_transform(predictions)

    # Calculate RMSE score
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("Accuracy: {}%".format(rmse))
