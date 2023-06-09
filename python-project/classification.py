# Import module math for general purpose
import math

# Import module tqdm for progress bar utility
from sklearn.ensemble import RandomForestClassifier

# Import module numpy for general purpose
from tqdm import tqdm

# Import module numpy for general purpose
import numpy as np

# Import module tensorflow for machine learning and deep learning algorithms
import tensorflow as tf

# Import module tensorflow decision forests for Random Forest algorithm
import tensorflow_decision_forests as tfdf

# Import module xgboost for better performing algorithms
import xgboost as xgb

# Import module scikit-learn for scaling data
from sklearn.preprocessing import MinMaxScaler

# Import module keras for LSTM algorithm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Import module scikit-learn for grid search
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def train_test_split(users_measurements, perc_train=70):
    """
    Splits the set of data into data for training and data for testing.
    :param users_measurements: the list of users measurements. Each i-th element of the list must be a list of measurements for the i-th user
    :param perc_train: the percentage of dataset used for training (1 - perc_train/100 will be used for testing)
    :return: x_train, y_train, x_test, y_test as follows: x is for data, y for labels.
    """
    # Create an instance of a scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # For each subject, scale data in order to have them between 0 and 1
    # Scale all the data to be values between 0 and 1
    for user_measurement in users_measurements:
        # Apply scaling
        new_values = scaler.fit_transform(user_measurement.values)
        # Change subject data with the scaled ones
        user_measurement.values = new_values

    # Create lists of tuples (m, i) where m is a numpy array and i is the id of the subject (source of measurements).
    train_data = []
    test_data = []

    # For each subject split into train and test
    for user_measurement in users_measurements:
        # Calculate the number of measurements to use for training
        training_dataset_length = math.ceil(len(user_measurement) * perc_train / 100)

        # Get the measurement values
        data_set = user_measurement.values

        # Shuffle the measurement values
        np.random.shuffle(data_set)

        # Get the training portion
        train_data_0 = data_set[0:training_dataset_length]
        train_data.append((train_data_0, user_measurement.subject_id))

        # Get the testing portion
        test_data_0 = data_set[training_dataset_length:]
        test_data.append((test_data_0, user_measurement.subject_id))

    # Create a list of measurements (x_train) and a list of labels (y_train) for training data
    x_train = train_data[0][0]
    y_train = []

    # Append one label for each measurement of the first session
    for _ in range(len(x_train)):
        y_train.append(train_data[0][1])

    # Appending the other two sessions
    for single_user in train_data[1:]:
        # Reshape the list in order to always have the same number of columns
        train_0 = (np.delete(single_user[0], np.s_[-1:], axis=1), single_user[1])
        # Append the measurements to the list of training data
        x_train = np.concatenate((x_train, train_0[0]))
        # Append one label for each measurement
        for _ in range(len(train_0[0])):
            y_train.append(single_user[1])

    # Convert to a numpy array
    y_train = np.array(y_train)

    # Create a list of measurements (x_test) and a list of labels (y_test) for test data
    x_test = test_data[0][0]
    y_test = []

    # Append one label for each measurement of the first session
    for _ in range(len(x_test)):
        y_test.append(test_data[0][1])
    # Append the other two sessions
    for single_user in test_data[1:]:
        # Reshape the list in order to always have the same number of columns
        test_0 = (np.delete(single_user[0], np.s_[-1:], axis=1), single_user[1])
        # Append the measurements to the list of testing data
        x_test = np.concatenate((x_test, test_0[0]))
        # Append the subject label for each measurement
        for _ in range(len(test_0[0])):
            y_test.append(single_user[1])

    # Convert to a numpy array
    y_test = np.array(y_test)

    # Shuffle data so that the algorithm doesn't train on the order of subject
    # Zip the arrays together to keep coherence
    train_zip = list(zip(x_train, y_train))
    test_zip = list(zip(x_test, y_test))

    # Shuffle zipped lists
    np.random.shuffle(train_zip)
    np.random.shuffle(test_zip)

    # Unzip lists
    x_train, y_train = zip(*train_zip)
    x_test, y_test = zip(*test_zip)

    # Convert back to numpy arrays
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    return x_train, y_train, x_test, y_test


def train_test_split_session(users_measurements, n_session_train = 2):
    """
    Splits the set of data into data for training and data for testing based on the session id contained in the Measuremnet object.
    :param users_measurements: the array-like object of Measurement sets containing user_id and sessions
    :param n_session_train: the number of sessions used for training (the rest will be used for testing)
    :return: an array-like object of Measurement objects for training and an array-like object for testing
    """
    # Create an instance of a scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # For each subject, scale data in order to have them between 0 and 1
    # Scale all the data to be values between 0 and 1
    for user_measurement in users_measurements:
        # Apply scaling
        new_values = scaler.fit_transform(user_measurement.values)
        # Change subject data with the scaled ones
        user_measurement.values = new_values

    # Create lists of tuples (m, i) where m is a numpy array and i is the id of the subject (source of measurements).
    train_data = []
    test_data = []

    # For each user extract the rows of each session
    for user_measurement in users_measurements:
        first_session_rows = user_measurement.sessions[1]
        second_session_rows = user_measurement.sessions[2]

        # Split data into three list of values containing each session
        first_session_measurements = user_measurement.values[:first_session_rows]
        second_session_measurements = user_measurement.values[first_session_rows:second_session_rows]
        third_session_measurements = user_measurement.values[second_session_rows:]

        user_session_measurements = [first_session_measurements, second_session_measurements, third_session_measurements]

        # Gather train data
        for i in range(n_session_train):
            train_data_0 = (user_session_measurements[i], user_measurement.subject_id)
            train_data.append(train_data_0)

        # Gather test data
        for i in range(n_session_train, len(user_session_measurements)):
            test_data_0 = (user_session_measurements[i], user_measurement.subject_id)
            test_data.append(test_data_0)

    # Create a list of measurements (x_train) and a list of labels (y_train) for training data
    x_train = train_data[0][0]
    y_train = []

    # Append one label for each measurement of the first session
    for _ in range(len(x_train)):
        y_train.append(train_data[0][1])

    # Appending the other two sessions
    for i, single_user in enumerate(train_data[1:]):
        # Reshape the list in order to always have the same number of columns
        if i > 0:
            train_0 = (np.delete(single_user[0], np.s_[-1:], axis=1), single_user[1])
        else:
            train_0 = single_user

        # Append the measurements to the list of training data
        x_train = np.concatenate((x_train, train_0[0]))
        # Append one label for each measurement
        for _ in range(len(train_0[0])):
            y_train.append(single_user[1])

    # Convert to a numpy array
    y_train = np.array(y_train)

    # Create a list of measurements (x_test) and a list of labels (y_test) for test data
    x_test = test_data[0][0]
    y_test = []

    # Append one label for each measurement of the first session
    for _ in range(len(x_test)):
        y_test.append(test_data[0][1])

    # Append the other two sessions
    for single_user in test_data[1:]:
        # Reshape the list in order to always have the same number of columns
        test_0 = (np.delete(single_user[0], np.s_[-1:], axis=1), single_user[1])
        # Append the measurements to the list of testing data
        x_test = np.concatenate((x_test, test_0[0]))
        # Append the subject label for each measurement
        for _ in range(len(test_0[0])):
            y_test.append(single_user[1])

    # Convert to a numpy array
    y_test = np.array(y_test)

    # Shuffle data so that the algorithm doesn't train on the order of subject
    # Zip the arrays together to keep coherence
    train_zip = list(zip(x_train, y_train))
    test_zip = list(zip(x_test, y_test))

    # Shuffle zipped lists
    np.random.shuffle(train_zip)
    np.random.shuffle(test_zip)

    # Unzip lists
    x_train, y_train = zip(*train_zip)
    x_test, y_test = zip(*test_zip)

    # Convert back to numpy arrays
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    return x_train, y_train, x_test, y_test


def classification_by_random_forest(x_train, y_train):
    """
    Trains a model using the random forest algorithm.
    :param x_train: the measurements for the training
    :param y_train: the labels for the training
    :return: the trained model
    """
    # Define the new model as a RandomForestModel
    model = tfdf.keras.RandomForestModel()

    # Fit the random forest model
    model.fit(x_train, y_train, verbose=1)

    return model

def prediction_by_random_forest(model, x_test):
    """
    Makes predictions given a model trained with a random forest algorithm
    :param model: the pre-trained model
    :param x_test: the measurements of the test portion of the dataset
    :return: the predictions
    """

    # Make predictions
    print("\n## INFO: starting predictions")
    predictions = model.predict(x_test, verbose=1)

    # Round values to the nearest integer
    predictions = np.round(predictions)

    # Create the list of outputs
    predictions_0 = []

    # Adjust values
    for val in predictions:
        val = np.argmax(val)
        predictions_0.append(val)
    
    predictions = predictions_0

    return predictions


def classification_by_gridsearch(x_train, y_train):
    """
    Trains a model using the random forest algorithm and the gridsearch tecnique to compute the best params.
    :param x_train: the measurements for the training
    :param y_train: the labels for the training
    :return: the trained model
    """
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    print("#### GridSearch output ####")
    print("Migliori parametri:", grid_search.best_params_)
    print("Miglior punteggio di validazione incrociata:", grid_search.best_score_)
    print("Miglior modello trovato:", grid_search.best_estimator_)

    return grid_search



def classification_by_xgboost(x_train, y_train):
    """
    Trains a model using the xgboost algorithm for better performances
    :param x_train: the measurements for the training
    :param y_train: the labels for the training
    :return: the trained model
    """
    # Compute hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "hist"}


    # Create a DMatrix based on x_train
    d_train = xgb.DMatrix(x_train, y_train, enable_categorical=True)

    # Create the XGBoost trained model
    model = xgb.train(params, d_train)

    return model

def prediction_by_xgboost(model, x_test, y_test):
    """
    Makes predictions given a model trained with a random forest algorithm
    :param model: the pre-trained model
    :param x_test: the measurements of the test portion of the dataset
    :return: the predictions
    """
    # Create a DMatrix based on x_train
    d_test = xgb.DMatrix(x_test, y_test, enable_categorical=True)

    # Make predictions
    predictions = model.predict(d_test)

    # Fix predicted values
    predictions = np.round(predictions)

    return predictions


def classification_by_lstm(x_train, y_train):
    """
    Trains a model using the LSTM algorithm.
    :param x_train: the measurements for the training
    :param y_train: the labels for the training
    :return: the trained model
    """
    # Initialising the RNN
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a third LSTM layer and Dropout layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and Dropout layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))


    # Adding the output layer
    # For full connection layer we use dense
    # Since the output is 1D - we use unit=1
    model.add(Dense(units=1))

    # compile and fit the model on 30 epochs
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Convert numpy arrays to tensors
    print("\n## INFO: converting to tensors")
    # Create a new list of tensors
    new_x = []
    # For each value in x_train, convert it into a Tensor for the model
    for i in tqdm(range(len(x_train))):
        tensor = tf.convert_to_tensor(x_train[i])
        new_x.append(tensor)

    x_train = new_x
    # Stack values for fulfilling LSTM standard input
    # Fitting tensorflow model
    print("## INFO: fitting model...")

    x_train = tf.stack(x_train)
    y_train = tf.stack(y_train)
    # Fit LSTM
    model.fit(x_train, y_train, epochs=30, batch_size=150, verbose=1)

    return model


def prediction_by_lstm(model, x_test):
    """
    Makes predictions given a model trained with a lstm algorithm
    :param model: the pre-trained model
    :param x_test: the measurements of the test portion of the dataset
    :return: the predictions
    """
    # Convert numpy arrays to tensors
    print("\n## INFO: converting to tensors")
    # Create a new list of tensors
    new_x = []
    for i in tqdm(range(len(x_test))):
        # For each value in x_test, convert it into a Tensor for the model
        tensor = tf.convert_to_tensor(x_test[i])
        new_x.append(tensor)

    x_test = new_x

    # Check predicted values
    print("## INFO: Starting prediction...")
    predictions = model.predict(x_test)

    # Undo scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    predictions = scaler.inverse_transform(predictions)

    return predictions

def compute_confusion_matrix(predictions, correct_labels):
    """
    Computes the confusion matrix given the predictions and the correct labels
    :param predictions: the predictions of a model
    :param correct_labels: the expected predictions
    :return: a dictionary containing keys as couples (p, c) i.e. predicted class and correct class, and values as the counters of each instance of (p, c)
    """

    # Create a dictionary where:
    # the key is a tuple (p, c) where p is the predicted value and c is the correct label;
    # the value is the number of times the model predicted p and the expected prediction was c.
    confusion_matrix = dict()
    # For each prediction
    for i in range(len(predictions)):
        # p is the predicted value
        p = predictions[i].item()
        # c is the expected value
        c = correct_labels[i].item()
        # (p, c) will be the key of the dictionary
        key = (p, c)

        if not key in confusion_matrix.keys():
            # If (p, c) still doesn't exist in the dictionary create a new entry
            confusion_matrix[key] = 0
            confusion_matrix[key] = 1
        else:
            # Otherwise increase the number of times the model predicted p and the correct label was c
            confusion_matrix[key] += 1

    return confusion_matrix

def compute_metrics(confusion_matrix):
    """
    Calculates accuracy, precision, recall and fscore for each class of the confusion matrix.
    :param confusion_matrix: the confusion matrix previously computed
    :return: a dictionary of key, values where keys are the classes and values are tuples like (accuracy, precision, recall, fscore)
    """

    # Identify all the classes
    classes = set()
    # For each entry of the confusion matrix find the classes of predictions
    for k in confusion_matrix.keys():
        # The second element of the key is the expected class
        _, c = k
        classes.add(c)

    # Create a dictionary in which keys are the classes and values are (Accuracy, Precision, Recall, F-Score)
    metrics = dict()
    # For each class compute the 4 standard metrics
    for cl in classes:
        # Initialize the number of true positives, true negatives, false positives and false negatives
        # For each class we are computing TP, TN, FP, FN
        tp = tn = fp = fn = 0
        # For each prediction, compare the predicted class and the expected class and increase the relative counter
        for k, v in confusion_matrix.items():
            p, c = k
            if p == c and p == cl:
                tp += v
            elif p == cl and c != cl:
                fp += v
            elif p != cl and c == cl:
                fn += v
            elif p != cl and c != cl:
                tn += v

        # Avoid division by zero
        if tp == tp == fp == fn == 0:
            acc = pr = recall = fscore = 0
        else:
            acc = ((tp + tn) / (tp + tn + fp + fn)) * 100
            pr = (tp / (tp + fp)) * 100
            recall = (tp / (tp + fn)) * 100
            fscore = (2 * recall * pr) / (recall + pr) / 100

        # Create an entry in the dictionary for the current class
        metrics[cl] = (acc, pr, recall, fscore)

    return metrics
