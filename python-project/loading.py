import scipy
import numpy
from measurement import Measurement


# Setting up
PATH_TO_DS_1 = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_1_PREP = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/Features/Verification/"
PATH_TO_DS_2 = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_2_PREP = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/Features/Verification/"

# Set the limit of users' data to populate the pandas dataframe. Choose values between 1 and 21
_users_limit = 2
# Set the limit of sessions to use for the pandas dataframe. Choose values between 1 and 3
_session_limit = 3
# Set True if you want to show logs
_logging = True


def load(path):
    """
    Loads the matlib dataset into a list of measurements.
    Each element of the list refers to a different user and contains a Measurement object
    :param path: the path to the dataset
    :return: a list of _user_limit Measurement objects
    """
    users_measurements = []

    for i in range(1, _users_limit + 1):
        current_user_measurement = []

        # A dictionary containing the indexes of record and the relative sessions
        # For example: 1:100, 2:500, 3:580 means that the first 100 records are from the first session and so on

        sessions = dict()

        for j in range(1, _session_limit + 1):
            # Reading .mat file
            dataset_path = "{0}s{1}_s{2}.mat".format(path, i, j)

            # Loading .mat file
            dataset = scipy.io.loadmat(dataset_path)

            # Extracting recordings
            table = dataset['recording']
            # Transposing
            table = table.T
            # Dropping "Counter" and "Interpolated"
            table = table[2:]
            # Gathering recordings
            if j == 1:
                current_user_measurement = table.tolist()
            else:
                for sensor_number in range(0, len(current_user_measurement)):
                    current_user_measurement[sensor_number].extend(table[sensor_number])

            # Saving session information
            sessions[j] = len(current_user_measurement[0])

        if _logging:
            print("## INFO: loading {0} measurements for subject {1}".format(len(current_user_measurement[0]), i))

        measurement = Measurement(current_user_measurement, subject_id=i, sessions=sessions)
        users_measurements.append(measurement)


    # Logging
    if _logging:
        print("## INFO: loading {} subjects".format(len(users_measurements)))

    return users_measurements
