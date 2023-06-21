# Import module scipy for loading matlab files
import scipy

# Import class Measurement from module measurement
from measurement import Measurement

# Useful variables for dataset path
PATH_TO_DS_1 = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_1_PREP = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/Features/Verification/"
PATH_TO_DS_2 = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_2_PREP = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/Features/Verification/"

# Set the limit of users' data to load. Choose values between 1 and 21
_users_limit = 5
# Set the limit of sessions to load. Choose values between 1 and 3
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

    # Define a list of users. Each user will be a Measurement instance
    users_measurements = []

    # Load each user data
    for i in range(1, _users_limit + 1):
        current_user_measurement = []

        # Define a dictionary containing the indexes of record and the relative sessions for enabling session splitting
        # For example: 1:100, 2:500, 3:580 means that the first 100 records are from the first session and so on
        sessions = dict()

        # Load each session
        for j in range(1, _session_limit + 1):

            # Read matlib file
            dataset_path = "{0}s{1}_s{2}.mat".format(path, i, j)

            # Load matlib file
            dataset = scipy.io.loadmat(dataset_path)

            # Extract recordings stored in the file
            table = dataset['recording']

            # Transpose table
            table = table.T

            # Drop "Counter" and "Interpolated" as they are useless values
            table = table[2:]

            # Gather recordings of different sessions
            if j == 1:
                # The first session creates a new list of recordings
                current_user_measurement = table.tolist()
            else:
                # From the second session append values to the already created list
                for sensor_number in range(0, len(current_user_measurement)):
                    current_user_measurement[sensor_number].extend(table[sensor_number])

            # Saving session information
            sessions[j] = len(current_user_measurement[0])

        if _logging:
            print("## INFO: loading {0} measurements for subject {1}".format(len(current_user_measurement[0]), i))

        # Create an instance of Measurement
        measurement = Measurement(current_user_measurement, subject_id=i, sessions=sessions)

        # Append the user relative Measurement instance to the list of users
        users_measurements.append(measurement)

    return users_measurements
