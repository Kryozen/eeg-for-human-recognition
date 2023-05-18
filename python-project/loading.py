import scipy
import pandas as pd

# Setting up
PATH_TO_DS_1 = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_1_PREP = "/home/smasu/Documents/FVAB/BED_Biometric_EEG_dataset/BED_Biometric_EEG_dataset/BED/Features/Verification/"
PATH_TO_DS_2 = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/RAW_PARSED/"
PATH_TO_DS_2_PREP = "/Users/mattiadargenio/Desktop/Unisa/Corsi/1:2/biometria/ProgettoEEG/BED_Biometric_EEG_dataset/BED/Features/Verification/"

# Set the limit of users' data to populate the pandas dataframe. Choose values between 1 and 21
_users_limit = 5
# Set the limit of sessions to use for the pandas dataframe. Choose values between 1 and 3
_session_limit = 2
# Set True if you want to show logs
_logging = True


def load(path):
    """
    Loads the matlib dataset into a pandas data stracture
    :param path: the path to the dataset
    :return: an array of _user_limit pandas DataFrame containing each an unspecified number of measurements.
    Each user is identified as its position in the array.
    """
    # Loading 18 labels as given in the instructions

    labels = [
        'COUNTER',         # this field should be ignored
        'INTERPOLATED',    # this field should be ignored
        'F3',
        'FC5',
        'AF3',
        'F7',
        'T7',
        'P7',
        'O1',
        'O2',
        'P8',
        'T8',
        'F8',
        'AF4',
        'FC6',
        'F4',
        'UNIX_TIMESTAMP'
    ]

    users_measurements = []

    for i in range(1, _users_limit + 1):
        current_user_measurement = []
        for j in range(1, _session_limit + 1):
            # Reading .mat file
            dataset_path = "{0}s{1}_s{2}.mat".format(path, i, j)

            # Loading .mat file
            dataset = scipy.io.loadmat(dataset_path)

            # Extracting recordings
            table = dataset['recording']

            # Gathering recordings
            current_user_measurement.extend(table)

        if _logging:
            print("## INFO: loading {} measurements for subject {}".format(len(current_user_measurement), i))

        users_measurements.append(current_user_measurement)

    # Logging
    if _logging:
        print("## INFO: loading {} subjects".format(len(users_measurements)))

    array_of_pd_dataframes = []
    # Convert each element into a pandas DataFrame
    for single_user_measurements in users_measurements:
        df = pd.DataFrame(single_user_measurements, columns=labels)
        df = df.drop(["INTERPOLATED"], axis=1)

        array_of_pd_dataframes.append(df)

    return array_of_pd_dataframes


def gather(users_measurements, users_names):
    """

    :param users_measurements:
    :param users_names:
    :return:
    """

    # Add ID column to each pd dataframe
    for i, user_measurement in enumerate(users_measurements):
        user_measurement["ID"] = users_names[i]

    # Concatenate all the pandas dataframes
    gathered = pd.concat([df for df in users_measurements])

    return gathered
