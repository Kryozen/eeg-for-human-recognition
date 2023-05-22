class Measurement:
    """

    """

    columns = [
    #    'COUNTER',  # this field should be ignored
    #    'INTERPOLATED',  # this field should be ignored
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
    values = None
    subject_id = None
    sessions = None

    def __init__(self, measurements, subject_id=None, sessions=None):
        self.values = measurements
        self.subject_id = subject_id
        self.sessions = sessions

    def __str__(self):
        to_string = 'Subject {0} - Session {1}\n'.format(self.subject_id, self.sessions)
        for column in self.columns:
            to_string += '{:20} '.format(column)
        to_string += '\n'
        for i in range(0, len(self.values[0])):
            for column, _ in enumerate(self.columns):
                to_string += '{:20} '.format(self.values[column][i])
            to_string += '\n'

        return to_string
