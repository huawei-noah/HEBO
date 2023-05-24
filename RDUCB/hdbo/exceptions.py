# For BO early Termination
class EarlyTerminationException(Exception):
    def __init__(self, message, metrics):

        # Call the base class constructor with the parameters it needs
        super(EarlyTerminationException, self).__init__(message)

        self.metrics = metrics