import abc


class BaseActor(abc):
    def __init__(self):
        pass

    def observe(self):
        raise NotImplementedError

    def suggest(self):
        raise NotImplementedError
