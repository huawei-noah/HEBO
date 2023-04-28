from febo.plots.plot import DataPlot


class Time(DataPlot):
    def __init__(self, experiment):
        super().__init__(experiment)
        self._title = 'Time per Step'

    def _get_value(self, row, t):
        return row["time"]

class CumulativeTime(DataPlot):

    def __init__(self, experiment):
        super().__init__(experiment)
        self._title = 'Cumulative Time'

    def _init_dset(self, dset):
        self._total_time = 0

    def _get_value(self, row, t):
        self._total_time += row["time"]
        return self._total_time

