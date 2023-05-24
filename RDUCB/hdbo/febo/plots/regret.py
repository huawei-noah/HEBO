from febo.plots.plot import DataPlot
from febo.utils import get_logger

logger = get_logger("plotting")

class Regret(DataPlot):
    """
    Plot Cumulative Regret
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self._title = 'Cumulative Regret'
        self._max_value_warning = True

    def _init_dset(self, dset):
        self._check_max_value_available(dset)
        self._total = 0
        self._y_field = "y"
        if "y_exact" in dset.dtype.fields:
            self._y_field = "y_exact"

    def _get_value(self, row, t):
        self._total += self._to_regret(row[self._y_field], row)
        return self._total

    def _check_max_value_available(self, dset):
        if self._max_value_warning:
            if not "y_max" in dset.dtype.fields:
                logger.warning("No y_max values available. Showing reward plots instead of regret.")
                self._title = self._title.replace('Regret', 'Reward')
            self._max_value_warning = False

    def _to_regret(self, value, row):
        if 'y_max' in row.dtype.fields:
            return row["y_max"] - value
        else:
            return value


class SimpleRegret(Regret):
    """
    Plot Simple Regret
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self._title = 'Simple Regret'

    def _init_dset(self, dset):
        super()._init_dset(dset)

        if 'y_max' in dset.dtype.fields:
            self._best = 10e10
        else:
            self._best = -10e10

    def _get_value(self, row, t):
        value = self._to_regret(row[self._y_field], row)
        if 'y_max' in row.dtype.fields:
            self._best = min(self._best, value)
        else:
            self._best = max(self._best, value)

        return self._best


class InferenceRegret(Regret):
    """
    Plot Best Predicted
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self._title = 'Inference Regret'

    def _init_dset(self, dset):
        super()._init_dset(dset)
        if "y_exact_bp" in dset.dtype.fields:
            self._y_field = "y_exact_bp"
        elif "y_bp" in dset.dtype.fields:
            self._y_field = "y_bp"
        else:
            logger.warning(f"No field for best_predicted found. Fall back to {self._y_field}.")


    def _get_value(self, row, t):
        return self._to_regret(row[self._y_field], row)
