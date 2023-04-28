from threading import Event, Thread
from febo.algorithms import Algorithm
import sys


class ThreadAlgorithm(Algorithm):
    """
        This algorithms runs a minimize route in a separate thread. Using this base class,
        one can easily adapt existing `minimize` (like scipy's optimize.minimize), which
        by default don't allow manual 'stepping' through the optimization via next() and
        add_data(x,y).
    """

    def initialize(self, **kwargs):
        super().initialize(**kwargs)

        self._exit_thread = False  # exit flag

        # events for mutual locking
        self._event_x_ready = Event()
        self._event_x_ready.clear()

        self._event_y_ready = Event()
        self._event_y_ready.clear()

        # start thread
        self._optimizer_thread = Thread(target=self._minimize)
        self._optimizer_thread.start()

    def _next(self):
        # let the optimizer thread calculate x
        self._event_x_ready.wait()
        return self._x

    def add_data(self, evaluation):
        super().add_data(evaluation)
        self._y = evaluation['y']

        self._event_x_ready.clear()
        self._event_y_ready.set()
        self._event_x_ready.wait()  # block main thread until next x is calculated, or optimizer thread terminates

    def finalize(self):
        self._exit_thread = True
        self._event_y_ready.set()  # unblock optimizer thread

        self._optimizer_thread.join()
        return super().finalize()

    def _minimize(self):
        self.minimize()
        self._exit = True
        self._event_x_ready.set()  # unblock main thread

    def minimize(self):
        """ Start the minimize routine here.
        e.g. call here: scipy.optimize.minimize(self.f, self.x0, method='Nelder-Mead')
        """
        raise NotImplementedError

    def f(self, x):
        self._x = x

        self._event_x_ready.set()
        self._event_y_ready.wait()
        self._event_y_ready.clear()

        # if exit flag is set, terminate thread
        if self._exit_thread:
            sys.exit()

        return -1*self._y # maximize signal, hence "-1*"
