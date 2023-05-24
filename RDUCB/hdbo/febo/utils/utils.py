from pydoc import locate  as pydoc_locate

from io import StringIO
import numpy as np
import sys
import select
import tty
import termios
from datetime import datetime
import os.path
import logging
import importlib
import pkgutil
import coloredlogs
import yaml
import math
import scipy.special


def sherrman_morrision_update(A_inv, x_inc):
        A_x = A_inv.dot(x_inc)
        return  A_inv - A_x.dot(x_inc.reshape(1,-1)).dot(A_inv)/(1 + np.asscalar(x_inc.reshape(1,-1).dot(A_x)))


# https://arxiv.org/abs/1309.1541
def project_onto_simplex(p, m):
    u = np.sort(np.copy(p))[::-1]
    rho = -1
    s = 0
    s_rho = 0
    j = 1

    for j in range(m):
        s += u[j]
        if u[j] + 1 / (j + 1) * (1 - s) > 0:
            rho = j + 1
            s_rho = s
        else:
            break

    l = 1 / rho * (1 - s_rho)

    for i in range(m):
        p[i] = max(p[i] + l, 0)

def cartesian(arrays, out=None):
        """
        Generate a cartesian product of input arrays.

        Parameters
        ----------
        arrays : list of array-like
                1-D arrays to form the cartesian product of.
        out : ndarray
                Array to place the cartesian product in.

        Returns
        -------
        out : ndarray
                2-D array of shape (M, len(arrays)) containing cartesian products
                formed of input arrays.

        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
                     [1, 4, 7],
                     [1, 5, 6],
                     [1, 5, 7],
                     [2, 4, 6],
                     [2, 4, 7],
                     [2, 5, 6],
                     [2, 5, 7],
                     [3, 4, 6],
                     [3, 4, 7],
                     [3, 5, 6],
                     [3, 5, 7]])

        """

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = n / arrays[0].size
        m = int(m)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
        return out


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " \
                             "(or 'y' or 'n').\n")

def query_options(question, options, default=None):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is one of "yes" or "no".
    """
    if default is None:
        prompt = " (%s) " % "/".join(options)
    elif default.lower() in options:
        options_with_default = options.copy()
        options_with_default[options_with_default.index(default)] = "[%s]" % default
        prompt = " (%s) " % "/".join(options_with_default)
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input()
        if default is not None and choice == '':
            return default
        elif choice in options:
            return choice
        else:
            sys.stdout.write("Please respond with %s .\n" % prompt)

def query_continue():
    return query_yes_no('Do you want to continue?')

class NonBlockingConsole(object):

    def __init__(self, interupt_keys=['\x1b']):
        self._interupt_keys = interupt_keys
        self.old_settings = termios.tcgetattr(sys.stdin)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def __del__(self):
        self.stop()

    def key_pressed(self):
        key_pressed = False
        while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            if sys.stdin.read(1) in self._interupt_keys:
                key_pressed = True  # not returning here to empty stdin stream
        if key_pressed:
            self.stop()

        return key_pressed

    def stop(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def start(self):
        tty.setcbreak(sys.stdin.fileno())


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def in_unit_cube(x):
    return all(x >= 0) and (all(x <= 1))

def str_to_bool(s):
    if s in ['true', 'True', '1', 'T']:
        return True
    if s in ['false', 'False', '0', 'F']:
        return False

    raise ValueError

def join_path_if_exists(*args):
    path = os.path.join(*args)
    if not os.path.exists(path):
        raise Exception(f"Could not find {path}.")

    return path


def join_path_if_not_exists(*args):
    path = os.path.join(*args)
    if os.path.exists(path):
        raise Exception(f"Path {path} already exists.")

    return path

def mkdir_if_not_exits(path):
    if not os.path.exists(path):
        os.mkdir(path)

def mkdir_fail_if_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        raise Exception(f"Directory {dir} already exists.")


def join_dtypes(*args):
    """
    Helper which joins dtypes d1 and d2, and returns a new dtype containing the fields of both d1 and d2.
    TODO: Does not check for field name collisions right now.

    Args:
        d1:
        d2:

    Returns:

    """
    fields = []
    for dtype in args:
        fields += [(f, dt[0]) for f, dt in dtype.fields.items()]

    return np.dtype(fields)

def join_dtype_arrays(a1, a2, target_dtype):
    """
    Initializes a new array with dtype target_dtype, and copies matching fields from a1 and a2 to the new array.

    Args:
        a1:
        a2:
        target_dtype:

    Returns:

    """
    new_ar = np.zeros(shape=(), dtype=target_dtype)
    fields1 = a1 if isinstance(a1, dict) else a1.dtype.fields
    for f in fields1:
        if f in target_dtype.fields:
            new_ar[f] = a1[f]

    fields2 = a2 if isinstance(a2, dict) else a2.dtype.fields
    for f in fields2:
        if f in target_dtype.fields:
            new_ar[f] = a2[f]

    return new_ar



class Logger:
    """
    Helper class to create loggers.
    Colors messages in terminal.
    Adds filehandler, a soon as a path is set.
    """
    def __init__(self):
        self._path = None
        self._level = 'INFO'
        self._file_level = 'INFO'
        self._loggers = []
        self._file_handler = None

    def __call__(self, name):
        """ return a logger with given name """
        logger = logging.getLogger(name)
        self._loggers.append(logger)
        coloredlogs.install(fmt='%(message)s', logger=logger, level=self._level, stream=sys.stdout)

        if self._path:
            logger.addHandler(self._file_handler)
        return logger

    def set_level(self, level):
        self._level = level
        # update level of existing loggers
        for logger in self._loggers:
            coloredlogs.install(fmt='%(message)s', logger=logger, level=self._level, stream=sys.stdout)

    def set_filehandler_level(self, level):
        self._file_level = level
        if not self._file_handler is None:
            self._file_handler.setLevel(level)

    def set_path(self, path):
        # remove previous filehandler
        if not self._file_handler is None:
            for logger in self._loggers:
                logger.removeHandler(self._file_handler)

        self._path = path
        self._file_handler = logging.FileHandler(os.path.join(path,  'febo.log'))
        # by default, set level to 'INFO'
        self._file_handler.setLevel(self._file_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")
        self._file_handler.setFormatter(formatter)
        # add filehandler to existing loggers.
        for logger in self._loggers:
            logger.addHandler(self._file_handler)

get_logger = Logger()



def import_submodules(package, recursive=True):
    """
    https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
    Import all submodules of a module, recursively, including subpackages

    Args:
        package (str | module): package (name or actual module)
        recursive:

    Returns (dict[str, types.ModuleType]):

    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

def locate(path):
    """
    Dynamically loads "path", raises an Exception if not found.

    Args:
        path:

    Returns:

    """
    (modulename, classname) = path.rsplit('.', 1)

    m = __import__(modulename, globals(), locals(), [classname])
    if not hasattr(m, classname):
        raise ImportError(f'Could not locate "{path}".')
    return getattr(m, classname)

def dict_to_yaml(dict):
    ss = StringIO()
    yaml.dump(dict, ss, default_flow_style=False)
    str = ss.getvalue()
    ss.close()
    return str


def split_int(i, p):
    """
    Split i into p buckets, such that the bucket size is as equal as possible
    Args:
        i: integer to be split
        p: number of buckets

    Returns: list of length p, such that sum(list) = i, and the list entries differ by at most 1

    """
    split = []
    n = i / p  # min items per subsequence
    r = i % p  # remaindered items

    for i in range(p):
        split.append(int(n + (i < r)))

    return split

def nan_stderr(data, axis=None):
    return np.nanstd(data, axis=axis)/np.sqrt(np.count_nonzero(~np.isnan(data), axis=axis))


SQRT2PI = 1/math.sqrt(2 * math.pi)

def norm_pdf(x):
    return np.exp(-x ** 2 / 2) * SQRT2PI

# this is way faster than scipy version
def norm_cdf(x):
    return (1.0 + scipy.special.erf(x / np.sqrt(2.0))) / 2.0

def parse_int_set(nputstr=""):
    """
        return a set of selected values when a string in the form:
        1-4,6
        would return:
        1,2,3,4,6
        as expected...

        Taken from https://stackoverflow.com/a/712483
    """
    if isinstance(nputstr, int):
        return set([nputstr])
    selection = set()
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        if len(i) > 0:
            if i[:1] == "<":
                i = "1-%s"%(i[1:])
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token)-1]
                    for x in range(first, last+1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    if len(invalid) > 0:
        Exception(f"Invalid set: {invalid}")
    return selection