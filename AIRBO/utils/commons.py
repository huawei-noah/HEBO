import sys, os
cur_dir = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.dirname(os.path.dirname(cur_dir))
sys.path.append(root_path)
import matplotlib
matplotlib.use("Agg")
# matplotlib.use("Qt5Agg")
matplotlib.rcParams.update({'figure.max_open_warning': 0})
import torch
from torch import cuda
import inspect
from datetime import datetime
import _pickle as pickle
import sys
import gc
import time
import os
from ast import literal_eval

COLORS = ["r", 'g', 'b', 'c', 'y', 'm', 'k']
MARKERS = ["x", "+", "d", "s", ".", ]
LINE_STYLES = ["-", ":", "--", "-."]


def str2time(strTime, strFormat="%Y%m%d_%H"):
    return datetime.strptime(strTime, strFormat)


def time2str(dtTime, strFormat='%Y%m%d_%H%M%S'):
    return dtTime.strftime(strFormat)


def serialize_obj(obj, file_path):
    # print("serialize to {}".format(file_path))
    with open(file_path, 'wb') as hFile:
        pickle.dump(obj, hFile)


def pickle_load(file_path):
    obj = None
    with open(file_path, 'rb') as hFile:
        gc.disable()  # trick to boost pickle load
        obj = pickle.load(hFile, encoding='latin1') if sys.version_info > (3, 0) \
            else pickle.load(hFile)
        gc.enable()
    return obj


def ensure_dir_exist(strDir):
    if not os.path.exists(strDir):
        os.makedirs(strDir)


def getCurrentTime_r():
    return time.time()


def get_current_time_str(strFormat="%Y%m%d%H%M%S"):
    return time2str(datetime.now(), strFormat)


def value_clip(v, v_min, v_max):
    return max(min(v_max, v), v_min)


def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return list(subclasses)


def dict_to_str(dc, precision=4):
    items = []
    for k,v in dc.items():
        items.append("{}={}".format(k, str(round(v, precision)) if type(v) is float else v))
    return ", ".join(items)


def parse_dict(raw):
    """
    Helper method for parsing string-encoded <dict>
    """
    try:
        pattern = raw.replace('\"', '').replace("\\'", "'")
        return literal_eval(pattern)
    except Exception as e:
        raise Exception('Failed to parse string-encoded <dict> {} with exception {}'.format(raw, e))


def parse_list(raw):
    """
    Helper method for parsing string-encoded <list>
    """
    try:
        pattern = raw.replace('\"', '').replace("\\'", "'")
        return literal_eval(pattern)
    except Exception as e:
        raise Exception('Failed to parse string-encoded <list> {} with exception {}'.format(raw, e))

def dict_2_str(d):
    """ """
    s = []
    for k, v in d.items():
        s.extend([k, str(v)])
    return '-'.join(s)



def get_less_used_gpu(gpus=(0,), debug=False):
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'
    else:
        warn = ''

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated



def free_memory(to_delete: list, debug=False):
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        if _var is not None:
            calling_namespace.f_locals.pop(_var, None)
    gc.collect()
    cuda.empty_cache()

    if debug:
        print('After:')
        get_less_used_gpu(debug=True)

def set_gmem_usage(device, reserved_gmem = 8):
    if device.type == 'cpu':
        total_gmem, ratio = None, None
    else:
        total_gmem = torch.cuda.get_device_properties(device).total_memory
        ratio = max(1.0 - reserved_gmem * (1024**3.0) / total_gmem, 0.0)
        torch.cuda.set_per_process_memory_fraction(ratio, device)

    return total_gmem, ratio
