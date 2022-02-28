# 2021.11.10-Add time formatter
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import datetime

from typing import Optional

import time


def time_formatter(t: float, show_ms: bool = False) -> str:
    """ Convert a duration in seconds to a str `dd:hh:mm:ss`

    Args:
        t: time in seconds
        show_ms: whether to show ms on top of dd:hh:mm:ss
    """
    n_day = time.gmtime(t).tm_yday - 1
    if n_day > 0:
        ts = time.strftime('%H:%M:%S', time.gmtime(t))
        ts = f"{n_day}:{ts}"
    else:
        ts = time.strftime('%H:%M:%S', time.gmtime(t))
    if show_ms:
        ts += f'{t - int(t):.3f}'.replace('0.', '.')
    return ts


def log(message, header: Optional[str] = None, end: Optional[str] = None):
    if header is None:
        header = ''
    print(f'[{header}' + ' {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message, end=end)


if __name__ == '__main__':
    d = 1
    h = 10
    m = 10
    s = 10
    ms = 123
    t = (d * 24 + h) * 3600 + m * 60 + s + ms * 1e-3
    print(time_formatter(t))
    print(time_formatter(t, show_ms=True))
