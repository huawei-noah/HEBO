from datetime import datetime
from functools import cmp_to_key


def get_timestr(fmt: str = "%Y%m%d-%H%M%S") -> str:
    now = datetime.now()  # current date and time
    return str(now.strftime(fmt))
