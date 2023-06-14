# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import time


class Stopwatch:

    def __init__(self):
        self._start_time = None
        self._stop_time = None
        self._elapsed_time = 0.
        self._total_time = 0.

    def start(self):
        self._start_time = time.time()
        self._elapsed_time = 0.

    def stop(self):
        self._stop_time = time.time()
        self._elapsed_time = self._stop_time - self._start_time
        self._total_time += self._elapsed_time

    def get_total_time(self):
        return self._total_time

    def get_elapsed_time(self):
        return self._elapsed_time

    def reset(self):
        self._start_time = None
        self._stop_time = None
        self._elapsed_time = 0.
        self._total_time = 0.

