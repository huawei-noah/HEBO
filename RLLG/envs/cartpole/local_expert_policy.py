# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.



from typing import Optional
import numpy as np


class SafeScripted:
    """
    SafeScripted class for scripted control.
    """

    def __init__(self) -> None:
        pass

    def get_action(self, observation: np.ndarray, init_action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the action for scripted control.

        Parameters:
        ----------
        observation : np.ndarray
            The observation.
        init_action : Any, optional
            The initial action (default is None).

        Returns:
        ----------
        np.ndarray
            The scripted action.
        """
        pos = observation[0]
        if pos > 0:
            return np.float32(np.array([-0.999]))
        return np.float32(np.array([0.999]))
