from gym import Env
import numpy as np


class SafeEnv(Env):
    """Safe environment wrapper."""
    def step(self, action:np.ndarray) -> np.ndarray:
        state = self._get_state()
        next_state, reward, done, info = super().step(action)
        info['cost'] = self._safety_cost_fn(state, action, next_state)
        return next_state, reward, done, info

    def _get_state(self):
        """Returns current state. Uses _get_obs() method if it is implemented."""
        if hasattr(self, "_get_obs"):
            return self._get_obs()
        else:
            raise NotImplementedError("Please implement _get_obs method returning the current state")                     

    def _safety_cost_fn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> np.ndarray:        
        """Returns current safety cost."""
        raise NotImplementedError("Please implement _safety_cost_fn method returning the current safety cost")    