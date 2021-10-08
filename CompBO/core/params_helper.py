from typing import List, Dict, Callable, Optional, Any, Union, Tuple, Set
import numpy as np


class ParamSpace:
    """ Class to handle parameter space in a GPyOpt-like way

        Attributes:
            bounds: list of bounds (search space) formatted as required by GPyOpt
            transfos: list of dictionaries containing transformations to apply to each parameter mapping its search
            space to its "real" space (think of log-transformation to do search in log-space)
    """

    def __init__(self, bounds: List[Dict[str, Any]], transfos: List[Dict[str, Callable]],
                 rec_transfos: Optional[List[Dict[str, Callable]]] = None):
        self.bounds: List[Dict[str, Any]] = bounds
        self.transfos: List[Dict[str, Callable]] = transfos
        self.rec_transfos = rec_transfos
        assert self.check_params()

    @property
    def params_list_names(self) -> List[str]:
        """ Get list of parameters names"""
        return [b['name'] for b in self.bounds]

    def check_params(self) -> bool:
        """ Check that each parameter has an associated transformation """
        b: bool = list(map(lambda param_dic: param_dic['name'], self.bounds)) == list(
            map(lambda dic: list(dic.keys())[0], self.transfos))
        if b and self.rec_transfos is not None:
            b = list(map(lambda dic: list(dic.keys())[0], self.transfos)) == list(
                map(lambda dic: list(dic.keys())[0], self.rec_transfos))
        return b

    def get_int_mask(self, variables: Set[str]) -> List[int]:
        """ Get list of indices corresponding to discrete variables (used for rounding GP)

        Args:
            variables: list of variables to treat as discrete

        Returns:
            int_mask: list of corresponding indices

        """
        int_mask = []
        for i, k in enumerate(self.params_list_names):
            if k in variables:
                int_mask.append(i)
        return int_mask

    def clip_params(self, search_params: np.ndarray):
        """ Clip search_params within bounds (inplace) """
        for i in range(self.d):
            b_type: str = self.bounds[i]['type']
            assert b_type in ['continuous', 'discrete'], b_type
            a, b = self.bounds[i]['domain']
            if b_type == 'discrete':
                search_params[i] = round(search_params[i])
            search_params[i] = max(min(search_params[i], b), a)

    def get_real_params(self, params: Union[Dict[str, float], np.ndarray]) -> Dict[str, Any]:
        """ transform parameters according to `params_transfo` """
        return transform_params(params, self.transfos)

    def get_search_params(self, params: Union[Dict[str, float], np.ndarray]) -> Dict[str, Any]:
        """ transform parameters according to `rec_transfo` if specified """
        assert self.rec_transfos, "No reciprocal transformations specified"
        return transform_params(params, self.rec_transfos)

    def get_list_params_from_dict(self, params_dict: Dict[str, Any]) -> List[Any]:
        """ Get parameter list from hypeparameter dictionary
        Args:
            params_dict
        """
        values: List[Any] = []
        for bound_dict in self.bounds:
            k: str = bound_dict['name']
            assert k in params_dict, (k, params_dict)
            values.append(params_dict[k])
        return values

    def get_dict_params_from_list(self, params_list: Union[List, np.ndarray]) -> Dict[str, Any]:
        return {k: v for k, v in zip(self.params_list_names, params_list)}

    def get_array_bounds(self) -> np.ndarray:
        """ Convert `bounds` in a `2 x dim` array """
        array_bounds = np.zeros((2, len(self.bounds)))
        for i, search_b in enumerate(self.bounds):
            array_bounds[:, i] = search_b['domain']
        return array_bounds

    def get_random_search_point(self, n_points=1) -> np.ndarray:
        """ Return a random point within bounds """

        random_points: np.ndarray = np.zeros((n_points, self.d))
        for i, bound_dict in enumerate(self.bounds):
            a, b = bound_dict['domain']
            if bound_dict['type'] == 'continuous':
                random_points[:, i] = np.random.random(size=n_points) * (b - a) + a
            elif bound_dict['type'] == 'discrete':
                random_points[:, i] = np.random.choice(np.arange(a, b + 1), size=n_points)
            else:
                raise ValueError(f"Unexpected type for bound {bound_dict['name']}: {bound_dict['type']}")
        if n_points == 1:
            return random_points.flatten()
        return random_points

    def get_random_point(self) -> Dict[str, Any]:
        return self.get_real_params(self.get_random_search_point())

    @property
    def d(self):
        return len(self.bounds)


class OptParamSpace(ParamSpace):
    """ Class to handle optimizer hyperparameters that need to be tuned

    Attributes:
        bounds: list of bounds (search space) formatted as required by GPyOpt
        transfos: list of dictionaries containing transformations to apply to each hyperparameter mapping its search
                  space to its "real" space (think of log-transformation to do search in log-space)
        constraints: list of dictionaries of constraints following GPyOpt requirements (notice that constraints
                     are applied in search space, not after transformation)
    """

    def __init__(self, bounds: List[Dict[str, Any]], transfos: List[Dict[str, Callable]],
                 constraints: Optional[List[Dict[str, Any]]] = None,
                 rec_transfos: Optional[List[Dict[str, Callable]]] = None):
        super(OptParamSpace, self).__init__(bounds, transfos, rec_transfos=rec_transfos)
        self.constraints = constraints

    def get_scheduler(self, all_params_kwargs: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """ If optimizer associated to this object is used with a scheduler, recover scheduler name and kwargs

        Args:
            all_params_kwargs: kwargs for optimizer (if contains scheduler kwargs, then object should be instance of
            `OptSchedulerParams`)

        Returns:
              scheduler_name: `None`
              scheduler_kwargs: `{}`
        """
        return None, {}

    def get_true_opt_kwargs(self, all_params_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """ Some optimizers can require tuples as arguments, in which case reorganization of parameters is needed
            (e.g. `{'beta1': b1, 'beta2': b2}` -> `{'betas': (b1, b2)}` for Adam)

        Args:
            all_params_kwargs: optimizer (and scheduler) kwargs.

        Returns:
            Reorganized optimizer (and scheduler) kwargs
        """
        return all_params_kwargs

    def get_search_opt_kwargs(self, all_params_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return all_params_kwargs

    def reorder_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: params[k] for k in self.params_list_names}

    def get_search_params(self, hyperparams: Union[Dict[str, float], np.ndarray]) -> Dict[str, Any]:
        """ transform hyperparameters according to `rec_transfo` if specified """
        return super(OptParamSpace, self).get_search_params(self.get_search_opt_kwargs(hyperparams))


class OptScheduledParamSpace(OptParamSpace):
    """ Class for optimizer requiring scheduler tuning (supported scheduler being ExponentialLR) """

    def get_scheduler(self, all_params_kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        scheduler: str = 'ExponentialLR'
        scheduler_kwargs: dict[str, Any] = {'gamma': all_params_kwargs.pop('gamma')}
        return scheduler, scheduler_kwargs


def transform_params(params: Union[Dict[str, float], np.ndarray], params_transfo: List[Dict[str, Callable]]) -> Dict[
    str, Any]:
    """ transform hyperparameters according to `params_transfo` """
    aux_params: np.ndarray = None
    if isinstance(params, dict):
        aux_params = np.array(list(params.values()))
    else:
        aux_params = params
    assert aux_params.size == len(params_transfo), (aux_params.size, len(params_transfo), aux_params)
    params_dict = {}
    for hyperparam, hyperparam_transfo in zip(aux_params, params_transfo):
        assert len(hyperparam_transfo.keys()) == 1
        for hyperparam_name, tranfo in hyperparam_transfo.items():
            params_dict[hyperparam_name] = tranfo(hyperparam)
    return params_dict
