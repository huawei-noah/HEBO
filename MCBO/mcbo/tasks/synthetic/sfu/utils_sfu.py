from typing import Dict, Type, Union, List, Optional, Any

import numpy as np

from mcbo.tasks.synthetic.sfu.sfu_base import SfuFunction
from mcbo.tasks.synthetic.sfu.sfu_functions import MANY_LOCAL_MINIMA, BOWL_SHAPED, PLATE_SHAPED, VALLEY_SHAPED, \
    STEEP_RIDGES, OTHER

SFU_FUNCTIONS: Dict[str, Type[SfuFunction]] = dict(**MANY_LOCAL_MINIMA, **BOWL_SHAPED, **PLATE_SHAPED, **VALLEY_SHAPED,
                                                   **STEEP_RIDGES, **OTHER)


def _sfu_search_space_params_factory(variable_type: Union[str, List[str]], num_dims: Union[int, List[int]],
                                     lb: Union[float, np.ndarray], ub: Union[float, np.ndarray],
                                     num_categories: Optional[Union[int, List[int]]] = None) \
        -> List[Union[Dict[str, Union[Union[str, float, int], Any]], Dict[str, Union[Union[str, object], Any]], Dict[
            str, Union[str, float, int]], Dict[str, Union[str, object]]]]:
    # Basic checks to ensure all arguments are correct
    assert isinstance(variable_type, (str, list))
    assert isinstance(num_dims, (int, list))

    if isinstance(variable_type, list):
        assert isinstance(num_dims, list)
    else:
        assert isinstance(num_dims, int)

    if isinstance(variable_type, list):
        for var_type in variable_type:
            assert isinstance(var_type, str)
            assert var_type in ['num', 'int', 'nominal', 'ordinal']

    if isinstance(variable_type, list) and ('nominal' in variable_type or 'ordinal' in variable_type):
        assert isinstance(num_categories, list)
        for num_cats, var_type in zip(num_categories, variable_type):
            if var_type in ['nominal', 'ordinal']:
                assert isinstance(num_cats, int) and num_cats > 0

    elif isinstance(variable_type, str) and (variable_type == 'nominal' or variable_type == 'ordinal'):
        assert isinstance(num_categories, int) and num_categories > 0

    if isinstance(num_dims, list):
        for num in num_dims:
            assert isinstance(num, int)

    assert isinstance(lb, np.ndarray) or isinstance(lb, (int, float))
    assert isinstance(ub, np.ndarray) or isinstance(ub, (int, float))
    assert np.all(lb < ub)

    params = []
    counter = 1

    def get_bounds_i(ind: int):
        if isinstance(lb, np.ndarray):
            lb_i = lb[ind]
        else:
            lb_i = lb
        if isinstance(ub, np.ndarray):
            ub_i = ub[ind]
        else:
            ub_i = ub
        return lb_i, ub_i

    if isinstance(variable_type, list):
        for i, var_type in enumerate(variable_type):
            lb_i, ub_i = get_bounds_i(counter)

            if var_type in ['num', 'int']:
                for _ in range(num_dims[i]):
                    params.append({'name': f'var_{counter}', 'type': var_type, 'lb': lb_i, 'ub': ub_i})
                    counter += 1
            elif var_type in ['nominal', 'ordinal']:
                categories = np.linspace(lb_i, ub_i, num_categories[i]).tolist()
                for _ in range(num_dims[i]):
                    params.append({'name': f'var_{counter}', 'type': var_type, 'categories': categories})
                    counter += 1

    else:
        if variable_type in ['num', 'int']:
            for i in range(num_dims):
                lb_i, ub_i = get_bounds_i(i)
                params.append({'name': f'var_{counter}', 'type': variable_type, 'lb': lb_i, 'ub': ub_i})
                counter += 1
        elif variable_type in ['nominal', 'ordinal']:
            for i in range(num_dims):
                lb_i, ub_i = get_bounds_i(i)
                categories = np.linspace(lb_i, ub_i, num_categories).tolist()
                params.append({'name': f'var_{counter}', 'type': variable_type, 'categories': categories})
                counter += 1

    return params


def default_sfu_params_factory(task_name: str, num_dims: int, task_name_suffix: Optional[str] = None):
    assert isinstance(task_name, str)
    assert isinstance(num_dims, int)

    if task_name == 'ackley':
        params = {'num_dims': num_dims,
                  'lb': -32.768,
                  'ub': 32.768,
                  'a': 20,
                  'b': 0.2,
                  'c': 2 * np.pi
                  }

    elif task_name == 'griewank':
        params = {'num_dims': num_dims,
                  'lb': -600,
                  'ub': 600
                  }

    elif task_name == 'langermann':
        params = {'num_dims': num_dims,
                  'lb': 0,
                  'ub': 10
                  }
        if num_dims == 2:
            params['m'] = 5
            params['c'] = np.array([[1., 2., 5., 2., 3.]])
            params['a'] = np.array([[3., 5.], [5., 2.], [2., 1.], [1., 4], [7., 9]])

    elif task_name == 'levy':
        params = {'num_dims': num_dims,
                  'lb': -10,
                  'ub': 10
                  }

    elif task_name == 'rastrigin':
        params = {'num_dims': num_dims,
                  'lb': -5.12,
                  'ub': 5.12
                  }

    elif task_name == 'schwefel':
        params = {'num_dims': num_dims,
                  'lb': -500,
                  'ub': 500
                  }

    elif task_name == 'perm0':
        params = {'num_dims': num_dims,
                  'lb': -num_dims,
                  'ub': num_dims,
                  'beta': 10
                  }

    elif task_name == 'rot_hyp':
        params = {'num_dims': num_dims,
                  'lb': -65.536,
                  'ub': 65.536,
                  }

    elif task_name == 'sphere':
        params = {'num_dims': num_dims,
                  'lb': -5.12,
                  'ub': 5.12,
                  }

    elif task_name == 'modified_sphere':
        params = {'num_dims': num_dims,
                  'lb': 0,
                  'ub': 1,
                  }

    elif task_name == 'sum_pow':
        params = {'num_dims': num_dims,
                  'lb': -1,
                  'ub': 1,
                  }

    elif task_name == 'sum_squares':
        params = {'num_dims': num_dims,
                  'lb': -10,
                  'ub': 10,
                  }

    elif task_name == 'trid':
        params = {'num_dims': num_dims,
                  'lb': -num_dims ** 2,
                  'ub': num_dims ** 2,
                  }

    elif task_name == 'power_sum':
        params = {'num_dims': num_dims,
                  'lb': 0.,
                  'ub': num_dims
                  }
        if num_dims == 4:
            params['b'] = np.array([[8., 18., 44., 114.]])

    elif task_name == 'zakharov':
        params = {'num_dims': num_dims,
                  'lb': -5,
                  'ub': 10
                  }

    elif task_name == 'dixon_price':
        params = {'num_dims': num_dims,
                  'lb': -10,
                  'ub': 10
                  }

    elif task_name == 'rosenbrock':
        params = {'num_dims': num_dims,
                  'lb': -5,
                  'ub': 10
                  }

    elif task_name == 'michalewicz':
        params = {'num_dims': num_dims,
                  'lb': 0,
                  'ub': np.pi
                  }

    elif task_name == 'perm':
        params = {'num_dims': num_dims,
                  'lb': -num_dims,
                  'ub': num_dims,
                  'beta': 10
                  }
    elif task_name == 'powell':
        params = {'num_dims': num_dims,
                  'lb': -4,
                  'ub': 5,
                  }
    elif task_name == 'styblinski_tang':
        params = {'num_dims': num_dims,
                  'lb': -5,
                  'ub': 5,
                  }

    else:
        raise NotImplemented(f'Task {task_name} is not implemented')
    if task_name_suffix is not None:
        params["task_name_suffix"] = task_name_suffix
    return params
