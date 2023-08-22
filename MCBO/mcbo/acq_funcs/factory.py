# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from mcbo.acq_funcs.additive_lcb import AddLCB
from mcbo.acq_funcs.cei import CEI
from mcbo.acq_funcs.ei import EI
from mcbo.acq_funcs.lcb import LCB
from mcbo.acq_funcs.pi import PI
from mcbo.acq_funcs.thompson_sampling import ThompsonSampling


def acq_factory(acq_func_id: str, **kwargs):
    if acq_func_id == 'lcb':
        beta = kwargs.get('beta', 1.96)
        acq_func = LCB(beta)

    elif acq_func_id == "addlcb":
        beta = kwargs.get('beta', 1.96)
        acq_func = AddLCB(beta)

    elif acq_func_id == 'ei':
        acq_func = EI(augmented_ei=False)

    elif acq_func_id == 'aei':
        acq_func = EI(augmented_ei=True)

    elif acq_func_id == 'pi':
        acq_func = PI()

    elif acq_func_id == 'ts':
        acq_func = ThompsonSampling()

    elif acq_func_id == 'cei':
        acq_func = CEI(num_constr=kwargs["num_constr"], augmented_ei=False)

    elif acq_func_id == 'caei':
        acq_func = CEI(num_constr=kwargs["num_constr"], augmented_ei=True)

    else:
        raise NotImplementedError(f'Acquisition function {acq_func_id} is not implemented.')

    return acq_func
