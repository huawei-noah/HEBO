# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')

from hebo.design_space.design_space import DesignSpace

def test_design_space():
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : 0, 'ub' : 7}, 
        {'name' : 'x1', 'type' : 'int', 'lb' : 0, 'ub' : 7}, 
        {'name' : 'x2', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1e-2, 'base' : 10}, 
        {'name' : 'x3', 'type' : 'cat', 'categories' : ['a', 'b', 'c']}, 
        {'name' : 'x4', 'type' : 'bool'}, 
        {'name' : 'x5', 'type' : 'pow_int', 'lb' : 1, 'ub' : 10000, 'base' : 10}, 
        {'name' : 'x6', 'type' : 'int_exponent', 'lb' : 32, 'ub' : 1024, 'base' : 2}, 
        {'name' : 'x7', 'type' : 'step_int', 'lb' : 1, 'ub' : 9, 'step' : 2}, 
    ])
    assert space.numeric_names   == ['x0', 'x1', 'x2', 'x4', 'x5', 'x6', 'x7']
    assert space.enum_names      == ['x3']
    assert space.para_names      == space.numeric_names + space.enum_names
    assert space.num_paras       == 8
    assert space.num_numeric     == 7
    assert space.num_categorical == 1

    samp    = space.sample(10)
    x, xe   = space.transform(samp)
    x_, xe_ = space.transform(space.inverse_transform(x, xe))
    assert (x - x_).abs().max() < 1e-4
    assert (xe == xe_).all()

    assert (space.opt_lb <= space.opt_ub).all()

    assert not space.paras['x0'].is_discrete
    assert space.paras['x1'].is_discrete
    assert not space.paras['x2'].is_discrete
    assert space.paras['x3'].is_discrete
    assert space.paras['x4'].is_discrete
    assert space.paras['x5'].is_discrete
    assert space.paras['x6'].is_discrete
    assert space.paras['x7'].is_discrete

    assert not space.paras['x0'].is_discrete_after_transform
    assert space.paras['x1'].is_discrete_after_transform
    assert not space.paras['x2'].is_discrete_after_transform
    assert space.paras['x3'].is_discrete_after_transform
    assert space.paras['x4'].is_discrete_after_transform
    assert not space.paras['x5'].is_discrete_after_transform
    assert space.paras['x6'].is_discrete_after_transform
    assert space.paras['x7'].is_discrete_after_transform
    
    for _, para in space.paras.items():
        assert para.is_discrete == (type(para.sample(1).tolist()[0]) != float)

        if para.is_discrete_after_transform:
            samp = para.sample(5)
            x    = para.transform(samp)
            assert (x == x.round()).all()
