#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from tbe.dsl import auto_schedule
from te import tvm
from te import platform as cce_conf
from impl.non_zero_with_value import non_zero_with_value

def test_non_zero_with_value():
    '''
    for non_zero_with_value single op
    '''
    input_list = [{
        'ori_shape': (1000, 21136),
        'shape': (1000, 21136),
        'ori_format': 'ND',
        'format': 'ND',
        'dtype': 'float32'
    }, {
        'ori_shape': (21136000),
        'shape': (21136000),
        'ori_format': 'ND',
        'format': 'ND',
        'dtype': 'float32'
    }, {
        'shape': (2, 21136000),
        'format': 'ND',
        'dtype': 'int32'
    },{
        'shape': (1),
        'format': 'ND',
        'dtype': 'int32'
    }, True]
    with tbe.common.context.op_context.OpContext():
        non_zero_with_value(*input_list)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Ascend710")
    test_non_zero_with_value()
    cce_conf.te_set_version(soc_version)
