#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from tbe.dsl import auto_schedule
from te import tvm
from te import platform as cce_conf
from impl.trans_data_rnn import trans_data_rnn


def test_transdatarnn():
    '''
    for transdatarnn single op
    '''
    input_list = [{
        'ori_shape': (1, 2, 16, 16),
        'shape': (1, 2, 16, 16),
        'ori_format': 'FRACTAL_ZN_RNN',
        'format': 'FRACTAL_ZN_RNN',
        'dtype': 'float16'
    }, {
        'ori_shape': (2, 24),
        'shape': (2, 24),
        'ori_format': 'ND',
        'format': 'ND',
        'dtype': 'float16'
    }, 'FRACTAL_ZN_RNN', 'ND', 2, 12]
    with tbe.common.context.op_context.OpContext():
        trans_data_rnn(*input_list)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Ascend310")
    test_transdatarnn()
    cce_conf.te_set_version(soc_version)
