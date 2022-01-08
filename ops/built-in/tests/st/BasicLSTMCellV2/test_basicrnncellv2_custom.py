#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from tbe.dsl import auto_schedule
from te import tvm
from te import platform as cce_conf
from impl.basic_lstm_cell_v2 import basic_lstm_cell_v2


def test_basicrnncellv2():
    '''
    for basicrnncellv2 single op
    '''
    input_list = [{"shape": (1, 3, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 48),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "ND", "ori_shape": (16,),"ori_format": "ND"},
                    None,
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #h
                    {"shape": (2, 1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #c
                    {"shape": (5, 8,16,16), "dtype": "float16", "format": "FRACTAL_ZN_LSTM", "ori_shape": (80, 128),"ori_format": "ND"}, #w
                    {"shape": (128,), "dtype": "float16", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},  #b
                    None, #mask
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"}, #ht
                    {"shape": (2, 1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 32),"ori_format": "ND"} #ct
                    ]
    with tbe.common.context.op_context.OpContext():
        basic_lstm_cell_v2(*input_list)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_basicrnncellv2()
    cce_conf.te_set_version(soc_version)
