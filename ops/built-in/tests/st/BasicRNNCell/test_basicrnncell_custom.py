#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from tbe.dsl import auto_schedule
from te import tvm
from te import platform as cce_conf
from impl.basic_rnn_cell import basic_rnn_cell


def test_basicrnncell():
    '''
    for basicrnncell single op
    '''
    input_list = [
        {"ori_shape": (1, 1, 1024), "shape": (64, 1, 16, 16), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"},
        {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,),"ori_format": "NCHW"},
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 512),"ori_format": "NCHW"},
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 512),"ori_format": "ND"},
                    {"shape": (64, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 1024),"ori_format": "HWCN"},
                    {"shape": (512,), "dtype": "float16", "format": "NCHW", "ori_shape": (512,),"ori_format": "NCHW"},
                    {"shape": (32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 512),"ori_format": "HWCN"},
                    {"shape": (32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 512),"ori_format": "HWCN"},
                    {"shape": (512,), "dtype": "float16", "format": "NCHW", "ori_shape": (512,),"ori_format": "NCHW"},
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 512),"ori_format": "ND"},
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1, 512),"ori_format": "ND"},
                    True, 0]
    with tbe.common.context.op_context.OpContext():
        basic_rnn_cell(*input_list)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_basicrnncell()
    cce_conf.te_set_version(soc_version)
