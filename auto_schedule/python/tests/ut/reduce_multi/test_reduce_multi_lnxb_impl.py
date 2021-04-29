# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from layer_norm_x_backprop import layer_norm_x_backprop

warnings.filterwarnings("ignore")


def dsl_reduce_multi_lnxb(input_dy, input_x, input_variance, input_mean, 
                     input_gamma, output_pd_x, kernel_name='test_dsl_reduce_multi_lnxb'):
    layer_norm_x_backprop(input_dy, input_x, input_variance, input_mean,
                          input_gamma, output_pd_x,
                          kernel_name)


ut_case = OpUT("reduce_multi", "reduce_multi.test_reduce_multi_lnxb_impl", "dsl_reduce_multi_lnxb")


case1 = {"params": [{"shape": (64,768,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288,1024),"ori_format": "ND"},
                    {"shape": (64,768,16,16), "dtype": "float16", "format": "NCHW", "ori_shape": (12288,1024),"ori_format": "ND"},
                    {"shape": (12288,), "dtype": "float16", "format": "ND", "ori_shape": (12288,),"ori_format": "ND"},
                    {"shape": (12288,), "dtype": "float16", "format": "ND", "ori_shape": (12288,),"ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"},
                    {"shape": (64,768,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64,768,16,16),"ori_format": "ND"}],
         "case_name": "test_dsl_reduce_multi_lnxb_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True
         }


compile_case_list = [
    case1,
]

for item in compile_case_list:
    ut_case.add_case(case=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
