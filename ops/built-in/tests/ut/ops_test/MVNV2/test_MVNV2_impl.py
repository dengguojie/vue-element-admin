#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("Mvn_v2", None, None)


case1 = {"params": [{"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"},
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"}],
         "case_name": "mvn_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"},
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"},
                    1e-9, [1, 3]],
         "case_name": "mvn_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"},
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"},
                    1e-9, [0, 2]],
         "case_name": "mvn_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Hi3796CV300CS"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Hi3796CV300CS"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Hi3796CV300CS"], case3)


def test_lhisi_mvn_v2(test_args):
    from impl.mvn_v2 import mvn_v2
    from te import platform as cce_conf
    cce_conf.cce_conf.te_set_version("Hi3796CV300CS")
    mvn_v2({"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"},
           {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 2, 16, 16),"ori_format": "NCHW"},
           1e-9, [0, 2])
    cce_conf.cce_conf.te_set_version(test_args)
ut_case.add_cust_test_func(test_func=test_lhisi_mvn_v2)


def calc_expect_func(x, y, eps=1e-9, axis=None):
    if x['shape'] != y['shape']:
        return
    if not axis:
        axis = (0, 2, 3)

    shape_x = x['shape']

    num = 1
    for i in axis:
        num *= shape_x[i]
    if num != 0:
        num_rec = 1.0 / num

    mean = np.mean(x['value'], axis, keepdims=True)
    var = np.var(x['value'], axis, keepdims=True)
    sqrt = np.sqrt(var).astype(x['dtype'])
    sqrt_add = sqrt + eps   # according to caffe impl
    out = (x['value'] - mean) / sqrt_add

    return out


ut_case.add_precision_case("all", {"params": [{"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "input"},
                                              {"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "output"},
                                              ],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
                                   })
#ut_case.add_precision_case("all", {"params": [{"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "input"},
#                                              {"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "output"},
#                                              1e-9, (1, 3)],
#                                   "calc_expect_func": calc_expect_func,
#                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
#                                   })
#ut_case.add_precision_case("all", {"params": [{"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "input"},
#                                              {"shape": (2,4,16,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,4,16,16),"ori_format": "NCHW", "param_type": "output"},
#                                              1e-9, (0, 2)],
#                                   "calc_expect_func": calc_expect_func,
#                                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
#                                   })
#

if __name__ == '__main__':
    ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
    ut_case.run(["Hi3796CV300CS"], simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
