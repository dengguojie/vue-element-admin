#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = ElementwiseOpUT("BNLL", "impl.bnll", "bnll")


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (512, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (100, 100))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (11, 33))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 12))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================

def calc_expect_func(input, output):
    data_a = input['value']
    input_shape = input['shape']
    s_type = input['dtype']
    data_reverse = np.where(data_a > 0, data_a * (-1), data_a)
    data_positive = np.where(data_a > 0, data_a, np.zeros(input_shape, s_type))

    exp_data = np.exp(data_reverse)
    log_data = np.log(np.add(exp_data, 1))
    in_tensor0 = log_data + data_positive
    return in_tensor0

precision_case1 = {"params": [{"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"output"}],
                   "case_name":"bnll_precision_1",
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (16,2,32), "dtype": "float16", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,2,32), "dtype": "float16", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"output"}],
                   "case_name":"bnll_precision_2",
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (2,1024), "dtype": "float16", "format": "ND", "ori_shape": (2,1024),"ori_format": "ND","param_type":"input"},
                              {"shape": (2,1024), "dtype": "float16", "format": "ND", "ori_shape": (2,1024),"ori_format": "ND","param_type":"output"}],
                   "case_name":"bnll_precision_3",
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case4 = {"params": [{"shape": (11,33), "dtype": "float16", "format": "ND", "ori_shape": (11,33),"ori_format": "ND","param_type":"input"},
                              {"shape": (11,33), "dtype": "float16", "format": "ND", "ori_shape": (11,33),"ori_format": "ND","param_type":"output"}],
                   "case_name":"bnll_precision_4",
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case5 = {"params": [{"shape": (10,12), "dtype": "float16", "format": "ND", "ori_shape": (10,12),"ori_format": "ND","param_type":"input"},
                              {"shape": (10,12), "dtype": "float16", "format": "ND", "ori_shape": (10,12),"ori_format": "ND","param_type":"output"}],
                   "case_name":"bnll_precision_5",
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}


ut_case.add_precision_case("Ascend910", precision_case1)
ut_case.add_precision_case("Ascend910", precision_case2)
ut_case.add_precision_case("Ascend910", precision_case3)
ut_case.add_precision_case("Ascend910", precision_case4)
ut_case.add_precision_case("Ascend910", precision_case5)


if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.76.T1.0.B010/toolkit/tools/simulator")


