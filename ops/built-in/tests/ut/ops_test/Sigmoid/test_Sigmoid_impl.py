"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Sigmoid ut case
"""
import numpy as np
from op_test_frame.common import precision_info
# from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.ut import OpUT
from te import platform as cce_conf
from tbe.common.platform.platform_info import set_current_compile_soc_info
from impl.sigmoid import sigmoid
from unittest.mock import MagicMock
from unittest.mock import patch
from functools import wraps
# ut_case = ElementwiseOpUT("Sigmoid", None, None)
ut_case = OpUT("Sigmoid", "impl.sigmoid", "sigmoid")

# ============ auto gen ["Ascend910"] test cases start ===============
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1,))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (16, 2, 4, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (512, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (100, 100))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (11, 33))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 12))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================


def calc_expect_func(input_x, output_y):
    dtype = input_x["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    res = (1/(1+np.exp(-input_x["value"]))).astype(sdtype)
    return res


ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

# ut_case.add_precision_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9973, 1), "shape": (9973, 1), "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9973, 1), "shape": (9973, 1), "param_type": "output"}],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 2, 4, 32), "shape": (16, 2, 4, 32), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

# ut_case.add_precision_case("all", {
#     "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33), "param_type": "input"},
#                {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33),"param_type": "output"},
#                 "sigmoid",
#                 "high_performance"],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })

# print('run case for high_performance')
# ut_case.add_case(
#     'Ascend710',
#     {
#         'params': [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33)},
#                    {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33)}],
#         'addition_params': {'impl_mode': 'high_performance'},
#         'case_name': 'high_performance_case',
#         'expect': 'success',
#         'format_expect': [],
#         'support_expect': True,
#     },
# )


def sigmoid_test_high(test_arg):
    #patch('te.utils.para_check.check_op_params', mock_decorator).start()
    cce_conf.cce_conf.te_set_version("Ascend710")
    sigmoid({"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33)},
            {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 33), "shape": (11, 33)},
            "sigmoid",
            "high_performance")
    cce_conf.cce_conf.te_set_version(test_arg)


def mock_decorator(*args, **kwargs):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated_function
    return True


ut_case.add_cust_test_func(test_func=sigmoid_test_high)
if __name__ == '__main__':
    patch('te.utils.para_check.check_op_params', mock_decorator).start()
    ut_case.run('Ascend710')
    exit(0)
