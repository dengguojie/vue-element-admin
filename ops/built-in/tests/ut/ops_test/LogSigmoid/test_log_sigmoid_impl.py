# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
log_sigmoid
"""
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np
ut_case = BroadcastOpUT("log_sigmoid")

#pylint: disable=unused-argument
def calc_expect_func(input_x, output_y):
    """
    Calculate log_sigmoid of input_x using numpy.
    :param input_x: dict, contains value of input_x
    :param output_y: dict, not used in the function
    :return: lsit of numpy array, log_sigmoid value of input_x
    """
    input_x = np.array(input_x.get("value"))
    dtype = input_x.dtype
    # print(dtype)
    if dtype != "float32":
        input_x = input_x.astype("float32")
    tmp = np.add(1.0, np.exp(-1.0 * input_x))
    tmp = np.log(tmp)
    res = -1.0 * tmp
    if dtype != "float32":
        res = res.astype("float16")
    return [res, ]


ut_case.add_case("all", {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2), "shape": (2, 2),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_1",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("all", {
    "params": [{"dtype": "float64", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "float64", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_2",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_3",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("all", {
    "params": [{"dtype": "double", "format": "ND", "ori_format": "ND", "ori_shape": (20, 12), "shape": (20, 12),
                "param_type": "input"},
               {"dtype": "double", "format": "ND", "ori_format": "ND", "ori_shape": (20, 12), "shape": (20, 12),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_4",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100000, 1000000), "shape": (100000, 1000000),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100000, 1000000), "shape": (100000, 1000000),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_5",
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (41,), "shape": (41,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (41,), "shape": (41,),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_6",
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005),
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100, 23), "shape": (100, 23),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100, 23), "shape": (100, 23),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_7",
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005),
    "calc_expect_func": calc_expect_func,
    "expect": "success"
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (15, 20, 10, 15), "shape": (15, 20, 10, 15),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (15, 20, 10, 15), "shape": (15, 20, 10, 15),
                "param_type": "output"}],
    "case_name": "test_is_log_sigmoid_case_8",
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005),
    "calc_expect_func": calc_expect_func,
    "expect": "success"
})
