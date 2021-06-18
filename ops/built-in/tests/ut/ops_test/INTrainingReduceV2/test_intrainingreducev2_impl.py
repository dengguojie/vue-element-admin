# Copyright 2021 Huawei Technologies Co., Ltd
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
INTrainingReduceV2 ut testcase
"""

# pylint: disable=too-many-arguments,unused-variable,invalid-name,missing-function-docstring,unused-argument
import os
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("INTrainingReduceV2", "impl.in_training_reduce_v2", "in_training_reduce_v2")


def gen_in_training_reduce_v2_case(shape_x, shape_result, data_format, dtype, dtype_others, kernel_name, expect):
    return {
        "params": [{
            "shape":
                shape_x,
            "ori_shape": [shape_x[0], shape_x[1] *
                          shape_x[4], shape_x[2], shape_x[3]] if len(shape_x) == 5 else shape_x,
            "dtype":
                dtype,
            "format":
                data_format,
            "ori_format":
                "NCHW"
        }, {
            "shape":
                shape_result,
            "dtype":
                dtype_others,
            "ori_shape": [shape_result[0], shape_result[1] * shape_result[4], shape_result[2], shape_result[3]]
                         if len(shape_result) == 5 else shape_result,
            "format":
                data_format,
            "ori_format":
                "NCHW"
        }, {
            "shape":
                shape_result,
            "ori_shape": [shape_result[0], shape_result[1] * shape_result[4], shape_result[2], shape_result[3]]
                         if len(shape_result) == 5 else shape_result,
            "dtype":
                dtype_others,
            "format":
                data_format,
            "ori_format":
                "NCHW"
        }],
        "case_name": kernel_name,
        "expect": expect
    }


def calc_expect_func(input_arr, output_sum, square_sum):
    format_x = input_arr["format"]
    shape = input_arr["shape"]
    dtype_x = input_arr["dtype"]
    input_x = input_arr["value"]

    if format_x == "NC1HWC0":
        axis = [2, 3]
    else:
        idx_h = format_x.index("H")
        idx_w = format_x.index("W")
        axis = [idx_h, idx_w]
    axis = tuple(axis)

    if dtype_x == "float16":
        input_x = input_x.astype(np.float32)

    result_sum = np.sum(input_x, axis=axis, keepdims=True)
    square_sum = np.sum(input_x * input_x, axis=axis, keepdims=True)

    return result_sum, square_sum


def generate_precision_case(shape_x, shape_result, data_format, dtype, dtype_others, kernel_name, expect):
    ori_shape = [shape_x[0], shape_x[1] * shape_x[4], shape_x[2], shape_x[3]]
    return {
        "params": [{
            "shape": shape_x,
            "ori_shape": ori_shape if len(shape_x) == 5 else shape_x,
            "dtype": dtype,
            "format": data_format,
            "ori_format": "NCHW",
            "param_type": "input",
            "value_range": [-1.0, 1.0]
        }, {
            "shape": shape_result,
            "dtype": dtype_others,
            "ori_shape": ori_shape if len(shape_result) == 5 else shape_result,
            "format": data_format,
            "ori_format": "NCHW",
            "param_type": "output"
        }, {
            "shape": shape_result,
            "ori_shape": ori_shape if len(shape_result) == 5 else shape_result,
            "dtype": dtype_others,
            "format": data_format,
            "ori_format": "NCHW",
            "param_type": "output"
        }],
        "case_name": kernel_name,
        "expect": expect,
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    }

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_in_training_reduce_v2_case((6, 5, 8, 7, 16), (6, 5, 1, 1, 16), "NC1HWC0", "float32", "float32",
                                                "test_right_001", "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_in_training_reduce_v2_case((6, 5, 8, 7, 16), (6, 5, 1, 1, 16), "NC1HWC0", "float16", "float32",
                                                "test_right_002", "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_in_training_reduce_v2_case((6, 5, 8, 123, 16), (6, 5, 1, 1, 16), "NC1HWC0", "float16", "float32",
                                                "test_right_003", "success"))

ut_case.add_case(["Ascend910A", "Ascend710"],
                 gen_in_training_reduce_v2_case((6, 5, 8, 123, 16), (6, 5, 1, 1, 16), "NC1HWC0", "float32", "float32",
                                                "test_right_004", "success"))

ut_case.add_precision_case(["Ascend910A"],
                           generate_precision_case((6, 5, 8, 7, 16), (6, 5, 1, 1, 16), "NC1HWC0", "float32", "float32",
                                                   "in_training_reduce_v2_precision_001", "success"))

ut_case.add_precision_case(["Ascend910A"],
                           generate_precision_case((6, 5, 8, 7, 16), (6, 5, 1, 1, 16), "NC1HWC0", "float16", "float32",
                                                   "in_training_reduce_v2_precision_002", "success"))

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, "/usr/local/Ascend/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
