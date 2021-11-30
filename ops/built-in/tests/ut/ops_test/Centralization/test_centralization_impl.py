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
ut for Centralization
"""
import os
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
ut_case = OpUT("Centralization", None, None)


# pylint: disable=unused-argument,invalid-name,too-many-locals
def calc_expect_func(x, y, dimension):
    """
    calc_expect_func
    """
    x_value = x.get("value")
    output_data = np.mean(x_value, axis=dimension[0], keepdims=True)
    for axis in dimension[1:]:
        output_data = np.mean(output_data, axis=axis, keepdims=True)
    result = x_value - output_data

    res = (result,)
    return res


def calc_expect_func_fz(x, y, dimension):
    """
    calc_expect_func_fz
    """
    x_shape = x.get("shape")
    x_ori_shape = x.get("ori_shape")
    input_ori_foramt = x.get("ori_format")
    dict_zip_shape = dict(zip(list(input_ori_foramt), x_ori_shape))
    input_ori_n = dict_zip_shape["N"]
    input_ori_c = dict_zip_shape["C"]
    input_ori_h = dict_zip_shape["H"]
    input_ori_w = dict_zip_shape["W"]
    x_value = x.get("value")
    if input_ori_c % x_shape[-1] != 0:
        input_shape_c1 = input_ori_c // x_shape[-1]
        x_value[input_shape_c1 * input_ori_h * input_ori_w:, :, :, input_ori_c % x_shape[-1]:] = 0
    output_data = np.sum(x_value, axis=0, keepdims=True)
    output_data = np.sum(output_data, axis=3, keepdims=True)
    output_data = output_data * np.array(1.0 / float(input_ori_c * input_ori_h * input_ori_w),
                                         output_data.dtype)
    result = x_value - output_data
    if input_ori_c % x_shape[-1] != 0:
        input_shape_c1 = input_ori_c // x_shape[-1]
        result[input_shape_c1 * input_ori_h * input_ori_w:, :, :, input_ori_c % x_shape[-1]:] = 0
    if input_ori_n % x_shape[-2] != 0:
        result[:, input_ori_n // x_shape[-2]:, input_ori_n % x_shape[-2]:, :] = 0

    res = (result,)
    return res


def calc_expect_func_nz_with_dim_zero(x, y, dimension):
    """
    calc_expect_func_nz_with_dim_zero
    """
    x_ori_shape = x.get("ori_shape")
    x_value = x.get("value")
    if x_ori_shape[0] % 16 != 0:
        x_value[:, x_ori_shape[0] // 16:, x_ori_shape[0] % 16:, :] = 0
    output_data = np.sum(x_value, axis=1, keepdims=True)
    output_data = np.sum(output_data, axis=2, keepdims=True)
    output_data = output_data * np.array(1.0 / float(x_ori_shape[0]), output_data.dtype)
    result = x_value - output_data
    if x_ori_shape[0] % 16 != 0:
        result[:, x_ori_shape[0] // 16:, x_ori_shape[0] % 16:, :] = 0
    # if x_ori_shape[1] % 16 != 0:
    #     result[x_ori_shape[1] // 16:, :, :, x_ori_shape[1] % 16:] = 0

    res = (result,)
    return res


def calc_expect_func_nz_with_dim_one(x, y, dimension):
    """
    calc_expect_func_nz_with_dim_one
    """
    x_ori_shape = x.get("ori_shape")
    x_value = x.get("value")
    if x_ori_shape[1] % 16 != 0:
        x_value[x_ori_shape[1] // 16:, :, :, x_ori_shape[1] % 16:] = 0
    output_data = np.sum(x_value, axis=0, keepdims=True)
    output_data = np.sum(output_data, axis=3, keepdims=True)
    output_data = output_data * np.array(1.0 / float(x_ori_shape[1]), output_data.dtype)
    result = x_value - output_data
    if x_ori_shape[1] % 16 != 0:
        result[x_ori_shape[1] // 16:, :, :, x_ori_shape[1] % 16:] = 0
    if x_ori_shape[0] % 16 != 0:
        result[:, x_ori_shape[0] // 16:, x_ori_shape[0] % 16:, :] = 0

    res = (result,)
    return res


case_1 = {
    "params": [
        {
            "shape": (7 * 7 * 16, 1, 16, 16),
            "dtype": "float32",
            "ori_shape": (16, 7, 7, 256),
            "ori_format": "NHWC",
            "format": "FRACTAL_Z",
            "param_type": "input"
        },
        {
            "shape": (7 * 7 * 16, 1, 16, 16),
            "dtype": "float32",
            "ori_shape": (16, 7, 7, 256),
            "ori_format": "NHWC",
            "format": "NHWC",
            "param_type": "output"
        },
        [1, 2, 3],
    ],
    "calc_expect_func": calc_expect_func_fz,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}

ut_case.add_precision_case("Ascend910A", case_1)

case_2 = {
    "params": [
        {
            "shape": (63, 2048 // 16, 16, 16),
            "dtype": "float32",
            "ori_shape": (2047, 1000),
            "ori_format": "NHWC",
            "format": "FRACTAL_NZ",
            "param_type": "input"
        },
        {
            "shape": (63, 2048 // 16, 16, 16),
            "dtype": "float32",
            "ori_shape": (2047, 1000),
            "ori_format": "NHWC",
            "format": "FRACTAL_NZ",
            "param_type": "output"
        },
        [0],
    ],
    "calc_expect_func": calc_expect_func_nz_with_dim_zero,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}

ut_case.add_precision_case("Ascend910A", case_2)

case_3 = {
    "params": [
        {
            "shape": (16, 16),
            "dtype": "float32",
            "ori_shape": (16, 16),
            "ori_format": "NHWC",
            "format": "NHWC",
            "param_type": "input"
        },
        {
            "shape": (16, 16),
            "dtype": "float32",
            "ori_shape": (16, 16),
            "ori_format": "NHWC",
            "format": "NHWC",
            "param_type": "output"
        },
        [0],
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}
ut_case.add_precision_case("Ascend910A", case_3)

case_4 = {
    "params": [
        {
            "shape": (7 * 7 * 16, 1, 16, 16),
            "dtype": "float16",
            "ori_shape": (16, 7, 7, 256),
            "ori_format": "NHWC",
            "format": "FRACTAL_Z",
            "param_type": "input"
        },
        {
            "shape": (7 * 7 * 16, 1, 16, 16),
            "dtype": "float16",
            "ori_shape": (16, 7, 7, 256),
            "ori_format": "NHWC",
            "format": "NHWC",
            "param_type": "output"
        },
        [1, 2, 3],
    ],
    "case_name": "FRACTAL_Z_fp16",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case_4)

case_5 = {
    "params": [
        {
            "shape": (128, 63, 16, 16),
            "dtype": "float32",
            "ori_shape": (1000, 2047),
            "ori_format": "NHWC",
            "format": "FRACTAL_NZ",
            "param_type": "input"
        },
        {
            "shape": (128, 63, 16, 16),
            "dtype": "float32",
            "ori_shape": (1000, 2047),
            "ori_format": "NHWC",
            "format": "FRACTAL_NZ",
            "param_type": "output"
        },
        [1],
    ],
    "calc_expect_func": calc_expect_func_nz_with_dim_one,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}

ut_case.add_precision_case("Ascend910A", case_5)

def test_op_select_format_001(test_arg):
    from impl.centralization import op_select_format
    op_select_format(
        {
            "shape": (16, 16, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (16, 16, 16, 16),
            "ori_format": "NHWC"
        }, {
            "shape": (16, 16, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (16, 16, 16, 16),
            "ori_format": "NHWC"
        }, [1])

ut_case.add_cust_test_func(test_func=test_op_select_format_001)

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, "/usr/local/Ascend/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
