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
log_sigmoid_grad
"""
from op_test_frame.ut import OpUT
import numpy as np

#pylint: disable=unused-argument
def calc_expect_func(input_x, input_y, output_z):
    """
    Calculate log_sigmoid_grad of input_x using numpy.
    :param input_x: dict, contains value of input_x
    :param input_y: dict, contains value of input_y
    :param output_z: dict, not used in the function
    :return: lsit of numpy array, log_sigmoid_grad value of input_x, input_y
    """
    input_x = np.array(input_x.get("value"))
    input_y = np.array(input_y.get("value"))
    dtype = input_x.dtype
    if dtype != "float32":
        input_x = input_x.astype("float32")
        input_y = input_y.astype("float32")
    tmp = np.add(1.0, np.exp(input_y))
    tmp = 1.0 / tmp
    res = input_x * tmp
    if dtype != "float32":
        res = res.astype("float16")
    return [res, ]


def get_case(shape_list, dtype, expect):
    """
    Generate ut cases.
    :param shape_list: list of tupe, contains shape of inputs and outputs
    :param dtype: string, dtype of inputs and outputs
    :param expect: string, "success" for good cases, bad cases using
                   relevant error type like "RuntimeError"
    :return: dict of ut case
    """
    params = []
    for shape in shape_list:
        tmp = {"shape": shape, "ori_shape": shape, "format": "ND", "ori_format": "ND", "dtype": dtype}
        params.append(tmp)
    case = {"params": params, "expect": expect}
    return case


def get_precision_case(shape_list, dtype, data_range):
    """
    Generate ut precision cases.
    :param shape_list: list of tupe, contains shape of inputs and outputs
    :param dtype: string, dtype of inputs and outputs
    :param data_range: list, contains the minimum and maximum of generated data
    :return: dict of ut precision case
    """
    params = []
    i = 0
    for shape in shape_list:
        tmp = {"shape": shape, "ori_shape": shape, "format": "ND", "ori_format": "ND", "dtype": dtype,
               "param_type": "input" if i != 2 else "output", "value_range": data_range}
        params.append(tmp)
        i = i + 1
    params[-1].pop("value_range")
    case = {"params": params, "calc_expect_func": calc_expect_func}
    print(case)
    return case


cases = [
    [[(41,), (41,), (41,)], "float32", "success", [-10, 10]],
    [[(100, 23), (100, 23), (100, 23)], "float16", "success", [-10, 10]],
    [[(15, 20, 10, 15), (15, 20, 10, 15), (15, 20, 10, 15)], "float32", "success", [-10, 10]],
]

ut_case = OpUT("log_sigmoid_grad")
for item in cases:
    ut_case.add_case(support_soc="Ascend910A", case=get_case(item[0], item[1], item[2]))
    if item[2] == "success":
        ut_case.add_precision_case(support_soc="Ascend910A", case=get_precision_case(item[0], item[1], item[3]))
