# Copyright 2019 Huawei Technologies Co., Ltd
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
arg_max_d ut testcase
"""

# pylint: disable=invalid-name,missing-function-docstring,unused-argument,import-error
import os
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from impl.arg_max_d import arg_max_d
from te import platform as cce_conf

ut_case = OpUT("ArgMaxD", None, None)

case1 = {
    "params": [
        {
            "shape": (31, 8190),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31, 8190),
            "ori_format": "ND"
        },
        {
            "shape": (31,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31,),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (31, 256),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31, 256),
            "ori_format": "ND"
        },
        {
            "shape": (31,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31,),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {
            "shape": (31, 127),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31, 127),
            "ori_format": "ND"
        },
        {
            "shape": (31,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31,),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [
        {
            "shape": (31, 24575),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31, 24575),
            "ori_format": "ND"
        },
        {
            "shape": (31,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (31,),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [
        {
            "shape": (31, 24575),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (31, 24575),
            "ori_format": "ND"
        },
        {
            "shape": (31,),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (31,),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_5",
    "expect": RuntimeError,
    "support_expect": True
}

case6 = {
    "params": [
        {
            "shape": (33, 11, 8191),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (33, 11, 8191),
            "ori_format": "ND"
        },
        {
            "shape": (33, 8191),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (33, 8191),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_6",
    "expect": RuntimeError,
    "support_expect": True
}

case7 = {
    "params": [
        {
            "shape": (33, 11, 8191),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (33, 11, 8191),
            "ori_format": "ND"
        },
        {
            "shape": (33, 8191),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (33, 8191),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_7",
    "expect": RuntimeError,
    "support_expect": True
}

case9 = {
    "params": [
        {
            "shape": (33, 11, 512),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (33, 11, 512),
            "ori_format": "ND"
        },
        {
            "shape": (33, 512),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (33, 512),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_9",
    "expect": RuntimeError,
    "support_expect": True
}

case10 = {
    "params": [
        {
            "shape": (3, 11, 810909),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (33, 11, 810909),
            "ori_format": "ND"
        },
        {
            "shape": (33, 810909),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (33, 810909),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_10",
    "expect": RuntimeError,
    "support_expect": True
}

case11 = {
    "params": [
        {
            "shape": (3, 11, 810909),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (33, 11, 810909),
            "ori_format": "ND"
        },
        {
            "shape": (33, 810909),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (33, 810909),
            "ori_format": "ND"
        },
        1,
    ],
    "case_name": "ArgMaxD_11",
    "expect": RuntimeError,
    "support_expect": True
}


def test_argmax_ascend920(test_arg):
    cce_conf.cce_conf.te_set_version("Ascend920A", core_type="VectorCore")
    arg_max_d({
        "shape": (31, 24575),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (31, 24575),
        "ori_format": "ND"
    }, {
        "shape": (31,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (31,),
        "ori_format": "ND"
    }, 1)
    cce_conf.cce_conf.te_set_version(test_arg)


ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case3)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case4)
ut_case.add_case(["Ascend910A", "Ascend710"], case5)
ut_case.add_case(["Ascend910A", "Ascend710"], case6)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case7)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case9)
ut_case.add_case(["Ascend910A", "Ascend710"], case10)
ut_case.add_case(["Ascend910A", "Ascend310", "Ascend710"], case11)
#ut_case.add_cust_test_func(test_func=test_argmax_ascend920)


def calc_expect_func(x, y, dimension):
    x_shape = x.get("shape")
    x_value = x.get("value")
    if dimension < 0:
        dimension = dimension + len(x_shape)
    output_data = np.argmax(x_value, axis=dimension)
    result = output_data.astype(np.int32)
    return (result,)


ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (1, 1),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (1, 1),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (1,),
                "dtype": "int32",
                "format": "ND",
                "ori_shape": (1,),
                "ori_format": "ND",
                "param_type": "output"
            },
            1,
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (2, 16, 32),
                "dtype": "float16",
                "format": "ND",
                "ori_shape": (2, 16, 32),
                "ori_format": "ND",
                "param_type": "input"
            },
            {
                "shape": (2, 16),
                "dtype": "int32",
                "format": "ND",
                "ori_shape": (2, 16),
                "ori_format": "ND",
                "param_type": "output"
            },
            2,
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
