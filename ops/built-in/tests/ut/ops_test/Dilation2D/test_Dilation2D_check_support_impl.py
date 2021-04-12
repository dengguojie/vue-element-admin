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
test_Dilation2D_check_support_impl.py
"""
# pylint: disable=unused-argument,unused-import,wrong-import-order,dangerous-default-value,too-many-arguments
# pylint: disable=invalid-name,superfluous-parens,missing-function-docstring,too-many-locals,pointless-string-statement
from op_test_frame.ut import OpUT

ut_case = OpUT("Dilation2D", "impl.dilation2d", "check_supported")

case1 = {
    "params": [{
        "shape": (32, 5, 5, 16),
        "dtype": "float32",
        "format": "NHWC",
        "ori_shape": (32, 5, 5, 16),
        "ori_format": "NHWC"
    }, {
        "shape": (3, 3, 16),
        "dtype": "float32",
        "format": "NHWC",
        "ori_shape": (3, 3, 16),
        "ori_format": "NHWC"
    }, {
        "shape": (32, 3, 3, 16),
        "dtype": "float32",
        "format": "NHWC",
        "ori_shape": (32, 3, 3, 16),
        "ori_format": "NHWC"
    }, [1, 1, 1, 1], [1, 1, 1, 1], "VALID", [0, 0, 0, 0], False, "NHWC"],
    "case_name": "test_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{
        "shape": (32, 16, 5, 5),
        "dtype": "float32",
        "format": "NCHW",
        "ori_shape": (32, 16, 5, 5),
        "ori_format": "NCHW"
    }, {
        "shape": (16, 3, 3),
        "dtype": "float32",
        "format": "NCHW",
        "ori_shape": (16, 3, 3),
        "ori_format": "NCHW"
    }, {
        "shape": (32, 16, 3, 3),
        "dtype": "float32",
        "format": "NCHW",
        "ori_shape": (32, 16, 3, 3),
        "ori_format": "NCHW"
    }, [1, 1, 1, 1], [1, 1, 1, 1], "VALID", [0, 0, 0, 0], False, "NHWC"],
    "case_name": "test_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{
        "shape": (32, 1, 5, 5, 16),
        "dtype": "float32",
        "format": "NC1HWC0",
        "ori_shape": (32, 1, 5, 5, 16),
        "ori_format": "NC1HWC0"
    }, {
        "shape": (1, 1, 3, 3, 16),
        "dtype": "float32",
        "format": "NC1HWC0",
        "ori_shape": (1, 1, 3, 3, 16),
        "ori_format": "NC1HWC0"
    }, {
        "shape": (32, 1, 3, 3, 16),
        "dtype": "float32",
        "format": "NC1HWC0",
        "ori_shape": (32, 1, 3, 3, 16),
        "ori_format": "NC1HWC0"
    }, [1, 1, 1, 1], [1, 1, 1, 1], "VALID", [0, 0, 0, 0], False, "NHWC"],
    "case_name": "test_3",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
