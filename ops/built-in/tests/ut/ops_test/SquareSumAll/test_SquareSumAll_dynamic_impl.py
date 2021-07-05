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
test_SquareSumAll_dynamic_impl.py
"""
# pylint: disable = invalid-name,
from op_test_frame.ut import OpUT

ut_case = OpUT("SquareSumAll", "impl.dynamic.square_sum_all", "square_sum_all")

case1 = {
    "params": [{
        "shape": (16, 16, 64, 32),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (16, 16, 64, 32),
        "ori_format": "ND"
    }, {
        "shape": (16, 16, 64, 32),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (16, 16, 64, 32),
        "ori_format": "ND"
    }, {
        "shape": (16, 16, 64, 32),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (16, 16, 64, 32),
        "ori_format": "ND"
    }, {
        "shape": (16, 16, 64, 32),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (16, 16, 64, 32),
        "ori_format": "ND"
    }],
    "case_name": "SquareSumAll_dynamic_1",
    "expect": "success"
}
case2 = {
    "params": [{
        "shape": (33,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (33, ),
        "ori_format": "ND",
    }, {
        "shape": (33,),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (33, ),
        "ori_format": "ND",
    }, {
        "shape": (33, ),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (33, ),
        "ori_format": "ND"
    }, {
        "shape": (33, ),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (33, ),
        "ori_format": "ND"
    }],
    "case_name": "SquareSumAll_dynamic_2",
    "expect": "success"
}
case3 = {
    "params": [{
        "shape": (9973, ),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (9973, ),
        "ori_format": "ND",
    }, {
        "shape": (9973, ),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (9973, ),
        "ori_format": "ND",
    }, {
        "shape": (9973, ),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (9973, ),
        "ori_format": "ND"
    }, {
        "shape": (9973, ),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (9973, ),
        "ori_format": "ND"
    }],
    "case_name": "SquareSumAll_dynamic_3",
    "expect": "success"
}
case4 = {
    "params": [{
        "shape": (9973, 13, 829),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (9973, 13, 829),
        "ori_format": "ND",
    }, {
        "shape": (9973, 13, 829),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (9973, 13, 829),
        "ori_format": "ND",
    }, {
        "shape": (9973, 13, 829),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (9973, 13, 829),
        "ori_format": "ND"
    }, {
        "shape": (9973, 13, 829),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (9973, 13, 829),
        "ori_format": "ND"
    }],
    "case_name": "SquareSumAll_dynamic_4",
    "expect": "success"
}
case5 = {
    "params": [{
        "shape": (32, 1913),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (32, 1913),
        "ori_format": "ND",
    }, {
        "shape": (32, 1913),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (32, 1913),
        "ori_format": "ND",
    }, {
        "shape": (32, 1913),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (32, 1913),
        "ori_format": "ND"
    }, {
        "shape": (32, 1913),
        "format": "ND",
        "dtype": "float32",
        "ori_shape": (32, 1913),
        "ori_format": "ND"
    }],
    "case_name": "SquareSumAll_dynamic_5",
    "expect": "success"
}
ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run(["Ascend910A"])
