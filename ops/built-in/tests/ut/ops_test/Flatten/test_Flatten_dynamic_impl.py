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
test_Flatten_dynamic_impl.py
"""

# pylint: disable = invalid-name,
from op_test_frame.ut import OpUT

ut_case = OpUT("Flatten", "impl.dynamic.flatten", "flatten")


# pylint: disable = unused-argument,
def test_get_op_support_info(test_arg):
    from impl.dynamic.flatten import get_op_support_info
    x = {'ori_shape': (-1, -1), 'shape': (2, 1), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    y = {'ori_shape': (-1, -1), 'shape': (5, 3), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    get_op_support_info(x, y)


case1 = {
    "params": [{
        "shape": (255, 8, 33),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (255, 8, 33),
        "ori_format": "ND",
        "range": [(255, 255), (8, 8), (33, 33)]
    }, {
        "shape": (255, 264),
        "dtype": "float32",
        "format": "ND",
        "ori_shape": (255, 8, 33),
        "ori_format": "ND",
        "range": [(255, 255), (264, 264)]
    }, 1],
    "case_name": "Flatten_dynamic_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case2 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (255, 8, 33),
        "ori_format": "ND",
        "range": [(1, None),]
    }, {
        "shape": (-2,),
        "dtype": "float16",
        "format": "ND",
        "ori_shape": (255, 8, 33),
        "ori_format": "ND",
        "range": [(1, None),]
    }, 1],
    "case_name": "Flatten_dynamic_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case3 = {
    "params": [{
        "shape": (4, -1, 64),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (4, 16, 64),
        "ori_format": "ND",
        "range": [(4, 4), (1, 10), (1, 10)]
    }, {
        "shape": (4, -1),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (4, 16, 64),
        "ori_format": "ND",
        "range": [(4, 4), (1, 100)]
    }, 1],
    "case_name": "Flatten_dynamic_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case4 = {
    "params": [{
        "shape": (5, -1, 64),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5, 16, 64),
        "ori_format": "ND",
        "range": [(5, 5), (1, None), (1, 10)]
    }, {
        "shape": (5, -1),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5, 16, 64),
        "ori_format": "ND",
        "range": [(5, 5), (1, None)]
    }, 1],
    "case_name": "Flatten_dynamic_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case5 = {
    "params": [{
        "shape": (-1, -1, 64),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5, 16, 64),
        "ori_format": "ND",
        "range": [(1, 5), (1, None), (1, 10)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5, 16, 64),
        "ori_format": "ND",
        "range": [(1, 5), (1, None)]
    }, 1],
    "case_name": "Flatten_dynamic_5",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case6 = {
    "params": [{
        "shape": (5,),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5,),
        "ori_format": "ND",
        "range": [(5, 5),]
    }, {
        "shape": (5,),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5,),
        "ori_format": "ND",
        "range": [(5, 5),]
    }, 1],
    "case_name": "Flatten_dynamic_6",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case7 = {
    "params": [{
        "shape": (-1,),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5,),
        "ori_format": "ND",
        "range": [(1, 5),]
    }, {
        "shape": (-1,),
        "dtype": "int32",
        "format": "ND",
        "ori_shape": (5,),
        "ori_format": "ND",
        "range": [(1, 5),]
    }, 1],
    "case_name": "Flatten_dynamic_7",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
ut_case.add_cust_test_func(test_func=test_get_op_support_info)
ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
ut_case.add_case("all", case6)
ut_case.add_case("all", case7)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
