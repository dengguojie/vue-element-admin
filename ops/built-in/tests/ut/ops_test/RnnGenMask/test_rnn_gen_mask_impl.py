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

Cast ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("rnn_gen_mask")

ut_case.add_case("all", {
    "params": [{"shape": (32,), "dtype": "int32", "format": "ND", "ori_shape": (32,),"ori_format": "ND",
                "param_type":"input"},
                {"shape": (6, 32, 16), "dtype": "float16", "format": "ND", "ori_shape": (6, 32, 16),"ori_format": "ND",
                "param_type":"outputput"}, 6, 16],
    "case_name": "test_1",
    "expect": "success",
    "support_expect": True})


ut_case.add_case("all", {
    "params": [{"shape": (32,), "dtype": "int32", "format": "ND", "ori_shape": (32,),"ori_format": "ND",
                "param_type":"input"},
                {"shape": (6, 32, 7), "dtype": "float16", "format": "ND", "ori_shape": (6, 32, 7),"ori_format": "ND",
                "param_type":"outputput"}, 6, 7],
    "case_name": "test_2",
    "expect": "success",
    "support_expect": True})


ut_case.add_case("all", {
    "params": [{"shape": (12,), "dtype": "int32", "format": "ND", "ori_shape": (12,),"ori_format": "ND",
                "param_type":"input"},
                {"shape": (6, 12, 7), "dtype": "float16", "format": "ND", "ori_shape": (6, 12, 7),"ori_format": "ND",
                "param_type":"outputput"}, 6, 7],
    "case_name": "test_3",
    "expect": "success",
    "support_expect": True})
