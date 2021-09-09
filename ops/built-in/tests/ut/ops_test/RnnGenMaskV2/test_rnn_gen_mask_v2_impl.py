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
ut_case = OpUT("rnn_gen_mask_v2", "impl.dynamic.rnn_gen_mask_v2", "rnn_gen_mask_v2")

ut_case.add_case("all", {
    "params": [ {"shape": (-1,), "dtype": "int32",
                "format": "ND", "ori_shape": (32,),
                "ori_format": "ND", "range": [(32, 32)]},
               {"shape": (-1, 32, -1), "dtype": "float16",
                "format": "ND", "ori_shape": (2, 32, 64),
                "ori_format": "ND", "range": [(2, 2), (32, 32), (64, 64)]},
               {"shape": (-1, 32, -1), "dtype": "float16",
                "format": "ND", "ori_shape": (2, 32, 64),
                "ori_format": "ND", "range": [(2, 2), (32, 32), (64, 64)]}, 64],

    "case_name": "test_1",
    "expect": "success",
    "support_expect": True})
