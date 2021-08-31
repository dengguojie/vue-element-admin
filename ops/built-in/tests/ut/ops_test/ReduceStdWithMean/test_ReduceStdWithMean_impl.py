"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dot ut case
"""
# # -*- coding:utf-8 -*-
import sys
import numpy as np
import torch
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("reduce_std_with_mean")

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1), "shape": (3, 1),
                "param_type": "output"},
               [1,],
               True,
               True],
    "case_name": "test_reduce_std_with_mean_case_1"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               [2,],
               True,
               False],
    "case_name": "test_reduce_std_with_mean_case_2",
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               [2,],
               False,
               False],
    "case_name": "test_reduce_std_with_mean_case_3",
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1), "shape": (3, 1),
                "param_type": "output"},
               [1,],
               True,
               True,
               True,
               0.001],
    "case_name": "test_reduce_std_with_mean_case_4"
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               [2,],
               True,
               False,
               True,
               0.001],
    "case_name": "test_reduce_std_with_mean_case_5",
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5), "shape": (3, 4, 5),
                "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output"},
               [2,],
               False,
               False,
               True,
               0.001],
    "case_name": "test_reduce_std_with_mean_case_6",
})
