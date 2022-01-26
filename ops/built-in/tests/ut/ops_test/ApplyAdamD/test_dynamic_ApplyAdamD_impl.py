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

ApplyAdamD ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("ApplyAdamD", "impl.dynamic.apply_adam_d", "apply_adam_d")


def get_input(shape, input_range, input_dtype):
    return {"shape": shape, "dtype":input_dtype, "format": "ND", "ori_shape": shape,"ori_format": "ND", "range": input_range}

case1 = {"params": [get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    True,True,
                    ],
         "case_name": "ApplyAdamD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    False,False,
                    ],
         "case_name": "ApplyAdamD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    get_input((-2, ), [(1, None)], "float32"),
                    False,False,
                    ],
         "case_name": "ApplyAdamD_3",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910A")