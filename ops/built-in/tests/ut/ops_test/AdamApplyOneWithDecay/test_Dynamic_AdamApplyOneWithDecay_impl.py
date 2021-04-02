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

AdamApplyOneWithDecay ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("AdamApplyOneWithDecay", "impl.dynamic.adam_apply_one_with_decay", "adam_apply_one_with_decay")


def get_input(shape, input_range, input_dtype):
    return {"shape": shape, "dtype": input_dtype, "format": "ND", \
        "ori_shape": shape, "ori_format": "ND", "range": input_range}

case1 = {"params": [get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32"),
                    get_input((-1, 8, 9), [(1, 200), (8, 8), (9, 9)], "float32")
                    ],
         "case_name": "AdamApplyOneWithDecay_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32")
                    ],
         "case_name": "AdamApplyOneWithDecay_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32"),
                    get_input((-1,), [(1, 200)], "float32")
                    ],
         "case_name": "AdamApplyOneWithDecay_3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
