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

ApplyRmsPropD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("ApplyRmsPropD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (16, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (128, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (128, 16), "ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND",
                     "ori_shape": (128,), "ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND",
                     "ori_shape": (811, 12, 73, 5), "ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND",
                     "ori_shape": (811, 12, 73, 5), "ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND",
                     "ori_shape": (811, 12, 73, 5), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND",
                     "ori_shape": (811, 12, 73, 5), "ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND",
                     "ori_shape": (811, 12, 73, 5), "ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND",
                     "ori_shape": (811, 12, 73, 5), "ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND",
                     "ori_shape": (811, 12, 73, 5), "ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 1, 16, 96), "ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 1, 16, 96), "ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 1, 16, 96), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 1, 16, 96), "ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 1, 16, 96), "ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 1, 16, 96), "ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 1, 16, 96), "ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)


if __name__ == '__main__':
    ut_case.run("Ascend910")
