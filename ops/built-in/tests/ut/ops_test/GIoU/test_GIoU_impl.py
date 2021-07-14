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

GIoU ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("GIoU", "impl.giou", "giou")

case1 = {"params": [{"shape": (1, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 4), "ori_format": "ND"},
                    {"shape": (1, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 4), "ori_format": "ND"},
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"}],
         "case_name": "giou_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16, 4), "dtype": "float32", "format": "ND", "ori_shape": (16, 4), "ori_format": "ND"},
                    {"shape": (16, 4), "dtype": "float32", "format": "ND", "ori_shape": (16, 4), "ori_format": "ND"},
                    {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"}],
         "case_name": "giou_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (33, 4), "dtype": "float32", "format": "ND", "ori_shape": (33, 4), "ori_format": "ND"},
                    {"shape": (33, 4), "dtype": "float32", "format": "ND", "ori_shape": (33, 4), "ori_format": "ND"},
                    {"shape": (33, 33), "dtype": "float32", "format": "ND", "ori_shape": (33, 33), "ori_format": "ND"}],
         "case_name": "giou_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (153600, 4), "dtype": "float32", "format": "ND", "ori_shape": (153600, 4), "ori_format": "ND"},
                    {"shape": (153600, 4), "dtype": "float32", "format": "ND", "ori_shape": (153600, 4), "ori_format": "ND"},
                    {"shape": (153600, 153600), "dtype": "float32", "format": "ND", "ori_shape": (153600, 153600), "ori_format": "ND"}],
         "case_name": "giou_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
