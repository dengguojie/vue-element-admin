"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

CIoU ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("CIoU", "impl.ciou", "ciou")

case1 = {"params": [{"shape": (4, 1), "dtype": "float32", "format": "ND", "ori_shape": (4, 1), "ori_format": "ND"},
                    {"shape": (4, 1), "dtype": "float32", "format": "ND", "ori_shape": (4, 1), "ori_format": "ND"},
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                    False, False, "iou", True],
         "case_name": "ciou_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (4, 4), "dtype": "float32", "format": "ND", "ori_shape": (4, 4), "ori_format": "ND"},
                    {"shape": (4, 4), "dtype": "float32", "format": "ND", "ori_shape": (4, 4), "ori_format": "ND"},
                    {"shape": (1, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 4), "ori_format": "ND"},
                    {"shape": (1, 4), "dtype": "float32", "format": "ND", "ori_shape": (1, 4), "ori_format": "ND"},
                    False, False, "iou", True],
         "case_name": "ciou_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (4, 33), "dtype": "float32", "format": "ND", "ori_shape": (4, 33), "ori_format": "ND"},
                    {"shape": (4, 33), "dtype": "float32", "format": "ND", "ori_shape": (4, 33), "ori_format": "ND"},
                    {"shape": (1, 33), "dtype": "float32", "format": "ND", "ori_shape": (1, 33), "ori_format": "ND"},
                    {"shape": (1, 33), "dtype": "float32", "format": "ND", "ori_shape": (1, 33), "ori_format": "ND"},
                    False, False, "iou", True],
         "case_name": "ciou_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (4, 153600), "dtype": "float32", "format": "ND", "ori_shape": (4, 153600), "ori_format": "ND"},
                    {"shape": (4, 153600), "dtype": "float32", "format": "ND", "ori_shape": (4, 153600), "ori_format": "ND"},
                    {"shape": (1, 153600), "dtype": "float32", "format": "ND", "ori_shape": (1, 153600), "ori_format": "ND"},
                    {"shape": (1, 153600), "dtype": "float32", "format": "ND", "ori_shape": (1, 153600), "ori_format": "ND"},
                    False, False, "iou", True],
         "case_name": "ciou_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (4, 153600), "dtype": "float32", "format": "ND", "ori_shape": (4, 153600), "ori_format": "ND"},
                    {"shape": (4, 153600), "dtype": "float32", "format": "ND", "ori_shape": (4, 153600), "ori_format": "ND"},
                    {"shape": (1, 153600), "dtype": "float32", "format": "ND", "ori_shape": (1, 153600), "ori_format": "ND"},
                    {"shape": (1, 153600), "dtype": "float32", "format": "ND", "ori_shape": (1, 153600), "ori_format": "ND"},
                    True, False, "iou", True],
         "case_name": "ciou_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910")
