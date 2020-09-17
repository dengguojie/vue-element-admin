# -*- coding:utf-8 -*-
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

NMSWithMask ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("NMSWithMask", "impl.nms_with_mask", "nms_with_mask")


case_small_shape_not_aligned = {
    "params":
        [
            {
                "shape": (6, 8),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (6, 8),
                "ori_format": "ND"
            },
            {
                "shape": (6, 5),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (6, 5),
                "ori_format": "ND"
            },
            {
                "shape": (6,),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (6,),
                "ori_format": "ND"
            },
            {
                "shape": (6,),
                "format": "ND",
                "dtype": "uint8",
                "ori_shape": (6,),
                "ori_format": "ND"
            },
            0.7
        ],
    "case_name": 'test_nms_with_mask_small_shape_not_aligned',
    "expect": "success"
}

case_big_shape_not_aligned = {
    "params":
        [
            {
                "shape": (2007, 8),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (2007, 8),
                "ori_format": "ND"
            },
            {
                "shape": (2007, 5),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (2007, 5),
                "ori_format": "ND"
            },
            {
                "shape": (2007,),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (2007,),
                "ori_format": "ND"
            },
            {
                "shape": (2007,),
                "format": "ND",
                "dtype": "uint8",
                "ori_shape": (2007,),
                "ori_format": "ND"
            },
            0.7
        ],
    "case_name": 'test_nms_with_mask_big_shape_not_aligned',
    "expect": "success"
}

case_aligned_with_iou_equal_one = {
    "params":
        [
            {
                "shape": (16, 8),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 8),
                "ori_format": "ND"
            },
            {
                "shape": (16, 5),
                "format": "ND",
                "dtype": "float16",
                "ori_shape": (16, 5),
                "ori_format": "ND"
            },
            {
                "shape": (16,),
                "format": "ND",
                "dtype": "int32",
                "ori_shape": (16,),
                "ori_format": "ND"
            },
            {
                "shape": (16,),
                "format": "ND",
                "dtype": "uint8",
                "ori_shape": (16,),
                "ori_format": "ND"
            },
            1.0
        ],
    "case_name": 'test_nms_with_mask_aligned_with_iou_equal_one',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_not_aligned)
ut_case.add_case(["Ascend910", "Ascend310"], case_big_shape_not_aligned)
ut_case.add_case(["Ascend910", "Ascend310"], case_aligned_with_iou_equal_one)

# ut_case.add_case(["Ascend310"], case1)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
