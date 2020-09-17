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

ExtractVolumePatches ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("ExtractVolumePatches")

case_small_shape_same_not_aligned_uint8 = {
    "params":
        [
            {
                "shape": (1, 6, 1, 6, 6, 32),
                "format": "NDC1HWC0",
                "dtype": "uint8",
                "ori_shape": (1, 6, 6, 6, 1),
                "ori_format": "NDHWC"
            },
            {
                "shape": (1, 3, 3, 3, 8),
                "format": "NDHWC",
                "dtype": "uint8",
                "ori_shape": (1, 3, 3, 3, 8),
                "ori_format": "NDHWC"
            },
            (1, 2, 2, 2, 1),
            (1, 2, 2, 2, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_small_shape_same_not_aligned_uint8',
    "expect": "success"
}

case_small_shape_same_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (1, 3, 2, 3, 3, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 3, 3, 3, 17),
                "ori_format": "NDHWC"
            },
            {
                "shape": (1, 3, 3, 3, 136),
                "format": "NDHWC",
                "dtype": "float16",
                "ori_shape": (1, 3, 3, 3, 136),
                "ori_format": "NDHWC"
            },
            (1, 2, 2, 2, 1),
            (1, 1, 1, 1, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_small_shape_same_not_aligned_fp16',
    "expect": "success"
}

case_small_shape_same_aligned_multi_batch_fp16 = {
    "params":
        [
            {
                "shape": (32, 2, 1, 2, 2, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (32, 2, 2, 2, 16),
                "ori_format": "NDHWC"
            },
            {
                "shape": (32, 2, 2, 2, 128),
                "format": "NDHWC",
                "dtype": "float16",
                "ori_shape": (32, 2, 2, 2, 128),
                "ori_format": "NDHWC"
            },
            (1, 2, 2, 2, 1),
            (1, 1, 1, 1, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_small_shape_same_aligned_multi_batch_fp16',
    "expect": "success"
}

case_medium_shape_same_howo_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (3, 101, 1, 6, 9, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (3, 101, 6, 9, 16),
                "ori_format": "NDHWC"
            },
            {
                "shape": (3, 51, 2, 5, 800),
                "format": "NDHWC",
                "dtype": "float16",
                "ori_shape": (3, 51, 2, 5, 800),
                "ori_format": "NDHWC"
            },
            (1, 2, 5, 5, 1),
            (1, 2, 3, 2, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_medium_shape_same_howo_not_aligned_fp16',
    "expect": "success"
}

case_big_shape_same_howo_aligned_fp16 = {
    "params":
        [
            {
                "shape": (1, 1, 1, 512, 512, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 1, 512, 512, 16),
                "ori_format": "NDHWC"
            },
            {
                "shape": (1, 1, 512, 512, 3456),
                "format": "NDHWC",
                "dtype": "float16",
                "ori_shape": (1, 1, 512, 512, 3456),
                "ori_format": "NDHWC"
            },
            (1, 6, 6, 6, 1),
            (1, 1, 1, 1, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_big_shape_same_howo_aligned_fp16',
    "expect": "success"
}

ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_same_not_aligned_uint8)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_same_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_same_aligned_multi_batch_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_medium_shape_same_howo_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_big_shape_same_howo_aligned_fp16)

# ut_case.add_case(["Ascend310"], case1)


if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
