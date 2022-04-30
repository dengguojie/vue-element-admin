#!/usr/bin/env python
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

Dynamic extract_image_patches ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("ExtractImagePatches", "impl.dynamic.extract_image_patches", "extract_image_patches")

case1 = {"params": [{"shape": (-1, 1, 8, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 8, 8, 2),"ori_format": "NHWC", "range": ((1, 8), (1, 1), (8, 8), (8, 8), (16, 16))},
                    {"shape": (-1, 1, 4, 4, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 4, 4, 12),"ori_format": "NHWC","range": ((1, 8), (1, 1), (4, 4), (4, 4), (16, 16))},
                    (1, 2, 2, 1),     # ksizes
                    (1, 2, 2, 1),     # strides
                    (1, 1, 1, 1),     # dilates
                    "VALID",          # padding
                    ],
         "expect": "success",
         "support_expect": True}
case2 = {"params": [{"shape": (-1, 1, 32, 32, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10000, 32, 32, 16),"ori_format": "NHWC", "range": ((1, 10000), (1, 1), (32, 32), (32, 32), (16, 16))},
                    {"shape": (-1, 1, 16, 16, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (10000, 16, 16, 64),"ori_format": "NHWC","range": ((1, 10000), (1, 1), (16, 16), (16, 16), (64, 64))},
                    (1, 2, 2, 1),     # ksizes
                    (1, 2, 2, 1),     # strides
                    (1, 1, 1, 1),     # dilates
                    "VALID",          # padding
                    ],
         "expect": "success",
         "support_expect": True}
case3 = {"params": [{"shape": (-1, 1, 128, 128, 48), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (128, 128, 128, 39),"ori_format": "NHWC", "range": ((1, 128), (1, 1), (128, 128), (128, 128), (48, 48))},
                    {"shape": (-1, 1, 1, 1, 39936), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 1, 1, 39936),"ori_format": "NHWC","range": ((1, 128), (1, 1), (1, 1), (1, 1), (39936, 39936))},
                    (1, 32, 32, 1),     # ksizes
                    (1, 32, 32, 1),     # strides
                    (1, 1, 1, 1),     # dilates
                    "VALID",          # padding
                    ],
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-1, 1, 128, 128, 48), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (128, 128, 128, 48),"ori_format": "NHWC", "range": ((1, 128), (1, 1), (128, 128), (128, 128), (48, 48))},
                    {"shape": (-1, 1, 1, 1, 49152), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 1, 1, 49152),"ori_format": "NHWC","range": ((1, 128), (1, 1), (1, 1), (1, 1), (49152, 49152))},
                    (1, 32, 32, 1),     # ksizes
                    (1, 32, 32, 1),     # strides
                    (1, 1, 1, 1),     # dilates
                    "VALID",          # padding
                    ],
         "expect": "success",
         "support_expect": True}

def test_generalization(args):
    from impl.dynamic.extract_image_patches import extract_image_patch_generalization
    extract_image_patch_generalization(
        {"shape": (-1, 1, 16, 16, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 16, 16, 2),"ori_format": "NHWC", "range": ((1, 2), (1, 1), (16, 16), (16, 16), (16, 16))},
        2,
        (1, 2, 2, 1),     # ksizes
        (1, 2, 2, 1),     # strides
        (1, 1, 1, 1),     # dilates
        (0, 0, 0, 0),          # padding
        {"shape": (-1, 1, 8, 8, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 8, 8, 8),"ori_format": "NHWC","range": ((1, 2), (1, 1), (8, 8), (8, 8), (16, 16))},
        )


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_cust_test_func(test_func=test_generalization)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")