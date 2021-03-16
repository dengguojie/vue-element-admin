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

TransposeD ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("MaxPoolWithArgmaxV1", "impl.max_pool_with_argmaxv1", "check_supported")

case1 = {"params": [
    {"shape": (16, 4, 120, 1200, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 4, 120, 1200, 16),
     "ori_format": "NHWC"},
    {"shape": (16, 4, 9, 2251, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 4, 9, 2251, 16),
     "ori_format": "NHWC"},
    {"shape": (16, 4, 60, 600, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (16, 4, 60, 600, 16),
     "ori_format": "NHWC"},
    [1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1], 3, [1, 1, 1, 1]],
    "case_name": "max_pool_with_argmax_v1_1",
    "expect": "success",
    "support_expect": True}

case2 = {"params": [
    {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),
     "ori_format": "NHWC"},
    {"shape": (1, 1, 169, 2, 16), "dtype": "float16", "format": "NC1HWC0",
     "ori_shape": (1, 1, 169, 2, 16), "ori_format": "NHWC"},
    {"shape": (1, 1, 1, 1, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (1, 1, 1, 1, 16),
     "ori_format": "NHWC"},
    [1, 56, 30, 1], [1, 1, 1, 1], [1, 6, 6, 1], 3, [1, 1, 1, 1]],
    "case_name": "max_pool_with_argmax_v1_2",
    "expect": "success",
    "support_expect": False}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
    exit(0)
