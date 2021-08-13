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

UnsortedSegmentSumD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("UnsortedSegmentSumD", None, None)

case1 = {"params": [{"shape": (160, 768), "dtype": "float16", "format": "ND", "ori_shape": (160, 768),"ori_format": "ND"}, #x
                    {"shape": (160,), "dtype": "float16", "format": "ND", "ori_shape": (160,),"ori_format": "ND"},
                    {"shape": (160, 768), "dtype": "float16", "format": "ND", "ori_shape": (160, 768),"ori_format": "ND"},
                    2,
                    ],
         "case_name": "UnsortedSegmentSumD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (7, 128 * 1400), "dtype": "float32", "format": "ND", "ori_shape": (7, 128 * 1400),"ori_format": "ND"}, #x
                    {"shape": (7,), "dtype": "float32", "format": "ND", "ori_shape": (7,),"ori_format": "ND"},
                    {"shape": (7, 128 * 1400), "dtype": "float32", "format": "ND", "ori_shape": (7, 128 * 1400),"ori_format": "ND"},
                    200,
                    ],
         "case_name": "UnsortedSegmentSumD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1280, 768), "dtype": "uint8", "format": "ND", "ori_shape": (1280, 768),"ori_format": "ND"}, #x
                    {"shape": (1280,), "dtype": "uint8", "format": "ND", "ori_shape": (1280,),"ori_format": "ND"},
                    {"shape": (1280, 768), "dtype": "uint8", "format": "ND", "ori_shape": (1280, 768),"ori_format": "ND"},
                    2,
                    ],
         "case_name": "UnsortedSegmentSumD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (3, 128, 70, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 128, 70, 2),"ori_format": "ND"}, #x
                    {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    {"shape": (3, 128, 70, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 128, 70, 2),"ori_format": "ND"},
                    200,
                    ],
         "case_name": "UnsortedSegmentSumD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (5, 17), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (5, 17),"ori_format": "NC1HWC0"}, #x
                    {"shape": (5,), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (5,),"ori_format": "NC1HWC0"},
                    {"shape": (5, 17), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (5, 17),"ori_format": "NC1HWC0"},
                    -6
                    ],
         "case_name": "UnsortedSegmentSumD_5",
         "expect": RuntimeError,
         "support_expect": True}
def test_op_select_format(test_arg):
    from impl.unsorted_segment_sum_d import op_select_format
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (28, 16, 16), "dtype": "int32", "format": "ND", "ori_shape": (28, 16, 16),"ori_format": "ND"},
                     {"shape": (200, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (200,28, 16, 16),"ori_format": "NCHW"},200)

    op_select_format({"shape": (25, 28, 16, 16), "dtype": "uint8", "format": "NCHW", "ori_shape": (25, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (25,), "dtype": "int32", "format": "ND", "ori_shape": (25,),"ori_format": "ND"},
                     {"shape": (300, 28, 16, 16), "dtype": "uint8", "format": "NCHW", "ori_shape": (300,28, 16, 16),"ori_format": "NCHW"},300)

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case5)
ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend710"])
    exit(0)
