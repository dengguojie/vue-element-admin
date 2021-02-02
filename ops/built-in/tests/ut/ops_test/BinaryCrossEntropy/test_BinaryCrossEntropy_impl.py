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

BinaryCrossEntropy ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BinaryCrossEntropy", None, None)

case1 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape":(1,),"ori_format": "ND"},
                    "mean"
                    ],
         "case_name": "BinaryCrossEntroy_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape":(1,),"ori_format": "ND"},
                    "sum"
                    ],
         "case_name": "BinaryCrossEntropy_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    "none"
                    ],
         "case_name": "BinaryCrossEntropy_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape": (16, 8, 375),"ori_format": "ND"}, #x
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},
                    None,
                    {"shape": (16, 8, 375), "dtype": "float16", "format": "ND", "ori_shape":(16, 8, 375),"ori_format": "ND"},                   
                    "none"
                    ],
         "case_name": "BinaryCrossEntropy_4",
         "expect": "success",
         "support_expect": True}


def test_op_select_format(test_arg):
    from impl.binary_cross_entropy import op_select_format
    op_select_format({"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NCHW", "ori_shape": (128, 128, 128, 128), "ori_format": "NCHW","param_type": "input"},  # x
                     {},  # y
                     {},  # weight
                     {},  # output
                     "none")

    op_select_format({"shape": (128, 128, 128, 128, 128), "dtype": "float16", "format": "NDCHW", "ori_shape": (128, 128, 128, 128, 128), "ori_format": "NDCHW", "param_type": "input"},  # x
                     {},  # y
                     {},  # weight
                     {},  # output
                     "none")

    op_select_format({"shape": (1, 128, 128, 128, 128), "dtype": "float16", "format": "NDCHW", "ori_shape": (1, 128, 128, 128, 128), "ori_format": "NDCHW", "param_type": "input"},  # x
                     {},  # y
                     {},  # weight
                     {},  # output
                     "none")

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_cust_test_func(test_func=test_op_select_format)


if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
