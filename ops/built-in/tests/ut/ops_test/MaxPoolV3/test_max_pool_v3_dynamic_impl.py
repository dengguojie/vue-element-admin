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

MaxPoolV3 ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolV3", "impl.dynamic.max_pool_v3", "max_pool_v3")

case1 = {"params": [{"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (56, 56), (56, 56), (16, 16)]}, #x
                    {"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (28, 28), (28, 28), (16, 16)]},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NC1HWC0",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (56, 56), (56, 56), (16, 16)]}, #x
                    {"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (28, 28), (28, 28), (16, 16)]},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NC1HWC0",
                    True,
                    False
                    ],
         "case_name": "MaxPoolV3_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (56, 56), (56, 56), (16, 16)]}, #x
                    {"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (28, 28), (28, 28), (16, 16)]},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NC1HWC0",
                    False,
                    True
                    ],
         "case_name": "MaxPoolV3_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (56, 56), (56, 56), (16, 16)]}, #x
                    {"shape": (-1, 4, 56, 56, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (4, 4), (28, 28), (28, 28), (16, 16)]},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "SAME",
                    (1, 1, 1, 1),
                    "NC1HWC0",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (64, 64), (56, 56), (56, 56)]}, #x
                    {"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NCHW", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NCHW", "range": [(1, 6), (64, 64), (28, 28), (28, 28)]},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NCHW",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (56, 56), (56, 56)]}, #x
                    {"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (28, 28), (28, 28)]},
                    [1, 1, 3],
                    [1, 1, 2, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NHWC",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (56, 56), (56, 56)]}, #x
                    {"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (28, 28), (28, 28)]},
                    [2, 1, 3, 3],
                    [1, 1, 2, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NHWC",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (56, 56), (56, 56)]}, #x
                    {"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (28, 28), (28, 28)]},
                    [2, 1, 3, 3],
                    [1, 1, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NHWC",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_8",
         "expect": RuntimeError,
         "support_expect": True}

case9 = {"params": [{"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (56, 56), (56, 56)]}, #x
                    {"shape": (-1, 64, 56, 56), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 64, 56, 56),
                     "ori_format": "NHWC", "range": [(1, 6), (64, 64), (28, 28), (28, 28)]},
                    [2, 1, 3, 3],
                    [1, 1, 2, 2],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NHWC",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_9",
         "expect": RuntimeError,
         "support_expect": True}

case10 = {"params": [{"shape": (-1, 56, 56, 64), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 56, 56, 64),
                     "ori_format": "NHWC", "range": [(1, 6), (56, 56), (56, 56), (64, 64)]}, #x
                    {"shape": (-1, 56, 56, 64), "dtype": "float16",
                     "format": "NHWC", "ori_shape": (1, 56, 56, 64),
                     "ori_format": "NHWC", "range": [(1, 6), (28, 28), (28, 28), (64, 64)]},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    "CALCULATED",
                    (1, 1, 1, 1),
                    "NHWC",
                    False,
                    False
                    ],
         "case_name": "MaxPoolV3_10",
         "expect": RuntimeError,
         "support_expect": True}


ut_case.add_case(["all"], case1)
ut_case.add_case(["all"], case2)
ut_case.add_case(["all"], case3)
ut_case.add_case(["all"], case4)
ut_case.add_case(["all"], case5)
ut_case.add_case(["all"], case6)
ut_case.add_case(["all"], case7)
ut_case.add_case(["all"], case8)
ut_case.add_case(["all"], case9)
ut_case.add_case(["all"], case10)

if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
