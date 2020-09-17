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

SpaceToBatch ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("StridedSliceD", None, None)

case1 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [0, 0, 0], [4, 8, 16], [3, 4, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 512, 32000), "dtype": "int32", "format": "ND", "ori_shape": (1, 512, 32000),"ori_format": "ND"}, #x
                    {"shape": (1, 512, 32000), "dtype": "int32", "format": "ND", "ori_shape": (1, 512, 32000),"ori_format": "ND"},
                    [0, 0, 0], [1, 512, 32000], [1, 16, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3,), "dtype": "int8", "format": "ND", "ori_shape": (3,),"ori_format": "ND"}, #x
                    {"shape": (3,), "dtype": "int8", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    [0], [2], [1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [0, 0, 0], [4, 8, 16], [3, 4, 1], 0, 0, 0, 0, 2
                    ],
         "case_name": "StridedSliceD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"}, #x
                    {"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    [0, 0, 0], [1, 512, 1024], [1, 512, 1], 5, 5, 0, 0, 0
                    ],
         "case_name": "StridedSliceD_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [7, 7, 0], [1, 1, 16], [-3, -1, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_6",
         "expect": RuntimeError,
         "support_expect": True}

case7 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"}, #x
                    {"shape": (8, 8, 16), "dtype": "float16", "format": "ND", "ori_shape": (8, 8, 16),"ori_format": "ND"},
                    [0, 0, 0], [4, 9, 16], [3, 4, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_7",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": [{"shape": (1, 512, 2048), "dtype": "int8", "format": "ND", "ori_shape": (1, 512, 2048),"ori_format": "ND"}, #x
                    {"shape": (1, 512, 2048), "dtype": "int8", "format": "ND", "ori_shape": (1, 512, 2048),"ori_format": "ND"},
                    [0, 0, 0], [1, 512, 2048], [1, 16, 1], 0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSliceD_8",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case8)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
