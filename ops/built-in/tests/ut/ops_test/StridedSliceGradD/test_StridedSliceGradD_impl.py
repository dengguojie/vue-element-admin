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

StridedSliceGradD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("StridedSliceGradD", None, None)

case1 = {"params": [{"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND", "ori_shape": (80, 56, 8, 1023),"ori_format": "ND"}, #x
                    {"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND", "ori_shape": (80, 56, 8, 1023),"ori_format": "ND"},
                    (80, 23, 8, 443), (1, 33, 0, 342),
                    (56, 51, 3, 785), (1, 1, 1, 1), 5, 7, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 512, 1024), "dtype": "float32", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"}, #x
                    {"shape": (1, 512, 1024), "dtype": "float32", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 1, 1024), (0, 0, 0), (1, 1, 1024), (1, 1, 1),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND", "ori_shape": (80, 56, 8, 1023),"ori_format": "ND"}, #x
                    {"shape": (80, 56, 8, 1023), "dtype": "float16", "format": "ND", "ori_shape": (80, 56, 8, 1023),"ori_format": "ND"},
                    (80, 23, 8, 783), (1, 33, 0, -1021),
                    (56, -7, 3, -238), (1, 1, 1, 1), 5, 7, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1, 512, 1024), "dtype": "float32", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"}, #x
                    {"shape": (1, 512, 1024), "dtype": "float32", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 1, 1024), (0, 0, 0), (1, 1, 1024), (1, 1, 1),
                    0, 0, 1, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"}, #x
                    {"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (), (0, 0, 0), (1, 1, 1024), (1, 1, 1),0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_5",
         "expect": RuntimeError,
                               "support_expect": True}

case6 = {"params": [{"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"}, #x
                    {"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 1, 1024), (0, 0), (1, 1, 1024), (1, 1, 1),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_6",
         "expect": RuntimeError,
                               "support_expect": True}

case7 = {"params": [{"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"}, #x
                    {"shape": (1, 512, 1024), "dtype": "float16", "format": "ND", "ori_shape": (1, 512, 1024), "ori_format": "ND"},
                    (1, 1, 1024), (0, -100, 0), (1, -200, 1024), (1, 1, 1),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_7",
         "expect": RuntimeError,
                               "support_expect": True}

case8 = {"params": [{"shape": (1, 512, 2048), "dtype": "int8", "format": "ND", "ori_shape": (1, 512, 2048),"ori_format": "ND"}, #x
                    {"shape": (1, 512, 2048), "dtype": "int8", "format": "ND", "ori_shape": (1, 512, 2048),"ori_format": "ND"},
                    (1, 1, 1024), (0, 0, 0), (1, 1, 1024), (1, 1, 2),
                    0, 0, 0, 0, 0,
                    ],
         "case_name": "StridedSliceGradD_8",
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
