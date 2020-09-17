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

SliceD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("SliceD", None, None)

case1 = {"params": [{"shape": (5, 13, 4), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 13, 4),"ori_format": "NCHW"}, #x
                    {"shape": (2, 12, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (2, 12, 3),"ori_format": "NCHW"},
                    (0, 1, 1), (2, -1, -1),
                    ],
         "case_name": "SliceD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (65, 75), "dtype": "float32", "format": "NCHW", "ori_shape": (65, 75),"ori_format": "NCHW"}, #x
                    {"shape": (15, 33), "dtype": "float32", "format": "NCHW", "ori_shape": (15, 33),"ori_format": "NCHW"},
                    (13, 25), (15, 33),
                    ],
         "case_name": "SliceD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (13, 7, 5, 3), "dtype": "int8", "format": "NCHW", "ori_shape": (13, 7, 5, 3),"ori_format": "NCHW"}, #x
                    {"shape": (2, 4, 3, 1), "dtype": "int8", "format": "NCHW", "ori_shape": (2, 4, 3, 1),"ori_format": "NCHW"},
                    (0, 0, 0, 0), (2, 4, 3, 1),
                    ],
         "case_name": "SliceD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (160000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (160000, 16),"ori_format": "NCHW"}, #x
                    {"shape": (16000, 16), "dtype": "int64", "format": "NCHW", "ori_shape": (16000, 16),"ori_format": "NCHW"},
                    (0, 0), (160000, 16),
                    ],
         "case_name": "SliceD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (459999,), "dtype": "float32", "format": "NCHW", "ori_shape": (459999,),"ori_format": "NCHW"}, #x
                    {"shape": (458752,), "dtype": "float32", "format": "NCHW", "ori_shape": (458752,),"ori_format": "NCHW"},
                    (3,), (458752,), 
                    ],
         "case_name": "SliceD_5",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
