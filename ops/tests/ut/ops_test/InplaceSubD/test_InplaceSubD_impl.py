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

InplaceSubD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("InplaceSubD", None, None)

case1 = {"params": [{"shape": (2,4,32), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32),"ori_format": "NHWC"}, #x
                    {"shape": (2,4,32), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32),"ori_format": "NHWC"},
                    {"shape": (2,4,32), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32),"ori_format": "NHWC"},
                    [0,1],
                    ],
         "case_name": "InplaceSubD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2,4,32), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32),"ori_format": "NHWC"}, #x
                    {"shape": (2,4,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"},
                    {"shape": (2,4,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"},
                    [0,1,3],
                    ],
         "case_name": "InplaceSubD_2",
         "expect": RuntimeError,
         "support_expect": True}

case3 = {"params": [{"shape": (2,4,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"}, #x
                    {"shape": (2,4,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"},
                    {"shape": (2,4,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"},
                    [0,1,3],
                    ],
         "case_name": "InplaceSubD_3",
         "expect": RuntimeError,
         "support_expect": True}

case4 = {"params": [{"shape": (2,4,32,3), "dtype": "float32", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"}, #x
                    {"shape": (2,4,32,3), "dtype": "float32", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"},
                    {"shape": (2,4,32,3), "dtype": "float32", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"},
                    [0,1],
                    ],
         "case_name": "InplaceSubD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (2,4,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"}, #x
                    {"shape": (4,2,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (4,2,32,3),"ori_format": "NHWC"},
                    {"shape": (2,4,32,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,4,32,3),"ori_format": "NHWC"},
                    [0,1],
                    ],
         "case_name": "InplaceSubD_5",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
