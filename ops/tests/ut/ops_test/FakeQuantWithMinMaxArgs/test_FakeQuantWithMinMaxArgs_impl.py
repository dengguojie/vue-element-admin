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

FakeQuantWithMinMaxArgs ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("FakeQuantWithMinMaxArgs", None, None)

case1 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "NHWC", "ori_shape": (128, 255, 36),"ori_format": "NHWC"}, #x
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NHWC", "ori_shape": (128, 255, 36),"ori_format": "NHWC"},
                    ],
         "case_name": "FakeQuantWithMinMaxArgs_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1024, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (1024, 1024),"ori_format": "NHWC"}, #x
                    {"shape": (1024, 1024), "dtype": "float32", "format": "NHWC", "ori_shape": (1024, 1024),"ori_format": "NHWC"},
                    -10.67,-5.55,6
                    ],
         "case_name": "FakeQuantWithMinMaxArgs_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (123,), "dtype": "float32", "format": "NCHW", "ori_shape": (123,),"ori_format": "NCHW"}, #x
                    {"shape": (123,), "dtype": "float32", "format": "NCHW", "ori_shape": (123,),"ori_format": "NCHW"},
                    7.778,30.123,9,True
                    ],
         "case_name": "FakeQuantWithMinMaxArgs_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (99991,), "dtype": "float32", "format": "NCHW", "ori_shape": (99991,),"ori_format": "NCHW"}, #x
                    {"shape": (99991,), "dtype": "float32", "format": "NCHW", "ori_shape": (99991,),"ori_format": "NCHW"},
                    -7,8,4,False
                    ],
         "case_name": "FakeQuantWithMinMaxArgs_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"}, #x
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "NCHW", "ori_shape": (128, 255, 36),"ori_format": "NCHW"},
                    -10.67,-5.55,6,True
                    ],
         "case_name": "FakeQuantWithMinMaxArgs_5",
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
    exit(0)
