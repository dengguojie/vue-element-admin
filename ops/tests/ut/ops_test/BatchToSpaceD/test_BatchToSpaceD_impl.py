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

BatchToSpaceD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BatchToSpaceD", None, None)

case1 = {"params": [{"shape": (288, 128, 6, 8, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 128, 6, 8, 16),"ori_format": "NHWC"}, #x
                    {"shape": (2, 128, 48, 72, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 128, 48, 72, 16),"ori_format": "NHWC"},
                    12,[[12, 12], [12, 12]],
                    ],
         "case_name": "BatchToSpaceD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (338, 128, 6, 8, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (338, 128, 6, 8, 16),"ori_format": "NHWC"}, #x
                    {"shape": (2, 128, 54, 80, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 128, 54, 80, 16),"ori_format": "NHWC"},
                    13,[[12, 12], [12, 12]],
                    ],
        "case_name": "BatchToSpaceD_2",
        "expect": "success",
        "support_expect": True}

case3 = {"params": [{"shape": (40,2,5,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape":(40,2,5,4,16),"ori_format": "NCHW"}, #x
                    {"shape": (10,2,6,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10,2,6,4,16),"ori_format": "NCHW"},
                    2,[[1, 3], [1, 3]],
                    ],
        "case_name": "BatchToSpaceD_3",
        "expect": "success",
        "support_expect": True}

case4 = {"params": [{"shape": (288, 2, 6, 4000, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 2, 6, 4000, 16),"ori_format": "NCHW"}, #x
                    {"shape": (288, 2, 6, 4000, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (288, 2, 6, 4000, 16),"ori_format": "NCHW"},
                    12, [[-1, -1], [-1, -1]],
                    ],
        "case_name": "BatchToSpaceD_4",
        "expect": RuntimeError,
        "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)


if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
