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

BiasAddGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BiasAddGrad", None, None)

case1 = {"params": [{"shape": (2,3), "dtype": "float16", "format": "NHWC", "ori_shape": (2,3),"ori_format": "NHWC"}, #x
                    {"shape": (2,3), "dtype": "float16", "format": "NHWC", "ori_shape":(2,3),"ori_format": "NHWC"},
                    "NHWC"
                    ],
         "case_name": "BiasAddGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (77, 30522), "dtype": "float32", "format": "NHWC", "ori_shape": (77, 30522),"ori_format": "NHWC"}, #x
                    {"shape": (77, 30522), "dtype": "float32", "format": "NHWC", "ori_shape": (77, 30522),"ori_format": "NHWC"},
                    "NHWC"
                    ],
        "case_name": "BiasAddGrad_2",
        "expect": "success",
        "support_expect": True}

case3 = {"params": [{"shape": (10, 10, 10, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (10, 10, 10, 10),"ori_format": "NCHW"}, #x
                    {"shape": (10, 10, 10, 10), "dtype": "float16", "format": "NCHW", "ori_shape": (10, 10, 10, 10),"ori_format": "NCHW"},
                    "NCHW"
                    ],
        "case_name": "BiasAddGrad_3",
        "expect": "success",
        "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)


if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
