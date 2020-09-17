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

MinimumGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("MinimumGrad", None, None)

case1 = {"params": [{"shape": (2, 2), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 2),"ori_format": "NHWC"}, #x
                    {"shape": (2, 2), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 2),"ori_format": "NHWC"},
                    {"shape": (2, 2), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 2),"ori_format": "NHWC"},
                    {"shape": (2, 2), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 2),"ori_format": "NHWC"},
                    {"shape": (2, 2), "dtype": "int32", "format": "NHWC", "ori_shape": (2, 2),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "MinimumGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 96, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 64),"ori_format": "NHWC"}, #x
                    {"shape": (3, 96, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 64),"ori_format": "NHWC"}, #h
                    {"shape": (3, 96, 64), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 64),"ori_format": "NHWC"},
                    {"shape": (3, 96, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 96),"ori_format": "NHWC"},
                    {"shape": (3, 96, 96), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 96, 96),"ori_format": "NHWC"},
                    False,True,
                    ],
         "case_name": "MinimumGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"}, #x
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    {"shape": (3, 32, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (3, 32, 128),"ori_format": "NHWC"},
                    True,False,
                    ],
         "case_name": "MinimumGrad_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 5),"ori_format": "NHWC"}, #x
                    {"shape": (2, 2), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 5),"ori_format": "NHWC"},
                    {"shape": (2, 2), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 5),"ori_format": "NHWC"},
                    {"shape": (2, 2), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 5),"ori_format": "NHWC"},
                    {"shape": (2, 2), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 5),"ori_format": "NHWC"},
                    True,True
                    ],
         "case_name": "MinimumGrad_4",
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
