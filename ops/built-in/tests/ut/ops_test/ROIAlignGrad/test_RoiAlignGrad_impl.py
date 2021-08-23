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

RoiAlignGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("RoiAlignGrad", None, None)

case1 = {"params": [{"shape": (2, 4, 16, 16, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 4, 16, 16, 16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (2, 5), "dtype": "float32", "format": "ND", "ori_shape": (2, 5),"ori_format": "ND"},
                    {"shape": (8, 2), "dtype": "int32", "format": "ND", "ori_shape": (8, 2),"ori_format": "ND"},
                    {"shape": (2, 4, 16, 16, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 4, 16, 16, 16),"ori_format": "NC1HWC0"},
                    [2,], 7, 7, 2.0, 0
                    ],
         "case_name": "RoiAlignGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 1, 36, 36, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 36, 36, 16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (200, 5), "dtype": "float32", "format": "ND", "ori_shape": (200, 5),"ori_format": "ND"},
                    {"shape": (7, 5), "dtype": "int32", "format": "ND", "ori_shape": (7, 5),"ori_format": "ND"},
                    {"shape": (1, 1, 36, 36, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 1, 36, 36, 16),"ori_format": "NC1HWC0"},
                    [2,], 7, 7, 2.0, 0
                    ],
         "case_name": "RoiAlignGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1, 1, 36, 36, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 36, 36, 16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (200, 5), "dtype": "float16", "format": "ND", "ori_shape": (200, 5),"ori_format": "ND"},
                    {"shape": (7, 5), "dtype": "int32", "format": "ND", "ori_shape": (7, 5),"ori_format": "ND"},
                    {"shape": (1, 1, 36, 36, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 1, 36, 36, 16),"ori_format": "NC1HWC0"},
                    [2,], 7, 7, 2.0, 0
                    ],
         "case_name": "RoiAlignGrad_3",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)


if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)
