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
ut_case = OpUT("SpaceToBatchD", None, None)

case1 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"},
                    2, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "int16", "format": "NC1HWC0", "ori_shape": (2, 1, 2, 1),"ori_format": "NHWC"},
                    0, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_2",
         "expect": RuntimeError,
         "support_expect": True}

case3 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "uint8", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"},
                    21, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_3",
         "expect": RuntimeError,
         "support_expect": True}

case4 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "uint8", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"},
                    2, ((-1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatch_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"},
                    2, ((-1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_5",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [{"shape": (2, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"}, #x
                    {"shape": (2, 2, 2, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (2, 2, 2, 1),"ori_format": "NHWC"},
                    2, ((1, 1), (1, 1)),
                    ],
         "case_name": "SpaceToBatchD_6",
         "expect": RuntimeError,
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
