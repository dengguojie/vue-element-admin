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

AscendAntiQuant ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("AscendAntiQuant", None, None)

case1 = {"params": [{"shape": (5,2,8,16,16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (5, 8,16,16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (5,2,8,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (5, 8,16,16),"ori_format": "NC1HWC0"}, #h
                    1.2, 2.9, False,
                    ],
         "case_name": "AscendAntiQuant_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (30000, 26, 99, 32, 16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (30000, 26, 99, 32, 16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (30000, 26, 99, 32, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (30000, 26, 99, 32, 16),"ori_format": "NC1HWC0"}, #h
                    1.2, 2.9, False,
                    ],
         "case_name": "AscendAntiQuant_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1,1,2,16,16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,1,2,16,16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (1,1,2,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,2,16,16),"ori_format": "NC1HWC0"}, #h
                    1.2, 2.9, False,
                    ],
         "case_name": "AscendAntiQuant_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2,10,1028,1,16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (2,10,1028,1,16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (2,10,1028,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,10,1028,1,16),"ori_format": "NC1HWC0"},
                    1.2, 2.9, False,
                    ],
         "case_name": "AscendAntiQuant_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (2,16,16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (2,16,16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (2,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,16,16),"ori_format": "NC1HWC0"},
                    1.2, 2.9, False,
                    ],
         "case_name": "AscendAntiQuant_5",
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
