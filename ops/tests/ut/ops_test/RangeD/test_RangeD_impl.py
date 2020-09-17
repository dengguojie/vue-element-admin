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

RangeD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("RangeD", None, None)

case1 = {"params": [{"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5,),"ori_format": "ND"}, #x
                    {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5,),"ori_format": "ND"},
                    1.1, 6.4, 2.2,
                    ],
         "case_name": "RangeD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (8192,), "dtype": "float32", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"}, #x
                    {"shape": (8192,), "dtype": "float32", "format": "ND", "ori_shape": (8192,),"ori_format": "ND"},
                    2.2, 8194.2, 1.0,
                    ],
         "case_name": "RangeD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (6,), "dtype": "int32", "format": "ND", "ori_shape": (6,),"ori_format": "ND"}, #x
                    {"shape": (6,), "dtype": "int32", "format": "ND", "ori_shape": (6,),"ori_format": "ND"},
                    5.0, -1.0, -1.0
                    ],
         "case_name": "RangeD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (2, 2), "dtype": "int32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"}, #x
                    {"shape": (2, 2), "dtype": "int32", "format": "ND", "ori_shape": (2, 2),"ori_format": "ND"},
                    10, 20, 2,
                    ],
         "case_name": "RangeD_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"}, #x
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    10, 20, 2,
                    ],
         "case_name": "RangeD_5",
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
