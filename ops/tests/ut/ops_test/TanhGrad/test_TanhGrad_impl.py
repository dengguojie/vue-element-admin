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

TanhGrad ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("TanhGrad", None, None)

case1 = {"params": [{"shape": (10,10241), "dtype": "float16", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"}, #x
                    {"shape": (10,10241), "dtype": "float16", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    {"shape": (10,10241), "dtype": "float16", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    ],
         "case_name": "TanhGrad_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (10,10241), "dtype": "float32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"}, #x
                    {"shape": (10,10241), "dtype": "float32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    {"shape": (10,10241), "dtype": "float32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    ],
         "case_name": "TanhGrad_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (10,10241), "dtype": "uint32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"}, #x
                    {"shape": (10,10241), "dtype": "uint32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    {"shape": (10,10241), "dtype": "uint32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    ],
         "case_name": "TanhGrad_3",
         "expect": RuntimeError,
         "support_expect": True}

case4 = {"params": [{"shape": (10,10241), "dtype": "int32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"}, #x
                    {"shape": (10,10241), "dtype": "int32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    {"shape": (10,10241), "dtype": "int32", "format": "ND", "ori_shape": (10,10241),"ori_format": "ND"},
                    ],
         "case_name": "TanhGrad_4",
         "expect": RuntimeError,
         "support_expect": True}

case5 = {"params": [{"shape": (2, 1, 2, 2, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 1, 2, 2, 16),"ori_format": "NC1HWC0"}, #x
                    {"shape": (2, 1, 2, 2, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 1, 2, 2, 16),"ori_format": "NC1HWC0"},
                    {"shape": (2, 1, 2, 2, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (2, 1, 2, 2, 16),"ori_format": "NC1HWC0"},
                    ],
         "case_name": "TanhGrad_5",
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
