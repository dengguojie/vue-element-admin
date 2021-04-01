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

MaskedFill ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("MaskedFill", None, None)


case1 = {"params": [{"shape": (8192, 1), "dtype": "float32", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"},
                    {"shape": (8192, 1), "dtype": "int8", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (8192, 1), "dtype": "float32", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (2, 1, 16), "dtype": "int32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "bool", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "int32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "int32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float32", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (2, 1, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "int8", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (2, 1, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 1, 16),"ori_format": "ND"}
                    ],
         "case_name": "masked_fill_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend910A"], case4)

if __name__ == '__main__':
    ut_case.run()
