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

FakeQuantWithMinMaxVarsPerChannel ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("FakeQuantWithMinMaxVarsPerChannel", None, None)

case1 = {"params": [{"shape": (2, 2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 2),"ori_format": "ND"}, #x
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2, 2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 2),"ori_format": "ND"},
                    8,True,
                    ],
         "case_name": "FakeQuantWithMinMaxVarsPerChannel_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2, 2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 2),"ori_format": "ND"}, #x
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2,), "dtype": "float32", "format": "ND", "ori_shape": (2,),"ori_format": "ND"},
                    {"shape": (2, 2, 2), "dtype": "float32", "format": "ND", "ori_shape": (2, 2, 2),"ori_format": "ND"},
                    16,False,
                    ],
         "case_name": "FakeQuantWithMinMaxVarsPerChannel_2",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend310P3"], case2)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend310P3"])
    exit(0)
