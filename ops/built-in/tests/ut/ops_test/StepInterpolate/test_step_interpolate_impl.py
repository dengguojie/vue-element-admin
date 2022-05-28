"""
Copyright (C) Huawei Technologies Co., Ltd 2022-2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

StepInterpolate ut case
"""

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ut of step_interpolate
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("StepInterpolate","impl.step_interpolate", "step_interpolate")

case1 = {"params": [{"shape": (65539,), "dtype": "float32", "format": "ND", "ori_shape": (65539,),
                     "ori_format": "ND"},
                    {"shape": (65539,), "dtype": "float32", "format": "ND",
                     "ori_shape": (65539,),"ori_format": "ND"},
                    {"shape": (2, 20480), "dtype": "float32", "format": "ND", "ori_shape": (2, 20480),
                     "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND"},
                    {"shape": (2, 20480), "dtype": "float32", "format": "ND", "ori_shape": (2, 20480),
                     "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND"}],
         "case_name": "step_interpolate_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case("Ascend910A", case1)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
