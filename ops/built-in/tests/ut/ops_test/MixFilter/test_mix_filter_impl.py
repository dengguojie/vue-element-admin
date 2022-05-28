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

MixFilter ut case
"""

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ut of mix_filter
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("MixFilter","impl.mix_filter", "mix_filter")

case1 = {"params": [{"shape": (65536,), "dtype": "float16", "format": "ND", "ori_shape": (65536,),
                     "ori_format": "ND"},
                    {"shape": (65536,), "dtype": "float16", "format": "ND",
                     "ori_shape": (65536,),"ori_format": "ND"},
                    {"shape": (65536,), "dtype": "float16", "format": "ND", "ori_shape": (65536,),
                     "ori_format": "ND"},
                    {"shape": (15,), "dtype": "float16", "format": "ND", "ori_shape": (15,),
                     "ori_format": "ND"},
                    {"shape": (65552,), "dtype": "float16", "format": "ND", "ori_shape": (65552,),
                     "ori_format": "ND"},
                    {"shape": (65552,), "dtype": "float16", "format": "ND", "ori_shape": (65552,),
                     "ori_format": "ND"},
                    {"shape": (65536,), "dtype": "float16", "format": "ND", "ori_shape": (65536,),
                     "ori_format": "ND"},
                    {"shape": (65536,), "dtype": "float16", "format": "ND", "ori_shape": (65536,),
                     "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),
                     "ori_format": "ND"}],
         "case_name": "mix_filter_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case("Ascend910A", case1)
if __name__ == '__main__':
    ut_case.run("Ascend910A")

