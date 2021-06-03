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

PopulationCount ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("PopulationCount", "impl.population_count", "get_op_support_info")

case1 = {"params": [{"shape": (1,3), "dtype": "int16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"}, #x
                    {"shape": (1,3), "dtype": "uint8", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    ],
         "case_name": "PopulationCount_1",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
