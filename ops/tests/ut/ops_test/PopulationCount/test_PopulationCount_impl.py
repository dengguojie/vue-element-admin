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
ut_case = OpUT("PopulationCount", None, None)

case1 = {"params": [{"shape": (1,3), "dtype": "int16", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"}, #x
                    {"shape": (1,3), "dtype": "uint8", "format": "ND", "ori_shape": (1,3),"ori_format": "ND"},
                    ],
         "case_name": "PopulationCount_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (2,2,3,40000), "dtype": "int16", "format": "ND", "ori_shape": (2,2,3,40000),"ori_format": "ND"}, #x
                    {"shape": (2,40,60,100), "dtype": "uint8", "format": "ND", "ori_shape": (2,40,60,100),"ori_format": "ND"},
                    ],
         "case_name": "PopulationCount_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (1,1,3,90000), "dtype": "uint16", "format": "ND", "ori_shape": (1,1,3,90000),"ori_format": "ND"}, #x
                    {"shape": (1,2,6,22500), "dtype": "uint8", "format": "ND", "ori_shape": (1,2,6,22500),"ori_format": "ND"},
                    ],
         "case_name": "PopulationCount_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1, 2, 3, 4, 5, 6, 7, 8), "dtype": "uint16", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 4, 5, 6, 7, 8), "dtype": "uint8", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8),"ori_format": "ND"},
                    ],
         "case_name": "PopulationCount_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (100, 150, 200, 300, 400, 500, 1024, 2048, 4096), "dtype": "uint16", "format": "ND", "ori_shape": (100, 150, 200, 300, 400, 500, 1024, 2048, 4096),"ori_format": "ND"}, #x
                    {"shape": (100, 150, 200, 300, 400, 500, 1024, 2048, 4096), "dtype": "uint8", "format": "ND", "ori_shape": (100, 150, 200, 300, 400, 500, 1024, 2048, 4096),"ori_format": "ND"},
                    ],
         "case_name": "PopulationCount_5",
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
