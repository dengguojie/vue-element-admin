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

Pack ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("Pack", None, None)

case1 = {"params": [[{"shape": (4,4,32), "dtype": "uint8", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}, #x
                     {"shape": (4,4,32), "dtype": "uint8", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}],
                     {"shape": (4,4,32), "dtype": "uint8", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    2,
                    ],
         "case_name": "Pack_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [[{"shape": (4,4,32), "dtype": "float16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}, #x
                    {"shape": (4,4,32), "dtype": "float16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}],
                    {"shape": (4,4,32), "dtype": "float16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    -2
                    ],
         "case_name": "Pack_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [[{"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}, #x
                    {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}],
                    {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    -2
                    ],
         "case_name": "Pack_3",
         "expect": "success",
         "support_expect": True}
         
case4 = {"params": [[{"shape": (1, 2, 3, 4, 5, 6, 7, 8), "dtype": "uint16", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 4, 5, 6, 7, 8), "dtype": "uint8", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8),"ori_format": "ND"}],
                    {"shape": (1, 2, 3, 4, 5, 6, 7, 8), "dtype": "uint8", "format": "ND", "ori_shape": (1, 2, 3, 4, 5, 6, 7, 8),"ori_format": "ND"},
                    5
                    ],
         "case_name": "Pack_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [[{"shape": (4,4,32), "dtype": "float16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}, #x
                    {"shape": (4,4,32), "dtype": "float16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}],
                    {"shape": (4,4,32), "dtype": "float16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    -10
                    ],
         "case_name": "Pack_5",
         "expect": RuntimeError,
         "support_expect": True}

case6 = {"params": [[{"shape": [], "dtype": "uint16", "format": "ND", "ori_shape": [],"ori_format": "ND"}, #x
                     {"shape": [], "dtype": "uint16", "format": "ND", "ori_shape": [],"ori_format": "ND"}],
                    {"shape": [2], "dtype": "uint16", "format": "ND", "ori_shape": [2],"ori_format": "ND"},
                    0
                    ],
         "case_name": "Pack_6",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [[{"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}, #x
                     {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}],
                    {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    3
                    ],
         "case_name": "Pack_7",
         "expect": "success",
         "support_expect": True}

case8 = {"params": [[{"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}, #x
                     {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}],
                    {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    0
                    ],
         "case_name": "Pack_8",
         "expect": "success",
         "support_expect": True}

case9 = {"params": [[{"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}, #x
                     {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"}],
                    {"shape": (4,4,32), "dtype": "uint16", "format": "ND", "ori_shape": (4,4,32),"ori_format": "ND"},
                    -4
                    ],
         "case_name": "Pack_9",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case8)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case9)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
