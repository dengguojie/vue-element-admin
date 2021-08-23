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

TransposeD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("TransposeD", "impl.transpose_d", "check_supported")

case1 = {"params": [{"shape": (2, 3, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 16),"ori_format": "ND"}, #x
                    {"shape": (2, 3, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (2, 3, 4, 16),"ori_format": "ND"},
                    (0, 2, 1, 3),
                    ],
         "case_name": "TransposeD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (3, 4, 5, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 5, 32),"ori_format": "ND"}, #x
                    {"shape": (3, 4, 5, 32), "dtype": "float32", "format": "ND", "ori_shape": (3, 4, 5, 32),"ori_format": "ND"},
                    (0, 2, 1, 3),
                    ],
         "case_name": "TransposeD_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (4032, 64), "dtype": "int32", "format": "ND", "ori_shape": (4032, 64),"ori_format": "ND"}, #x
                    {"shape": (4032, 64), "dtype": "int32", "format": "ND", "ori_shape": (4032, 64),"ori_format": "ND"},
                    (1, 0),
                    ],
         "case_name": "TransposeD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (-2,), "dtype": "float16", "format": "ND", "ori_shape": (-2,),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"},
                    (1, 0, 2),
                    ],
         "case_name": "TransposeD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (-1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 2, 3, 16),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"},
                    (1, 0, 5, 3),
                    ],
         "case_name": "TransposeD_5",
         "expect": "success",
         "support_expect": True}
case6 = {"params": [{"shape": (-1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (-1, 2, 3, 16),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 16), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"},
                    (1, 0, 5, 3),
                    ],
         "case_name": "TransposeD_6",
         "expect": "success",
         "support_expect": True}
case7 = {"params": [{"shape": (-1, 2, 3, 16), "dtype": "float32", "format": "ND", "ori_shape": (-1, 2, 3, 16),"ori_format": "ND"}, #x
                    {"shape": (1, 2, 3, 16), "dtype": "float32", "format": "ND", "ori_shape": (1, 2, 3, 16),"ori_format": "ND"},
                    (1, 0, 5, 3),
                    ],
         "case_name": "TransposeD_7",
         "expect": "success",
         "support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910A","Ascend310","Ascend710"], case7)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
