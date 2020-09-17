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

CumprodD ut case
"""
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

CumprodD ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("CumprodD", None, None)

case1 = {"params": [{"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"}, #x
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    ],
         "case_name": "CumprodD_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (1, 2), "dtype": "int8", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"}, #x
                    {"shape": (1, 2), "dtype": "int8", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                    True,
],
"case_name": "CumprodD_2",
"expect": "success",
"support_expect": True}

case3 = {"params": [{"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"}, #x
                    {"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    ],
         "case_name": "CumprodD_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": [{"shape": (1, 3), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"}, #x
                    {"shape": (1, 3), "dtype": "uint8", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                    ],
         "case_name": "CumprodD_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (15, 80, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"}, #x
                    {"shape": (15, 80, 2, 32), "dtype": "float16", "format": "ND", "ori_shape": (15, 80, 2, 32),"ori_format": "ND"},
                    2, True
],
"case_name": "CumprodD_5",
"expect": "success",
"support_expect": True}

case6 = {"params": [{"shape": (15, 80, 2, 76), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 76),"ori_format": "ND"}, #x
                    {"shape": (15, 80, 2, 76), "dtype": "float32", "format": "ND", "ori_shape": (15, 80, 2, 76),"ori_format": "ND"},
                    -1, True, True,
],
"case_name": "CumprodD_6",
"expect": "success",
"support_expect": True}

case7 = {"params": [{"shape": (15, 8, 50, 272), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 272),"ori_format": "ND"}, #x
                    {"shape": (15, 8, 50, 272), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 272),"ori_format": "ND"},
                    1, True, False,
],
"case_name": "CumprodD_7",
"expect": "success",
"support_expect": True}

case8 = {"params": [{"shape": (15, 8, 50, 271), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 271),"ori_format": "ND"}, #x
                    {"shape": (15, 8, 50, 271), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 271),"ori_format": "ND"},
                    1, False, False,
],
"case_name": "CumprodD_8",
"expect": "success",
"support_expect": True}

case9 = {"params": [{"shape": (15, 8, 50, 270), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 270),"ori_format": "ND"}, #x
                    {"shape": (15, 8, 50, 270), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 270),"ori_format": "ND"},
                    1, False, True,
],
"case_name": "CumprodD_9",
"expect": "success",
"support_expect": True}

case10 = {"params": [{"shape": (3, 8, 50, 273), "dtype": "float32", "format": "ND", "ori_shape": (3, 8, 50, 273),"ori_format": "ND"}, #x
                     {"shape": (15, 8, 50, 273), "dtype": "float32", "format": "ND", "ori_shape": (15, 8, 50, 273),"ori_format": "ND"},
                     1, True, True,
],
"case_name": "CumprodD_10",
"expect": "success",
"support_expect": True}

# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case7)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case8)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case9)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case10)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
