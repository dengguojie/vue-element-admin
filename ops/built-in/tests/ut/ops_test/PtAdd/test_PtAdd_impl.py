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

PtAdd ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("PtAdd", None, None)


case1 = {"params": [{"shape": (8192, 1), "dtype": "float32", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"},
                    {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                    {"shape": (8192, 1), "dtype": "float32", "format": "ND", "ori_shape": (8192, 1),"ori_format": "ND"}],
         "case_name": "pt_add_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (10241,), "dtype": "float16", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float16", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "float16", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"}
                    ],
         "case_name": "pt_add_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (1024, 256, 128, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (1024, 256, 128, 16),"ori_format": "NHWC"},
                    {"shape": (1, ), "dtype": "float32", "format": "NHWC", "ori_shape": (1, ),"ori_format": "NHWC"},
                    {"shape": (1024, 256, 128, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (1024, 256, 128, 16),"ori_format": "NHWC"}
                    ],
         "case_name": "pt_add_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (10241,), "dtype": "int8", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"},
                    {"shape": (10, 10241), "dtype": "int8", "format": "NHWC", "ori_shape": (10, 10241),"ori_format": "NHWC"},
                    {"shape": (10241,), "dtype": "int8", "format": "NHWC", "ori_shape": (10241,),"ori_format": "NHWC"}
                    ],
         "case_name": "pt_add_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
def test_op_select_format(test_arg):
    from impl.pt_add import op_select_format
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"})
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"},
                     {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1),"ori_format": "ND"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"},
                     {"shape": (1, ), "dtype": "float16", "format": "NCHW", "ori_shape": (1, ),"ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (20, 28, 16, 16),"ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (20, 28, 3, 3, 16),"ori_format": "NC1HWC0"},
                     {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (20, 28, 3, 3, 16),"ori_format": "NC1HWC0"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20, 28, 16, 16),"ori_format": "FRACTAL_NZ"},
                     {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20, 28, 16, 16),"ori_format": "FRACTAL_NZ"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (20, 28, 16, 16),"ori_format": "FRACTAL_Z"},
                     {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (20, 28, 16, 16),"ori_format": "FRACTAL_Z"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (20, 28, 16, 16),"ori_format": "NHWC"},
                     {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1, ),"ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (20, 28, 16, 16),"ori_format": "NHWC"})

ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend910"], case4)
ut_case.add_cust_test_func(test_func=test_op_select_format)


if __name__ == '__main__':
    ut_case.run()
