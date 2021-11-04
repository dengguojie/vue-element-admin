#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = BroadcastOpUT("Div", None, None)


# ============ auto gen ["Ascebd910A"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32", "int32", "int8", "uint8"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascebd910A"], ["float16", "float32", "int32", "int8", "uint8"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascebd910A"], ["float16", "float32", "int32", "int8", "uint8"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascebd910A"] test cases end =================
case1 = {"params": [{"shape": (1, 16, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 16, 16, 16), "ori_format": "FRACTAL_NZ"},
                    {"shape": (1, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 16), "ori_format": "ND"},
                    {"shape": (1, 16, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 16, 16, 16), "ori_format": "FRACTAL_NZ"}],
         "case_name": "mul_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (1, 16), "ori_format": "ND"},
                    {"shape": (1, 16, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 16, 16, 16), "ori_format": "FRACTAL_NZ"},
                    {"shape": (1, 16, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ",
                     "ori_shape": (1, 16, 16, 16), "ori_format": "FRACTAL_NZ"}],
         "case_name": "mul_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

def calc_expect_func(x1, x2, y):
    res = x1['value'] / x2['value']
    res = res.astype(y['dtype'])
    return res

ut_case.add_precision_case("Ascebd910A", {"params": [{"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "output"}],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascebd910A", {"params": [{"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "output"}],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascebd910A", {"params": [{"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "output"}],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascebd910A", {"params": [{"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "output"}],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascebd910A", {"params": [{"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
                                              {"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "output"}],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
            
# pylint: disable=unused-argument
def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.div import op_select_format
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NDHWC",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NDHWC"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NCHW",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NCHW"})
    op_select_format({"shape": (20, 28, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (20, 28, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (20, 28, 16), "ori_format": "ND"})
    op_select_format({"shape": (1, 1, 1), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (1, 1, 1), "ori_format": "NHWC"},
                     {"shape": (96, 1, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (96, 56, 56, 8), "ori_format": "NHWC"},
                     {"shape": (96, 1, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (96, 56, 56, 8), "ori_format": "NHWC"})
    op_select_format({"shape": (), "dtype": "float32", "format": "NHWC",
                      "ori_shape": (), "ori_format": "NHWC"},
                     {"shape": (96, 256), "dtype": "float32", "format": "FRACTAL_NZ",
                      "ori_shape": (96, 256), "ori_format": "NHWC"},
                     {"shape": (96, 256), "dtype": "float32", "format": "FRACTAL_NZ",
                      "ori_shape": (96, 256), "ori_format": "NHWC"})
    op_select_format({"shape": (25, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z",
                      "ori_shape": (6, 1, 5, 5), "ori_format": "NCHW"},
                     {"shape": (), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (), "ori_format": "NCHW"},
                     {"shape": (25, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z",
                      "ori_shape": (6, 1, 5, 5), "ori_format": "NCHW"})
    op_select_format({"shape": (512,), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (512,), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"})
    op_select_format({"shape": (33, 17, 3, 5, 3), "dtype": "float16", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float16", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "float32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "float16", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "int32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "int32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "int32", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int32", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "int32", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "uint8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "uint8", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"},
                     {"shape": (33, 17, 3, 5, 3), "dtype": "uint8", "format": "ND",
                      "ori_shape": (33, 17, 3, 5, 3), "ori_format": "ND"})
    op_select_format({"shape": (16, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (16, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (3, 32, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (3, 32, 32), "ori_format": "ND"},
                     {"shape": (3, 32, 32), "dtype": "float16", "format": "ND",
                      "ori_shape": (3, 32, 32), "ori_format": "ND"})
    op_select_format({"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (1,), "ori_format": "NCHW"},
                     {"shape": (16, 16, 512, 512), "dtype": "float32", "format": "NCHW",
                      "ori_shape": (16, 16, 512, 512), "ori_format": "NCHW"})
    op_select_format({"shape": (-1, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (-1, 32, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int8", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (-1, 32, 16), "dtype": "int8", "format": "ND",
                      "ori_shape": (-1, 32, 16), "ori_format": "ND"})
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NZ",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NZ"},
                     {"shape": (1,), "dtype": "float16", "format": "ND",
                      "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "NZ",
                      "ori_shape": (20, 28, 16, 16), "ori_format": "NZ"})

ut_case.add_cust_test_func(test_func=test_op_select_format)
if __name__ == '__main__':
    ut_case.run("Ascebd910A")  


