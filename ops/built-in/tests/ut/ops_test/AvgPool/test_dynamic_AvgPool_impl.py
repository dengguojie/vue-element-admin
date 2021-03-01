#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPool", "impl.dynamic.avg_pool", "avg_pool")

case1 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": "success",
         "support_expect": True}

# filter_h/w !=ksize_h/w
case2 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    [1,1,1,1], [1,1,1,1], "SAME", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# stride > 63 when filter is not None error
case3 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (1, 2), (1, 2)]},
                    [1,1,2,2], [1,1,64,64], "SAME", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# kernel_h/w > 21 error
case4 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (30, 100), (30, 100)]},
                    {"shape": (882,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 21, 21),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (30, 100), (30, 100)]},
                    [1,1,21,21], [1,1,1,1], "SAME", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# filter N != fmap Cin error
case5 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (30, 100), (30, 100)]},
                    {"shape": (16,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (64, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (30, 100), (30, 100)]},
                    [1,1,2,2], [1,1,1,1], "SAME", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# len(ksize)!=4 error
case6 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2,1], [1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# len(strides)!=4 error
case7 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# ksize_c != 1 error
case8 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,2,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# stride_c != 1 error
case9 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,2,2,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# padding error
case10 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALIDE", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# input_format error
case11 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "ND",
                      "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                     {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                      "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                     None,
                     {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                     [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
          "expect": RuntimeError,
          "support_expect": True}

# offsex_x!=0 error
case12 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                     {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                      "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                     None,
                     {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                     [1,1,2,2], [1,1,1,1], "VALIDE", "NCHW", 1],
          "expect": RuntimeError,
          "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)
ut_case.add_case(["Ascend910A"], case11)
ut_case.add_case(["Ascend910A"], case12)


if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)