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

# kernel_h/w > 21
case4 = {"params": [{"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (30, 100), (30, 100)]},
                    {"shape": (441,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 1, 21, 21),"ori_format": "NCHW",
                     "range":[(16, 16), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 16, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (30, 100), (30, 100)]},
                    [1,1,21,21], [1,1,1,1], "SAME", "NCHW"],
         "expect": "success",
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

# filter format error
case13 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "ND",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# filter_c != 1 error
case14 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                     {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 2, 2, 2),"ori_format": "NCHW",
                      "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                     None,
                     {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                     [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
          "expect": RuntimeError,
          "support_expect": True}

# bias is not None error
case15 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                     {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                      "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                     {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                     {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                      "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                     [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
          "expect": RuntimeError,
          "support_expect": True}

# input_ori_format != data_format
case16 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,2,2,1], [1,1,1,1], "VALID", "NHWC"],
         "expect": RuntimeError,
         "support_expect": True}

case17 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (3, 100), (32, 32)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, -1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (2, 99), (2, 99), (32, 32)]},
                    [1,2,2,1], [1,1,1,1], "VALID", "NHWC"],
         "expect": "success",
         "support_expect": True}

case18 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-2,), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": "success",
         "support_expect": True}

case19 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": "success",
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
ut_case.add_case(["Ascend910A"], case13)
ut_case.add_case(["Ascend910A"], case14)
ut_case.add_case(["Ascend910A"], case15)
ut_case.add_case(["Ascend910A"], case16)
ut_case.add_case(["Ascend910A"], case17)
ut_case.add_case(["Ascend910A"], case18)
ut_case.add_case(["Ascend910A"], case19)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
