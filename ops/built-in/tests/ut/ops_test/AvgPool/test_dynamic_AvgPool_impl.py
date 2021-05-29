#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPool", "impl.dynamic.avg_pool", "avg_pool")

# dynamic nw SAME NCHW range None
case1 = {"params": [{"shape": (-1,2,1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,32,1,-1), "ori_format": "NCHW",
                     "range":[(1, None), (32, 32), (1, 100), (1, None)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (-1, 2, 1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, 32, 1, -1),"ori_format": "NCHW",
                     "range":[(1, None), (32, 32), (1, 99), (1, None)]},
                    [1,1,2,2], [1,1,2,2], "SAME", "NCHW"],
         "expect": "success",
         "support_expect": True}

# kernel_h/w > 21
case4 = {"params": [{"shape": (-1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,16,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (30, 100), (30, 100)]},
                    {"shape": (441,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 1, 21, 21),"ori_format": "NCHW",
                     "range":[(16, 16), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (-1, 1, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, 16, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (16, 16), (30, 100), (30, 100)]},
                    [1,1,21,21], [1,1,1,1], "SAME", "NCHW"],
         "expect": "success",
         "support_expect": True}

# dynamic hw VALID NHWC
case17 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (3, 100), (32, 32)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, -1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (2, 99), (2, 99), (32, 32)]},
                    [1,2,2,1], [1,1,2,1], "VALID", "NHWC"],
         "expect": "success",
         "support_expect": True}

# dynamic n SAME NCHW
case33 = {"params": [{"shape": (-1,32,1,1), "format": "NCHW", "dtype": "float16", "ori_shape": (-1,32,1,1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (1, 100), (1, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (-1, 32, 1, 1),"format": "NCHW", "dtype": "float16", "ori_shape": (-1, 32, 1, 1),"ori_format": "NCHW",
                     "range":[(1, 10), (32, 32), (1, 99), (1, 99)]},
                    [1,1,2,2], [1,1,2,1], "SAME", "NCHW"],
         "expect": "success",
         "support_expect": True}

# dynamic h VALID NHWC 
case34 = {"params": [{"shape": (1,-1,2,32), "format": "NHWC", "dtype": "float16", "ori_shape": (1,-1,2,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (2, 100), (32, 32)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, -1, 1, 32),"format": "NHWC", "dtype": "float16", "ori_shape": (1, -1, 1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (1, 99), (1, 99), (32, 32)]},
                    [1,2,2,1], [1,2,1,1], "VALID", "NHWC"],
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

# unknown rank error
case18 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-2,), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

# dynamic c error
case19 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

case20 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,-1,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}

case21 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [3,1,-1,2,1], [3,1,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case22 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [3,1,-1,2], [3,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case23 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [3,1,-1,2], [3,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case24 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,-1,2], [3,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case25 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,2,-1,2], [1,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case26 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [3,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case27 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,-1,2], [3,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case28 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,2,-1,2], [1,1,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}

case29 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,-1,2], [1,2,1,1], "FRACTAL_Z", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}
case30 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,-1,2], [1,1,1,1], "VALID", "FRACTAL_Z"],
         "expect": RuntimeError,
         "support_expect": True}
case31 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW", 1],
         "expect": RuntimeError,
         "support_expect": True}
case32 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "FRACTAL_Z",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,1,1], "VALID", "NCHW"],
         "expect": RuntimeError,
         "support_expect": True}
#stide > 63
case35 = {"params": [{"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,32,-1,-1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (3, 100), (3, 100)]},
                    None,
                    None,
                    {"shape": (1, 2, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 32, -1, -1),"ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (2, 99), (2, 99)]},
                    [1,1,2,2], [1,1,64,64], "VALID", "NCHW"],
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
ut_case.add_case(["Ascend910A"], case20)
ut_case.add_case(["Ascend910A"], case21)
ut_case.add_case(["Ascend910A"], case22)
ut_case.add_case(["Ascend910A"], case23)
ut_case.add_case(["Ascend910A"], case24)
ut_case.add_case(["Ascend910A"], case25)
ut_case.add_case(["Ascend910A"], case26)
ut_case.add_case(["Ascend910A"], case27)
ut_case.add_case(["Ascend910A"], case28)
ut_case.add_case(["Ascend910A"], case29)
ut_case.add_case(["Ascend910A"], case30)
ut_case.add_case(["Ascend910A"], case31)
ut_case.add_case(["Ascend910A"], case32)
ut_case.add_case(["Ascend910A"], case33)
ut_case.add_case(["Ascend910A"], case34)
ut_case.add_case(["Ascend910A"], case35)

from impl.dynamic.avg_pool import check_supported, get_op_support_info

def test_check_support(test_arg):
    check_supported({"shape": (-1,2,1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,32,1,-1), "ori_format": "NCHW",
                     "range":[(1, None), (32, 32), (1, 100), (1, None)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (-1, 2, 1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, 32, 1, -1),"ori_format": "NCHW",
                     "range":[(1, None), (32, 32), (1, 99), (1, None)]},
                    [1,1,2,2], [1,1,2,2], "SAME", "NCHW")
    check_supported({"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (3, 100), (32, 32)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, -1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (2, 99), (2, 99), (32, 32)]},
                    [1,2,2,1], [1,1,2,1], "VALID", "NHWC")

def test_get_support(test_arg):
    get_op_support_info({"shape": (-1,2,1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,32,1,-1), "ori_format": "NCHW",
                     "range":[(1, None), (32, 32), (1, 100), (1, None)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (-1, 2, 1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, 32, 1, -1),"ori_format": "NCHW",
                     "range":[(1, None), (32, 32), (1, 99), (1, None)]},
                    [1,1,2,2], [1,1,2,2], "SAME", "NCHW")
    get_op_support_info({"shape": (1,2,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,-1,-1,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (3, 100), (32, 32)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, 2, -1, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, -1, -1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (2, 99), (2, 99), (32, 32)]},
                    [1,2,2,1], [1,1,2,1], "VALID", "NHWC")
    
    get_op_support_info({"shape": (-1,32,1,1), "format": "NCHW", "dtype": "float16", "ori_shape": (-1,32,1,1), "ori_format": "NCHW",
                     "range":[(1, 1), (32, 32), (1, 100), (1, 100)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (-1, 32, 1, 1),"format": "NCHW", "dtype": "float16", "ori_shape": (-1, 32, 1, 1),"ori_format": "NCHW",
                     "range":[(1, 10), (32, 32), (1, 99), (1, 99)]},
                    [1,1,2,2], [1,1,2,1], "SAME", "NCHW")

ut_case.add_cust_test_func(test_func=test_check_support)
ut_case.add_cust_test_func(test_func=test_get_support)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
