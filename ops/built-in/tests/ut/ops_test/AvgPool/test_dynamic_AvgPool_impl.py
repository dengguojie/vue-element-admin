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
         "case_name": "dynamic_AvgPool_case_1",
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
         "case_name": "dynamic_AvgPool_case_4",
         "expect": RuntimeError,
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
         "case_name": "dynamic_AvgPool_case_17",
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
         "case_name": "dynamic_AvgPool_case_33",
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
         "case_name": "dynamic_AvgPool_case_34",
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
         "case_name": "dynamic_AvgPool_case_2",
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
         "case_name": "dynamic_AvgPool_case_3",
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
         "case_name": "dynamic_AvgPool_case_5",
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
         "case_name": "dynamic_AvgPool_case_6",
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
         "case_name": "dynamic_AvgPool_case_7",
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
         "case_name": "dynamic_AvgPool_case_8",
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
         "case_name": "dynamic_AvgPool_case_9",
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
         "case_name": "dynamic_AvgPool_case_10",
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
          "case_name": "dynamic_AvgPool_case_11",
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
          "case_name": "dynamic_AvgPool_case_12",
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
         "case_name": "dynamic_AvgPool_case_13",
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
          "case_name": "dynamic_AvgPool_case_14",
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
          "case_name": "dynamic_AvgPool_case_15",
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
         "case_name": "dynamic_AvgPool_case_16",
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
         "case_name": "dynamic_AvgPool_case_18",
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
         "case_name": "dynamic_AvgPool_case_19",
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
         "case_name": "dynamic_AvgPool_case_20",
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
         "case_name": "dynamic_AvgPool_case_21",
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
         "case_name": "dynamic_AvgPool_case_22",
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
         "case_name": "dynamic_AvgPool_case_23",
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
         "case_name": "dynamic_AvgPool_case_24",
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
         "case_name": "dynamic_AvgPool_case_25",
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
         "case_name": "dynamic_AvgPool_case_26",
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
         "case_name": "dynamic_AvgPool_case_27",
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
         "case_name": "dynamic_AvgPool_case_28",
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
         "case_name": "dynamic_AvgPool_case_29",
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
         "case_name": "dynamic_AvgPool_case_30",
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
         "case_name": "dynamic_AvgPool_case_31",
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
         "case_name": "dynamic_AvgPool_case_32",
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
         "case_name": "dynamic_AvgPool_case_35",
         "expect": "success",
         "support_expect": True}

# kh * kw > 255 
case36 = {"params": [{"shape": (1,-1,2,32), "format": "NHWC", "dtype": "float16", "ori_shape": (1,-1,2,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (2, 100), (32, 32)]},
                    {"shape": (8,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (32, 1, 2, 2),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (1, -1, 1, 32),"format": "NHWC", "dtype": "float16", "ori_shape": (1, -1, 1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (1, 99), (1, 99), (32, 32)]},
                    [1,16,15,1], [1,2,1,1], "VALID", "NHWC"],
         "case_name": "dynamic_AvgPool_case_36",
         "expect": RuntimeError,
         "support_expect": True}

case37 = {"params": [{"shape": (-1,1,752,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1,5,752,-1), "ori_format": "NCHW",
                     "range":[(2, 21), (5, 5), (752, 752), (50, 2279)]},
                    {"shape": (5,1,1,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (5, 1, 1, 1),"ori_format": "NCHW",
                     "range":[(32, 32), (1, 1), (2, 2), (2, 2)]},
                    None,
                    {"shape": (-1, 1, 752, -1, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (-1, 5, 752, -1),"ori_format": "NCHW",
                     "range":[(1, None), (1, 1), (752, 752), (1, None), (16, 16)]},
                    [1,1,1,1], [1,1,3,1], "SAME", "NCHW"],
         "case_name": "dynamic_AvgPool_case_37",
         "expect": RuntimeError,
         "support_expect": True}


case38 = {"params": [{"shape": (1,-1,-1,32), "format": "NHWC", "dtype": "float16", "ori_shape": (1,-1,-1,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (2, 100), (32, 32)]},
                    None,
                    None,
                    {"shape": (1, -1, -1, 32),"format": "NHWC", "dtype": "float16", "ori_shape": (1, -1, -1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (1, 99), (1, 99), (32, 32)]},
                    [1,16,15,1], [1,2,2,1], "SAME", "NHWC"],
         "case_name": "dynamic_AvgPool_case_38",
         "expect": "success",
         "support_expect": True}

case39 = {"params": [{"shape": (1,-1,-1,32), "format": "NHWC", "dtype": "float16", "ori_shape": (1,-1,-1,32), "ori_format": "NHWC",
                     "range":[(1, 1), (3, 100), (2, 100), (32, 32)]},
                    None,
                    None,
                    {"shape": (1, -1, -1, 32),"format": "NHWC", "dtype": "float16", "ori_shape": (1, -1, -1, 32),"ori_format": "NHWC",
                     "range":[(1, 1), (1, 99), (1, 99), (32, 32)]},
                    [1,16,15,1], [1,2,2,1], "VALID", "NHWC"],
         "case_name": "dynamic_AvgPool_case_39",
         "expect": "success",
         "support_expect": True}

# check al1 buffer_tile
case_al1_buffer_tile = {"params": [{"shape": (-1,17,-1,-1), "format": "NCHW", "dtype": "float16", "ori_shape": (-1,17,-1,-1), "ori_format": "NCHW",
                     "range":[(32, 36), (17, 17), (64, 127), (128, 191)]},
                    {"shape": (168,1,16,16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (17, 1, 6, 14),"ori_format": "NCHW",
                     "range":[(17, 17), (1, 1), (6, 6), (14, 14)]},
                    None,
                    {"shape": (-1, 17, -1, -1),"format": "NCHW", "dtype": "float16", "ori_shape": (-1, 17, -1, -1),"ori_format": "NCHW",
                     "range":[(32, 36), (17, 17), (20, 30), (30, 35)]},
                    [1,1,6,14], [1,1,4,4], "VALID", "NCHW"],
         "case_name": "dynamic_AvgPool_case_al1_buffer_tile",
         "expect": RuntimeError,
         "support_expect": True}


ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
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
ut_case.add_case("all", case17)
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
ut_case.add_case("all", case33)
ut_case.add_case("all", case34)
ut_case.add_case(["Ascend910A"], case35)
ut_case.add_case(["Ascend910A"], case36)
ut_case.add_case(["Ascend910A"], case37)
ut_case.add_case("all", case38)
ut_case.add_case("all", case39)
ut_case.add_case(["Ascend910A"], case_al1_buffer_tile)

def test_avg_pool_fuzz_build_generalization(test_arg):
    from impl.dynamic.avg_pool import avg_pool_generalization
    input_list = [
        {
            'shape': (16, 3, 16, 16, 16),
            'ori_shape': (16, 33, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'ori_shape': (33, 1, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 3, 5), (1, 1, 1, 1), "VALID", 'NCHW', 0, 'avg_pool_fuzz_build_generalization']
    avg_pool_generalization(*input_list)
print("adding avg_pool test_avg_pool_fuzz_build_generalization testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_fuzz_build_generalization)

def test_avg_pool_fuzz_build_tilingcase(test_arg):
    import json
    from impl.dynamic.avg_pool import avg_pool
    from tbe.common.context import get_context
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        get_context().set_build_type("fuzzily_build")
        get_context().add_addition("max_kernel_id", -1)
        missing_info = [{
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [16, 32],
                                        [33, 33],
                                        [256, 512],
                                        [256, 512]
                                    ],
                                    "shape": [-1, 33, -1, -1]
                                }]
                            }]
                        }, {
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [16, 32],
                                        [33, 33],
                                        [512, 1024],
                                        [512, 1024]
                                    ],
                                    "shape": [-1, 33, -1, -1]
                                }]
                            }]
                        }]
        get_context().add_addition("missing_support_info", json.dumps(missing_info))

        input_list = [
            {
                'shape': (-1, 3, -1, -1, 16),
                'ori_shape': (-1, 33, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16',
                'range': ((16, 32), (33, 33), (256, 1024), (256, 1024))
            }, {
                'ori_shape': (33, 1, 3, 5),
                'ori_format': 'NCHW',
                'format': 'FRACTAL_Z',
                'dtype': 'float16'
            }, None, {
                'shape': (-1, 3, -1, -1, 16),
                'ori_shape': (-1, 33, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16'
            }, (1, 1, 3, 5), (1, 1, 1, 1), "VALID", 'NCHW', 0, 'avg_pool_fuzz_build_generalization']
        avg_pool(*input_list)
print("adding avg_pool test_conv2d_fuzz_build_tilingcase testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_fuzz_build_tilingcase)

def test_avg_pool_fuzz_build_correct_range(test_arg):
    from impl.dynamic.avg_pool import avg_pool_generalization
    input_list = [
        {
            'shape': (1, 4, 1080, 1080, 16),
            'ori_shape': (1, 64, 1080, 1080),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 2), (1, 1), (1024, 4096), (1024, 4096), (16, 16)),
            'ori_range': ((1, 2), (3, 3), (1024, 4096), (1024, 4096))
            }, {
                'ori_shape': (64, 64, 7, 7),
                'ori_format': 'NCHW',
                'format': 'FRACTAL_Z',
                'dtype': 'float16'
            }, None, {
            'shape': (-1, 4, -1, -1, 16),
            'ori_shape': (-1, 64, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 7, 7), (1, 1, 2, 2), "SAME", 'NCHW', 0, 'test_avg_pool_fuzz_build_correct_range']
    avg_pool_generalization(*input_list)
print("adding conv2d test_conv2d_fuzz_build_correct_range testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_fuzz_build_correct_range)

def test_avg_pool_fuzz_build_err_range(test_arg):
    from impl.dynamic.avg_pool import avg_pool_generalization
    input_list = [
        {
            'shape': (1, 4, 1080, 1080, 16),
            'ori_shape': (1, 64, 1080, 1080, 1080, 1080),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 2), (1, 1), (1024, 4096), (1024, 4096), (16, 16)),
            'ori_range': ((1, 2), (3, 3), (1024, 4096), (1024, 4096))
            }, {
                'ori_shape': (64, 64, 7, 7),
                'ori_format': 'NCHW',
                'format': 'FRACTAL_Z',
                'dtype': 'float16'
            }, None, {
            'shape': (-1, 4, -1, -1, 16),
            'ori_shape': (-1, 64, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 7, 7), (1, 1, 2, 2), "SAME", 'NCHW', 0, 'test_avg_pool_fuzz_build_err_range']
    try:
        avg_pool_generalization(*input_list)
    except RuntimeError:
        pass
print("adding conv2d test_avg_pool_fuzz_build_err_range testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_fuzz_build_err_range)

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

from impl.dynamic.avg_pool import avg_pool_generalization

# output_h lower than 1
# format NCHW, valid mode, kh is 17, larger than ori_range h lowest 16
def test_avg_pool_graph_mode_output_h_format_NCHW(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (16, 31), (64, 127)),
            'ori_range': ((1, 1), (16, 31), (16, 31), (64, 127))
        }, {
            # filter
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 1, 17, 17), (1, 1, 1, 1), "VALID", 'NCHW', 0, 'avg_pool_graph_mode_output_h_format_NCHW']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_output_h_format_NCHW failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_output_h_format_NCHW testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_output_h_format_NCHW)

# output_w lower than 1
# format NCHW, valid mode, kw is 17, larger than ori_range w lowest 16
def test_avg_pool_graph_mode_output_w_format_NCHW(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 1, 17, 17), (1, 1, 1, 1), "VALID", 'NCHW', 0, 'avg_pool_graph_mode_output_w_format_NCHW']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_output_w_format_NCHW failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_output_w_format_NCHW testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_output_w_format_NCHW)

# output_h lower than 1
# format NHWC, valid mode, kh is 17, larger than ori_range h lowest 16
def test_avg_pool_graph_mode_output_h_format_NHWC(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 17, 17, 1), (1, 1, 1, 1), "VALID", 'NHWC', 0, 'avg_pool_graph_mode_output_h_format_NHWC']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_output_h_format_NHWC failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_output_h_format_NHWC testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_output_h_format_NHWC)

# output_w lower than 1
# format NHWC, valid mode, kw is 17, larger than ori_range w lowest 16
def test_avg_pool_graph_mode_output_w_format_NHWC(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 17, 17),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 17, 17, 1), (1, 1, 1, 1), "VALID", 'NHWC', 0, 'avg_pool_graph_mode_output_w_format_NHWC']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["lower_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_output_w_format_NHWC failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_output_w_format_NHWC testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_output_w_format_NHWC)

# limit_size larger than l1_size
# format NCHW, valid mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_graph_mode_l1_size_NCHW_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (1024, 2047)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (1024, 2047))
        }, {
            # filter
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 1, 16, 1), (1, 1, 1, 1), "VALID", 'NCHW', 0, 'avg_pool_graph_mode_l1_size_NCHW_valid']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_l1_size_NCHW_valid failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_l1_size_NCHW_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_l1_size_NCHW_valid)

# limit_size larger than l1_size
# format NHWC, valid mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_graph_mode_l1_size_NHWC_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (1024, 2047), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (1024, 2047), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 16, 1, 1), (1, 1, 1, 1), "VALID", 'NHWC', 0, 'avg_pool_graph_mode_l1_size_NHWC_valid']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_l1_size_NHWC_valid failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_l1_size_NHWC_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_l1_size_NHWC_valid)

# limit_size larger than l1_size
# format NCHW, same mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_graph_mode_l1_size_NCHW_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (1024, 2047)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (1024, 2047))
        }, {
            # filter
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 1, 16, 1), (1, 1, 1, 1), "SAME", 'NCHW', 0, 'avg_pool_graph_mode_l1_size_NCHW_same']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_l1_size_NCHW_same failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_l1_size_NCHW_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_l1_size_NCHW_same)

# limit_size larger than l1_size
# format NHWC, same mode, kw is 1, kh is 16, ori_range w largest is 2047
def test_avg_pool_graph_mode_l1_size_NHWC_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (1024, 2047), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (1024, 2047), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 16, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name
        (1, 16, 1, 1), (1, 1, 1, 1), "SAME", 'NHWC', 0, 'avg_pool_graph_mode_l1_size_NHWC_same']
    ret = avg_pool_generalization(*input_list)
    if ret != [{"result" : "UNSUPPORTED", "reason" : {"param_index" : [0], "type" : ["upper_limit"]}}]:
        raise Exception("test_avg_pool_graph_mode_l1_size_NHWC_same failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_l1_size_NHWC_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_l1_size_NHWC_same)

# generalize_config unsupported
# raise error
def test_avg_pool_generalize_config_unsupported(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 1, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 1, 1, 1), (1, 1, 1, 1), "VALID", 'NHWC', 0, 'avg_pool_generalize_config_unsupported', {"mode": "keep"}]
    try:
        avg_pool_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding avg_pool test_avg_pool_generalize_config_unsupported testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_generalize_config_unsupported)

# ori_shape is unknown rank
# raise error
def test_avg_pool_unknown_rank(test_arg):
    input_list = [
        {
            # inputs
            'shape': [-2],
            'ori_shape': [-2],
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 1, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 1, 1, 1), (1, 1, 1, 1), "VALID", 'NHWC', 0, 'avg_pool_unknown_rank']
    try:
        avg_pool_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding avg_pool test_avg_pool_unknown_rank testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_unknown_rank)

# unsupported format
# raise error
def test_avg_pool_unsupported_format(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'HWCN',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 1, 1),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 1, 1, 1), (1, 1, 1, 1), "VALID", 'NHWC', 0, 'avg_pool_unsupported_format']
    try:
        avg_pool_generalization(*input_list)
    except RuntimeError:
        print("expected")
        pass
print("adding avg_pool test_avg_pool_unsupported_format testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_unsupported_format)

# supported range
# NCHW, valid
def test_avg_pool_graph_mode_supported_range_NCHW_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 1, 3, 3), (1, 1, 1, 1), "VALID", 'NCHW', 0, 'avg_pool_graph_mode_supported_range_NCHW_valid']
    ret = avg_pool_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_graph_mode_supported_range_NCHW_valid failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_supported_range_NCHW_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_supported_range_NCHW_valid)

# supported range
# NCHW, same
def test_avg_pool_graph_mode_supported_range_NCHW_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (16, 31), (64, 127), (16, 31)),
            'ori_range': ((1, 1), (16, 31), (64, 127), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 16, 1, 1),
            'ori_shape': (-1, 16, 1, 1),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 1, 3, 3), (1, 1, 1, 1), "SAME", 'NCHW', 0, 'avg_pool_graph_mode_supported_range_NCHW_same']
    ret = avg_pool_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_graph_mode_supported_range_NCHW_same failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_supported_range_NCHW_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_supported_range_NCHW_same)

# supported range
# NHWC, valid
def test_avg_pool_graph_mode_supported_range_NHWC_valid(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "VALID", 'NHWC', 0, 'avg_pool_graph_mode_supported_range_NHWC_valid']
    ret = avg_pool_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_graph_mode_supported_range_NCHW_same failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_supported_range_NHWC_valid testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_supported_range_NHWC_valid)

# supported range
# NHWC, same
def test_avg_pool_graph_mode_supported_range_NHWC_same(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, 127), (16, 31), (16, 31)),
            'ori_range': ((1, 1), (64, 127), (16, 31), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "SAME", 'NHWC', 0, 'avg_pool_graph_mode_supported_range_NHWC_same']
    ret = avg_pool_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_graph_mode_supported_range_NHWC_same failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_supported_range_NHWC_same testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_supported_range_NHWC_same)

# supported range
# range includes "None"
def test_avg_pool_graph_mode_supported_range_none(test_arg):
    input_list = [
        {
            # inputs
            'shape': (1, -1, -1, -1),
            'ori_shape': (1, -1, -1, -1),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            'range': ((1, 1), (64, None), (16, None), (16, 31)),
            'ori_range': ((1, 1), (64, None), (16, None), (16, 31))
        }, {
            # filter
            'ori_shape': (16, 1, 3, 3),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, {
            'shape': (-1, 1, 1, 16),
            'ori_shape': (-1, 1, 1, 16),
            'ori_format': 'NHWC',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        },
        # ksize, strides, padding, data_format, offset_x, kernel_name, generalize_config
        (1, 3, 3, 1), (1, 1, 1, 1), "SAME", 'NHWC', 0, 'avg_pool_graph_mode_supported_range_none']
    ret = avg_pool_generalization(*input_list)
    if ret != [input_list]:
        raise Exception("test_avg_pool_graph_mode_supported_range_none failed")
    else:
        print("expected")
print("adding avg_pool test_avg_pool_graph_mode_supported_range_none testcase")
ut_case.add_cust_test_func(test_func=test_avg_pool_graph_mode_supported_range_none)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
