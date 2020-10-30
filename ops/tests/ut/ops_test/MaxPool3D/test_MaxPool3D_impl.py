#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import math
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("MaxPool3D", "impl.max_pool3d", "max_pool3d")


#=====================================Compiler==================================
case_compiler_window_eq_stride = {
     "params": [{"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16"},
                {"shape" : (1, 1, 1, 1, 1, 16), "format" : "NDHWC",  "dtype" : "float16"},
                (1, 4, 4, 4, 1),   #kernels
                (1, 4, 4, 4, 1),   #strides
                "SAME",            #padding_mode
                (0,0,0,0,0,0),     #pads
                (1,1,1),           #dilation
                0,                 #ceil_mode
                "NDHWC",           #data_format
               ],
    "case_name": "case_compiler_window_eq_stride",
     "expect": "success",
     "format_expect":[],
     "support_expect": True
}


case_compiler_window_gt_stride = {
    "params": [{"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16"},
               {"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16"},
               (1, 4, 4, 4, 1),   #kernels
               (1, 1, 1, 1, 1),   #strides
               "SAME",            #padding_mode
               (0,0,0,0,0,1),     #pads
               (1,1,1),           #dilation
               0,                 #ceil_mode
               "NDHWC",           #data_format
               ],
    "case_name": "case_compiler_window_gt_stride",
    "expect": "success",
    "format_expect":[],
    "support_expect": True
}


case_compiler_window_lt_stride = {
    "params": [{"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16"},
               {"shape" : (1, 1, 1, 1, 1, 16), "format" : "NDHWC",  "dtype" : "float16"},
               (1, 2, 2, 2, 1),   #kernels
               (1, 4, 4, 4, 1),   #strides
               "SAME",            #padding_mode
               (0,0,0,0,0,0),     #pads
               (1,1,1),           #dilation
               0,                 #ceil_mode
               "NDHWC",           #data_format
               ],
    "case_name": "case_compiler_window_lt_stride",
    "expect": "success",
    "format_expect":[],
    "support_expect": True
}



#=====================================precision==================================

def expect_data_gen(input_x, kernels, strides, padding_mode = "SAME"):

    N, D,  C1, H,  W,  C0 = input_x['shape']
    k_d, k_h, k_w = kernels
    s_d, s_h, s_w = strides
    dilation = (1,1,1)

    if padding_mode == "SAME":
        # caculate output size in SAME mode
        # Dout = ceil(Di, Sd)
        # Hout = ceil(Hi, Sh)
        # Wout = ceil(Wi, Sw)
        o_d = (D + s_d - 1)//s_d
        o_h = (H + s_h - 1)//s_h
        o_w = (W + s_w - 1)//s_w

        # Total padding on rows and cols is
        # Pd = (D' - 1) * S + (Kd - 1) * Dd + 1 - D
        # Ph = (H' - 1) * S + (Kh - 1) * Dh + 1 - H
        # Pw = (W' - 1) * S + (Kw - 1) * Dw + 1 - W
        # where (D', H', W') are output dimensions, (D, H, W) are input dims.
        # S is stride, (Dd, Dh, Dw) are dilations, (Kd, Kh, Kw) are filter dims.
        # get total pad_d, pad_h, pad_w
        p_d = (o_d - 1) * s_d + ((k_d - 1) * dilation[0] + 1) - D
        p_h = (o_h - 1) * s_h + ((k_h - 1) * dilation[1] + 1) - H
        p_w = (o_w - 1) * s_w + ((k_w - 1) * dilation[2] + 1) - W

        d_p, h_p, w_p = (o_d - 1)*s_d + k_d, (o_h - 1)*s_h + k_h, (o_w - 1)*s_w + k_w

        p_ft  = p_d // 2
        p_bk  = d_p - math.ceil(p_d / 2)
        p_t   = p_h // 2
        p_b   = h_p - math.ceil(p_h / 2)
        p_l   = p_w // 2
        p_r   = w_p - math.ceil(p_w / 2)


    elif padding_mode == "VALID":
        # caculate output size in VALID mode
        # Dout = ceil(Hi - Fd + 1, Sd)
        # Hout = ceil(Hi - Fh + 1, Sh)
        # Wout = ceil(Wi - Fw + 1, Sw)
        o_d = (D - k_d + 1 + (s_d - 1))//s_d
        o_h = (H - k_h + 1 + (s_h - 1))//s_h
        o_w = (W - k_w + 1 + (s_w - 1))//s_w
        p_d = p_h = p_w = 0
        d_p, h_p, w_p = (o_d - 1) * s_d + k_d, (o_h - 1) * s_h + k_h, (o_w - 1) * s_w + k_w
        p_ft, p_bk, p_t, p_b, p_l, p_r = 0, d_p, 0, h_p, 0, w_p

    feature_map = np.zeros((N, d_p, C1, h_p, w_p, C0))
    expect_result = np.zeros((N, o_d, C1, o_h, o_w, C0))

    expect_d = np.zeros(o_d)
    w_reduce = np.zeros((N, d_p, C1, h_p, o_w, C0))
    h_reduce = np.zeros((N, d_p, C1, o_h, o_w, C0))
    d_reduce = np.zeros((N, o_d, C1, o_h, o_w, C0))

    for n in range (N):
        for d in range (d_p):
            for c1 in range(C1):
                for h in range(h_p):
                    for w in range(w_p):
                        for c0 in range(C0):
                            if w < p_l or w >= p_r or h < p_t or h >= p_b or d < p_ft or d >= p_bk:
                                feature_map[n][d][c1][h][w][c0] = -65504.0
                            else:
                                feature_map[n][d][c1][h][w][c0] = input_x['value'][n][d-p_ft][c1][h-p_t][w-p_l][c0]

    for n in range (N):
        for d in range (d_p):
            for c1 in range(C1):
                for h in range(h_p):
                    for w in range(o_w):
                        w_begin = w * s_w
                        for c0 in range(C0):
                            for i in range(k_w):
                                if feature_map[n][d][c1][h][w_begin+i][c0] > w_reduce[n][d][c1][h][w][c0]:
                                    w_reduce[n][d][c1][h][w][c0] = feature_map[n][d][c1][h][w_begin+i][c0]
    for n in range (N):
        for d in range (d_p):
            for c1 in range(C1):
                for h in range(o_h):
                    h_begin = h * s_h
                    for w in range(o_w):
                        for c0 in range(C0):
                            for i in range(k_h):
                                if w_reduce[n][d][c1][h_begin+i][w][c0] > h_reduce[n][d][c1][h][w][c0]:
                                    h_reduce[n][d][c1][h][w][c0] = w_reduce[n][d][c1][h_begin+i][w][c0]

    for n in range (N):
        for d in range (o_d):
            d_begin = d * s_d
            for c1 in range(C1):
                for h in range(o_h):
                    for w in range(o_w):
                        for c0 in range(C0):
                            for i in range(k_d):
                                if h_reduce[n][d_begin + i][c1][h][w][c0] > d_reduce[n][d][c1][h][w][c0]:
                                    d_reduce[n][d][c1][h][w][c0] = h_reduce[n][d_begin + i][c1][h][w][c0]

    return d_reduce

def calc_expect_func(input_x, output_y, kernels, strides, padding_mode, pads, dilation, ceil_mode, data_format):
    expect_value = expect_data_gen(input_x, kernels, strides, "SAME")
    return [expect_value, ]


case_precision_window_eq_stride = {
    "params": [{"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 1, 1, 1, 1, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "output"},
               (4, 4, 4),         #kernels
               (4, 4, 4),         #strides
               "SAME",            #padding_mode
               (0,0,0,0,0,0),     #pads
               (1,1,1),           #dilation
               0,                 #ceil_mode
               "NDHWC",           #data_format
               ],
    "case_name": "case_precision_kernel_eq_stride",
    "expect": "success",
    "calc_expect_func": calc_expect_func
}


case_precision_window_eq_stride_with_multiout = {
    "params": [{"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 2, 1, 2, 2, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "output"},
               (2, 2, 2),         #kernels
               (2, 2, 2),         #strides
               "SAME",            #padding_mode
               (0,0,0,0,0,0),     #pads
               (1,1,1),           #dilation
               0,                 #ceil_mode
               "NDHWC",           #data_format
               ],
    "case_name": "case_precision_window_eq_stride_with_multiout",
    "expect": "success",
    "calc_expect_func": calc_expect_func
}


case_precision_window_gt_stride = {
    "params": [{"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "output"},
               (2, 2, 2),         #kernels
               (1, 1, 1),         #strides
               "SAME",            #padding_mode
               (0,0,0,0,0,1),     #pads
               (1,1,1),           #dilation
               0,                 #ceil_mode
               "NDHWC",           #data_format
               ],
    "case_name": "case_precision_window_gt_stride",
    "expect": "success",
    "calc_expect_func": calc_expect_func
}

case_precision_window_lt_stride = {
    "params": [{"shape" : (1, 4, 1, 4, 4, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 2, 1, 2, 2, 16), "format" : "NDHWC",  "dtype" : "float16", "param_type": "output"},
               (2, 2, 2),         #kernels
               (3, 3, 3),         #strides
               "SAME",            #padding_mode
               (0,0,0,0,0,1),     #pads
               (1,1,1),           #dilation
               0,                 #ceil_mode
               "NDHWC",           #data_format
               ],
    "case_name": "case_precision_window_lt_stride",
    "expect": "success",
    "calc_expect_func": calc_expect_func
}


ut_case.add_case("all", case_compiler_window_eq_stride)
ut_case.add_case("all", case_compiler_window_gt_stride)
ut_case.add_case("all", case_compiler_window_lt_stride)
ut_case.add_precision_case("all", case_precision_window_eq_stride)
ut_case.add_precision_case("all", case_precision_window_eq_stride_with_multiout)
ut_case.add_precision_case("all", case_precision_window_gt_stride)
ut_case.add_precision_case("all", case_precision_window_lt_stride)


