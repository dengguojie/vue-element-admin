#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch
from op_test_frame.ut import OpUT
import math

ut_case = OpUT("MaxPool3DGrad", "impl.max_pool3d_grad",
               "max_pool3d_grad")


def _shape_5d_2_6d(in_shape, format):
    if format == "NDHWC":
        n, di, hi, wi, c = in_shape
    else:
        n, c, di, hi, wi = in_shape

    c0 = 16
    c1 = math.ceil(c / c0)
    new_shape = [n, di, c1, hi, wi, c0]
    return new_shape


def _calc_pads(paddings, in_shape, ksize, strides, data_format):
    pads = []
    if data_format == "NDHWC":
        id, ih, iw = in_shape[1], in_shape[2], in_shape[3]
        kd, kh, kw = ksize[1], ksize[2], ksize[3]
        sd, sh, sw = strides[1], strides[2], strides[3]
    else:
        id, ih, iw = in_shape[2], in_shape[3], in_shape[4]
        kd, kh, kw = ksize[2], ksize[3], ksize[4]
        sd, sh, sw = strides[2], strides[3], strides[4]

    # suppose CALCULATED is 0,0,0,0,0,0
    if paddings in ["VALID", "CALCULATED"]:
        pads = [0, 0, 0, 0, 0, 0]
    else:
        do = (id + sd - 1) // sd
        ho = (ih + sh - 1) // sh
        wo = (iw + sw - 1) // sw

        pad_h = max((ho - 1) * sh + kh - ih, 0)
        pad_hw_top = pad_h // 2
        pad_hw_bottom = pad_h - pad_hw_top
        pad_w = max((wo - 1) * sw + kw - iw, 0)
        pad_hw_left = pad_w // 2
        pad_hw_right = pad_w - pad_hw_left
        pad_d = max((do - 1) * sd + kd - id, 0)
        pad_d_top = pad_d // 2
        pad_d_bottom = pad_d - pad_d_top
        pads = [pad_d_top, pad_d_bottom, pad_hw_top, pad_hw_bottom, pad_hw_left, pad_hw_right]

    return pads


def gen_max_pool3d_grad_add_case(expect, case_name_val, in_dtype, ou_dtype,
                                 shape0, ori_shape0, shape1, ori_shape1, ksize,
                                 strides, pads, data_format, paddings):
    return {"params": [{"shape": shape0, "dtype": in_dtype, "ori_shape": ori_shape0,
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": shape1, "dtype": in_dtype, "ori_shape": ori_shape1,
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": shape1, "dtype": in_dtype, "ori_shape": ori_shape1,
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": shape0, "dtype": ou_dtype, "ori_shape": ori_shape0,
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       ksize, strides, paddings, pads, data_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format):
    shape0 = _shape_5d_2_6d(ori_shape0, data_format)
    shape1 = _shape_5d_2_6d(ori_shape1, data_format)
    pads = _calc_pads(paddings, ori_shape0, ksize, strides, data_format)
    ut_case.add_case(["Ascend910A", "Ascend610"],
                     gen_max_pool3d_grad_add_case(
                         "success", "VALID_X", "float16", "float32",
                         shape0, ori_shape0, shape1, ori_shape1,
                         ksize, strides, pads, data_format, paddings))


def gen_max_pool3d_grad_add_case_error(expect, case_name_val, in_dtype, ou_dtype,
                                       list_shape, list_ori_shape, ksize,
                                       strides, pads, data_format, paddings):
    return {"params": [{"shape": list_shape[0], "dtype": in_dtype, "ori_shape": list_ori_shape[0],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": list_shape[1], "dtype": in_dtype, "ori_shape": list_ori_shape[1],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": list_shape[2], "dtype": in_dtype, "ori_shape": list_ori_shape[2],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": list_shape[3], "dtype": ou_dtype, "ori_shape": list_ori_shape[3],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       ksize, strides, paddings, pads, data_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": False}


def do_case_error_one(ori_shape0, ori_shape1, ksize, strides, paddings, data_format):
    shape0 = _shape_5d_2_6d(ori_shape0, data_format)
    shape1 = _shape_5d_2_6d(ori_shape1, data_format)
    pads = _calc_pads(paddings, ori_shape0, ksize, strides, data_format)
    ut_case.add_case(["Ascend910A", "Ascend610"],
                     gen_max_pool3d_grad_add_case(
                         RuntimeError, "VALID_X", "float16", "float32",
                         shape0, ori_shape0, shape1, ori_shape1,
                         ksize, strides, pads, data_format, paddings))


def do_case_error(list_ori_shape, ksize, strides, paddings, data_format):
    list_shape = []
    for i, _ in enumerate(list_ori_shape):
        list_shape.append(_shape_5d_2_6d(list_ori_shape[i], data_format))
    pads = _calc_pads(paddings, list_ori_shape[0], ksize, strides, data_format)
    ut_case.add_case(["Ascend910A", "Ascend610"],
                     gen_max_pool3d_grad_add_case_error(
                         RuntimeError, "VALID_X", "float16", "float32",
                         list_shape, list_ori_shape,
                         ksize, strides, pads, data_format, paddings))


# ============================================
# VALID: split n and c1 as core
# ============================================
# not_tiling
ori_shape0 = [32, 12, 3, 3, 48]
ori_shape1 = [32, 4, 1, 1, 48]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "CALCULATED"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)


ori_shape0 = [32, 12, 3, 3, 48]
ori_shape1 = [32, 4, 1, 1, 48]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 12, 3, 3, 48]
ori_shape1 = [32, 4, 1, 1, 48]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 3, 3, 3]
paddings = "VALID"
data_format = "NDHWC"
do_case_error_one(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 12, 3, 3, 48]
ori_shape1 = [32, 4, 1, 1, 48]
ksize = [1, 1, 2, 2, 2]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case_error_one(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 48, 3, 3, 12]
ori_shape1 = [32, 48, 1, 1, 4]
ksize = [1, 1, 2, 2, 2]
strides = [1, 1, 3, 3, 3]
paddings = "VALID"
data_format = "NCDHW"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 48, 3, 3, 12]
ori_shape1 = [32, 48, 1, 1, 4]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 3, 3, 3]
paddings = "VALID"
data_format = "NCDHW"
do_case_error_one(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 48, 3, 3, 12]
ori_shape1 = [32, 48, 1, 1, 4]
ksize = [1, 1, 2, 2, 2]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NCDHW"
do_case_error_one(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 12, 3, 3, 48]
ori_shape1 = [32, 6, 1, 1, 48]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 12, 3, 3, 48]
ori_shape1 = [32, 11, 2, 2, 48]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do
ori_shape0 = [32, 32, 32, 32, 16]
ori_shape1 = [32, 11, 11, 11, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 32, 32, 32, 16]
ori_shape1 = [32, 16, 16, 16, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 16, 16, 16, 16]
ori_shape1 = [32, 15, 15, 15, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do_ho
ori_shape0 = [32, 64, 64, 64, 16]
ori_shape1 = [32, 21, 21, 21, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 64, 64, 64, 16]
ori_shape1 = [32, 32, 32, 32, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 64, 64, 64, 16]
ori_shape1 = [32, 63, 63, 63, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do_ho_wo
ori_shape0 = [32, 16, 16, 1280, 16]
ori_shape1 = [32, 5, 5, 427, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 16, 16, 1280, 16]
ori_shape1 = [32, 8, 8, 640, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 16, 16, 1280, 16]
ori_shape1 = [32, 15, 15, 1279, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# ============================================
# VALID: split n c1 x as core, x maybe d,h,w.
# ============================================
# tiling_do
ori_shape0 = [16, 12, 3, 3, 16]
ori_shape1 = [16, 4, 1, 1, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 144, 3, 3, 16]
ori_shape1 = [16, 48, 1, 1, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 12, 3, 3, 16]
ori_shape1 = [16, 6, 1, 1, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 12, 3, 3, 16]
ori_shape1 = [16, 11, 2, 2, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do_ho
ori_shape0 = [16, 64, 64, 64, 16]
ori_shape1 = [16, 21, 21, 21, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 64, 64, 64, 16]
ori_shape1 = [16, 32, 32, 32, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 64, 64, 64, 16]
ori_shape1 = [16, 63, 63, 63, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do_ho_wo
ori_shape0 = [16, 16, 16, 1280, 16]
ori_shape1 = [16, 5, 5, 427, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 16, 16, 1280, 16]
ori_shape1 = [16, 8, 8, 640, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 16, 16, 1280, 16]
ori_shape1 = [16, 15, 15, 1279, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# SPECIAL SPLIT CORE
ori_shape0 = [1, 25, 13, 4, 16]
ori_shape1 = [1, 8, 4, 1, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [1, 25, 7, 7, 16]
ori_shape1 = [1, 8, 2, 2, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [1, 4, 4, 4, 16]
ori_shape1 = [1, 1, 1, 1, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# ============================================
# SAME:SPLIT DIFFERENT AXIS AS CORE
# ============================================
# not_tiling
ori_shape0 = [32, 12, 3, 3, 16]
ori_shape1 = [32, 4, 1, 1, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 12, 3, 3, 16]
ori_shape1 = [32, 6, 2, 2, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [32, 12, 3, 3, 16]
ori_shape1 = [32, 12, 3, 3, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do
ori_shape0 = [16, 12, 3, 3, 16]
ori_shape1 = [16, 4, 1, 1, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 12, 3, 3, 16]
ori_shape1 = [16, 6, 2, 2, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 12, 3, 3, 16]
ori_shape1 = [16, 12, 3, 3, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do_ho
ori_shape0 = [16, 64, 64, 64, 16]
ori_shape1 = [16, 22, 22, 22, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 64, 64, 64, 16]
ori_shape1 = [16, 32, 32, 32, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 64, 64, 64, 16]
ori_shape1 = [16, 64, 64, 64, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# tiling_do_ho_wo
ori_shape0 = [16, 16, 16, 1280, 16]
ori_shape1 = [16, 6, 6, 427, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 3, 3, 3, 1]
paddings = "SAME"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 16, 16, 1280, 16]
ori_shape1 = [16, 8, 8, 640, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 2, 2, 2, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

ori_shape0 = [16, 16, 16, 1280, 16]
ori_shape1 = [16, 16, 16, 1280, 16]
ksize = [1, 2, 2, 2, 1]
strides = [1, 1, 1, 1, 1]
paddings = "VALID"
data_format = "NDHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# ============================================
# NCDHW AND ERROR
# ============================================
# NCDHW
# ori_shape0 = [16, 64, 64, 64, 16]
# ori_shape1 = [16, 64, 64, 64, 16]
# ksize = [1, 2, 2, 2, 1]
# strides = [1, 1, 1, 1, 1]
# paddings = "SAME"
# data_format = "NCDHW"
# do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# # TODO fixme run failed
# # error of format
# data_format = "NHWC"
# # do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# # error of ksize
# ksize = [2, 2, 2, 2, 1]
# data_format = "NCDHW"
# # do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# # error of strides
# ksize = [1, 2, 2, 2, 1]
# strides = [2, 1, 1, 1, 1]
# # do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# # TODO fixme erorr in case file
# # error of ksize
# # ksize = [2, 2, 2, 1]
# # strides = [1, 1, 1, 1, 1]
# # do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format)

# # error of ori_input_shape
# ori_shape0 = [16, 64, 64, 64]
# ori_shape1 = [16, 64, 64, 64, 16]
# ori_shape2 = [16, 64, 64, 64, 16]
# ori_shape3 = [16, 64, 64, 64, 16]
# list_ori_shape = [ori_shape0, ori_shape1, ori_shape2, ori_shape3]
# ksize = [1, 2, 2, 2, 1]
# strides = [1, 1, 1, 1, 1]
# paddings = "SAME"
# data_format = "NCDHW"
# # do_case_error(list_ori_shape, ksize, strides, paddings, data_format)

# # error of ori_output_shape
# list_ori_shape[0] = [16, 64, 64, 64, 16]
# list_ori_shape[1] = [16, 64, 64, 64]
# # do_case_error(list_ori_shape, ksize, strides, paddings, data_format)

# # error of grad_shape
# list_ori_shape[1] = [16, 64, 64, 64, 16]
# list_ori_shape[2] = [16, 64, 64, 64]
# # do_case_error(list_ori_shape, ksize, strides, paddings, data_format)

# # error: grad_shape != ori_output_shape
# list_ori_shape[2] = [16, 64, 64, 64, 16]
# list_ori_shape[1] = [16, 64, 64, 64, 16]
# list_ori_shape[2] = [16, 64, 63, 64, 16]
# # do_case_error(list_ori_shape, ksize, strides, paddings, data_format)

# if __name__ == '__main__':
#     ut_case.run("Ascend910A")

vals = {("tik.load3dv1",): False}
def side_effects(*args):
    return vals[args]
with patch("te.platform.cce_conf.api_check_support", MagicMock(side_effect=side_effects)):
    ut_case.run("Ascend910A")

