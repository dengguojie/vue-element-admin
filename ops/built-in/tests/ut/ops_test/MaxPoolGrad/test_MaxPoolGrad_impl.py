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

MaxPoolGrad ut case
"""
from op_test_frame.ut import OpUT
import math

ut_case = OpUT("MaxPoolGrad", "impl.max_pool_grad",
               "max_pool_grad")

from impl.max_pool_grad import op_select_format

def test_op_select_format(test_arg):
    x1={"shape":(1,10,10,16),"format":"NHWC","dtype":"float16","ori_shape":(1,10,10,16),"ori_format":"NHWC"}
    x2={"shape":(1,10,10,16),"format":"NHWC","dtype":"float16","ori_shape":(1,10,10,16),"ori_format":"NHWC"}
    grad={"shape":(1,10,10,16),"format":"NHWC","dtype":"float16","ori_shape":(1,10,10,16),"ori_format":"NHWC"}
    y={"shape":(1,10,10,16),"format":"NHWC","dtype":"float16","ori_shape":(1,10,10,16),"ori_format":"NHWC"}
    ksize=[1,2,2,1]
    strides=[1,1,1,1]
    padding="VALID"
    data_format="NHWC"
    x1_dy={"shape":(1,-1,-1,16),"format":"NHWC","dtype":"float16","ori_shape":(1,10,10,16),"ori_format":"NHWC"}
    op_select_format(x1, x2, grad, y, ksize, strides,padding, data_format)
    op_select_format(x1_dy, x2, grad, y, ksize, strides,padding, data_format)


ut_case.add_cust_test_func(test_func=test_op_select_format)
def _shape_4d_2_5d(in_shape, format):
    if format == "NHWC":
        n, hi, wi, c = in_shape
    else:
        n, c, hi, wi = in_shape

    c0 = 16
    c1 = math.ceil(c / c0)
    new_shape = [n, c1, hi, wi, c0]
    return new_shape


def _calc_pads(paddings, in_shape, ksize, strides, data_format):
    pads = []
    if data_format == "NHWC":
        ih, iw = in_shape[1], in_shape[2]
        kh, kw = ksize[1], ksize[2]
        sh, sw = strides[1], strides[2]
    else:
        ih, iw = in_shape[2], in_shape[3]
        kh, kw = ksize[2], ksize[3]
        sh, sw = strides[2], strides[3]

    if paddings == "VALID":
        pads = [0, 0, 0, 0]
    else:
        ho = (ih + sh - 1) // sh
        wo = (iw + sw - 1) // sw

        pad_h = max((ho - 1) * sh + kh - ih, 0)
        pad_hw_top = pad_h // 2
        pad_hw_bottom = pad_h - pad_hw_top
        pad_w = max((wo - 1) * sw + kw - iw, 0)
        pad_hw_left = pad_w // 2
        pad_hw_right = pad_w - pad_hw_left
        pads = [pad_hw_top, pad_hw_bottom, pad_hw_left, pad_hw_right]

    return pads


def gen_max_pool3d_grad_add_case(expect, case_name_val, in_dtype, ou_dtype,
                                 shape0, ori_shape0, shape1, ori_shape1, ksize,
                                 strides, pads, data_format):
    return {
        "params":
            [
                {
                    "shape": shape0,
                    "dtype": in_dtype,
                    "ori_shape": ori_shape0,
                    "ori_format": data_format,
                    "format": "NC1HWC0"
                },
                {
                    "shape": shape1,
                    "dtype": in_dtype,
                    "ori_shape": ori_shape1,
                    "ori_format": data_format,
                    "format": "NC1HWC0"
                },
                {
                    "shape": shape1,
                    "dtype": in_dtype,
                    "ori_shape": ori_shape1,
                    "ori_format": data_format,
                    "format": "NC1HWC0"
                },
                {
                    "shape": shape0,
                    "dtype": ou_dtype,
                    "ori_shape": ori_shape0,
                    "ori_format": data_format,
                    "format": "NC1HWC0"
                },
                ksize,
                strides,
                pads,
                data_format
            ],
        "case_name": case_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }


def do_case(ori_shape0,
            ori_shape1,
            ksize,
            strides,
            paddings,
            data_format,
            case_name):
    shape0 = _shape_4d_2_5d(ori_shape0, data_format)
    shape1 = _shape_4d_2_5d(ori_shape1, data_format)
    # pads = _calc_pads(paddings, ori_shape0, ksize, strides, data_format)
    pads = paddings

    params = gen_max_pool3d_grad_add_case("success", case_name,
                                          "float16", "float16",
                                          shape0, ori_shape0, shape1, ori_shape1,
                                          ksize, strides, pads, data_format)

    ut_case.add_case(["Ascend910A", "Ascend610"], params)


def gen_max_pool3d_grad_add_case_error(expect, case_name_val, in_dtype, ou_dtype,
                                       list_shape, list_ori_shape, ksize,
                                       strides, pads, data_format):
    return {"params": [{"shape": list_shape[0], "dtype": in_dtype, "ori_shape": list_ori_shape[0],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": list_shape[1], "dtype": in_dtype, "ori_shape": list_ori_shape[1],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": list_shape[2], "dtype": in_dtype, "ori_shape": list_ori_shape[2],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       {"shape": list_shape[3], "dtype": ou_dtype, "ori_shape": list_ori_shape[3],
                        "ori_format": data_format, "format": "NDC1HWC0"},
                       ksize, strides, pads, data_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def do_case_error(list_ori_shape, ksize, strides, paddings, data_format, case_name):
    list_shape = []
    for i, _ in enumerate(list_ori_shape):
        list_shape.append(_shape_4d_2_5d(list_ori_shape[i], data_format))
    # pads = _calc_pads(paddings, list_ori_shape[0], ksize, strides, data_format)
    pads = paddings
    ut_case.add_case(["Ascend910A"],
                     gen_max_pool3d_grad_add_case_error(
                         "success", case_name, "float16", "float32",
                         list_shape, list_ori_shape,
                         ksize, strides, pads, data_format))


# ============================================
# VALID: split n and c1 as core
# ============================================
# not_tiling
ori_shape0 = [158, 116, 188, 32]
ori_shape1 = [158, 2, 4, 32]
ksize = [1, 2, 108, 1]
strides = [1, 62, 21, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_0")

ori_shape0 = [32, 3, 3, 48]
ori_shape1 = [32, 1, 1, 48]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_1")


ori_shape0 = [32, 3, 3, 48]
ori_shape1 = [32, 1, 1, 48]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_2")

ori_shape0 = [32, 3, 3, 48]
ori_shape1 = [32, 2, 2, 48]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_3")

# tiling_do
ori_shape0 = [32, 32, 32, 16]
ori_shape1 = [32, 11, 11, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_4")

ori_shape0 = [32, 32, 32, 16]
ori_shape1 = [32, 16, 16, 16]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_5")

ori_shape0 = [32, 16, 16, 16]
ori_shape1 = [32, 15, 15, 16]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_6")

# tiling_do_ho
ori_shape0 = [32, 64, 64, 16]
ori_shape1 = [32, 21, 21, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_7")

ori_shape0 = [32, 64, 64, 16]
ori_shape1 = [32, 32, 32, 16]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_8")

ori_shape0 = [32, 64, 64, 16]
ori_shape1 = [32, 63, 63, 16]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_9")

# tiling_do_ho_wo
ori_shape0 = [32, 16, 1280, 16]
ori_shape1 = [32, 5, 427, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_10")

ori_shape0 = [32, 16, 1280, 16]
ori_shape1 = [32, 8, 640, 16]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_11")

ori_shape0 = [32, 16, 1280, 16]
ori_shape1 = [32, 15, 1279, 16]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_12")

# ============================================
# VALID: split n c1 x as core, x maybe d,h,w.
# ============================================
# tiling_do
ori_shape0 = [16, 3, 3, 16]
ori_shape1 = [16, 1, 1, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_13")

ori_shape0 = [16, 3, 3, 16]
ori_shape1 = [16, 1, 1, 16]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_14")

ori_shape0 = [16, 3, 3, 16]
ori_shape1 = [16, 2, 2, 16]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_15")

# tiling_do_ho
ori_shape0 = [16, 64, 64, 16]
ori_shape1 = [16, 21, 21, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_16")

ori_shape0 = [16, 64, 64, 16]
ori_shape1 = [16, 32, 32, 16]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_17")

ori_shape0 = [16, 64, 64, 16]
ori_shape1 = [16, 63, 63, 16]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_18")

# tiling_do_ho_wo
ori_shape0 = [16, 16, 1280, 16]
ori_shape1 = [16, 5, 427, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_19")

ori_shape0 = [16, 16, 1280, 16]
ori_shape1 = [16, 8, 640, 16]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_20")

ori_shape0 = [16, 16, 1280, 16]
ori_shape1 = [16, 15, 1279, 16]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_21")

ori_shape0 = [1, 935, 935, 16]
ori_shape1 = [1, 921, 461, 16]
ksize = [1, 15, 15, 1]
strides = [1, 1, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_22")

# SPECIAL SPLIT CORE
ori_shape0 = [1, 13, 4, 16]
ori_shape1 = [1, 4, 1, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_23")

ori_shape0 = [1, 7, 7, 16]
ori_shape1 = [1, 2, 2, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_24")

ori_shape0 = [1, 4, 4, 16]
ori_shape1 = [1, 1, 1, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_25")

# ============================================
# SAME:SPLIT DIFFERENT AXIS AS CORE
# ============================================
# not_tiling
ori_shape0 = [32, 3, 3, 16]
ori_shape1 = [32, 1, 1, 16]
ksize = [1, 2, 2, 1]
strides = [1, 3, 3, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_26")

ori_shape0 = [32, 3, 3, 16]
ori_shape1 = [32, 2, 2, 16]
ksize = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_27")

ori_shape0 = [32, 3, 3, 16]
ori_shape1 = [32, 3, 3, 16]
ksize = [1, 2, 2, 1]
strides = [1, 1, 1, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_28")

# ============================================
# GLOBAL:split n c1 as core
# ============================================
ori_shape0 = [16, 3, 3, 16]
ori_shape1 = [16, 1, 1, 16]
ksize = [1, 3, 3, 1]
strides = [1, 2, 2, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_29")

ori_shape0 = [1, 23, 41, 16]
ori_shape1 = [1, 1, 1, 16]
ksize = [1, 23, 41, 1]
strides = [1, 5, 7, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_30")

ori_shape0 = [16, 90, 90, 16]
ori_shape1 = [16, 1, 1, 16]
ksize = [1, 90, 90, 1]
strides = [1, 10, 10, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_31")

ori_shape0 = [1, 111, 91, 16]
ori_shape1 = [1, 1, 1, 16]
ksize = [1, 111, 91, 1]
strides = [1, 21, 17, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_32")

ori_shape0 = [1, 63, 8000, 16]
ori_shape1 = [1, 1, 1, 16]
ksize = [1, 63, 8000, 1]
strides = [1, 21, 47, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_33")

ori_shape0 = [13, 4, 391, 69]
ori_shape1 = [13, 1, 7, 69]
ksize = [1, 3, 1, 1]
strides = [1, 52, 61, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_34")

# ============================================
# exceed L1
# ============================================
ori_shape0 = [3, 1000, 6000, 112]
ori_shape1 = [3, 500, 3000, 112]
ksize = [1, 20, 20, 1]
strides = [1, 2, 2, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_35")

ori_shape0 = [301, 277, 155, 16]
ori_shape1 = [301, 3, 16, 16]
ksize = [1, 159, 1, 1]
strides = [1, 55, 10, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_76")

# ============================================
# Zero division
# ============================================
ori_shape0 = [107, 176, 16, 147]
ori_shape1 = [107, 4, 1, 147]
ksize = [1, 19, 12, 1]
strides = [1, 55, 61, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_36")

ori_shape0 = [10, 111, 117, 73]
ori_shape1 = [10, 2, 2, 73]
ksize = [1, 22, 5, 1]
strides = [1, 59, 62, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_37")

ori_shape0 = [10, 130, 98, 80]
ori_shape1 = [10, 3, 3, 80]
ksize = [1, 18, 4, 1]
strides = [1, 59, 46, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_38")

ori_shape0 = [10, 8, 167, 7]
ori_shape1 = [10, 1, 3, 7]
ksize = [1, 8, 15, 1]
strides = [1, 54, 61, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_39")

ori_shape0 = [119, 105, 80, 9]
ori_shape1 = [119, 3, 2, 9]
ksize = [1, 2, 2, 1]
strides = [1, 49, 55, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_40")

ori_shape0 = [11, 87, 85, 1]
ori_shape1 = [11, 2, 2, 1]
ksize = [1, 28, 7, 1]
strides = [1, 52, 55, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_41")

ori_shape0 = [120, 64, 32, 99]
ori_shape1 = [120, 1, 1, 99]
ksize = [1, 19, 8, 1]
strides = [1, 58, 58, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_42")

ori_shape0 = [122, 37, 10, 302]
ori_shape1 = [122, 1, 1, 302]
ksize = [1, 18, 7, 1]
strides = [1, 62, 57, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_43")

ori_shape0 = [12, 130, 176, 10]
ori_shape1 = [12, 130, 5, 10]
ksize = [1, 5, 26, 1]
strides = [1, 1, 36, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_44")

ori_shape0 = [12, 16, 77, 23]
ori_shape1 = [12, 1, 2, 23]
ksize = [1, 5, 17, 1]
strides = [1, 56, 55, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_45")

ori_shape0 = [12, 45, 36, 65]
ori_shape1 = [12, 1, 1, 65]
ksize = [1, 9, 20, 1]
strides = [1, 62, 62, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_46")

ori_shape0 = [12, 47, 328, 14]
ori_shape1 = [12, 1, 6, 14]
ksize = [1, 26, 7, 1]
strides = [1, 62, 57, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_47")

ori_shape0 = [12, 58, 170, 333]
ori_shape1 = [12, 2, 3, 333]
ksize = [1, 7, 30, 1]
strides = [1, 45, 59, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_48")

ori_shape0 = [12, 92, 369, 7]
ori_shape1 = [12, 2, 7, 7]
ksize = [1, 4, 18, 1]
strides = [1, 45, 56, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_49")

ori_shape0 = [13, 11, 189, 56]
ori_shape1 = [13, 1, 5, 56]
ksize = [1, 8, 7, 1]
strides = [1, 62, 45, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_50")

ori_shape0 = [13, 31, 5, 2]
ori_shape1 = [13, 1, 1, 2]
ksize = [1, 30, 1, 1]
strides = [1, 62, 48, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_51")

ori_shape0 = [13, 342, 48, 199]
ori_shape1 = [13, 6, 1, 199]
ksize = [1, 3, 22, 1]
strides = [1, 61, 50, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_52")

ori_shape0 = [13, 391, 84, 1]
ori_shape1 = [13, 7, 2, 1]
ksize = [1, 7, 3, 1]
strides = [1, 60, 49, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_53")

ori_shape0 = [13, 4, 391, 69]
ori_shape1 = [13, 1, 7, 69]
ksize = [1, 3, 1, 1]
strides = [1, 52, 61, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_54")

ori_shape0 = [13, 66, 310, 83]
ori_shape1 = [13, 2, 5, 83]
ksize = [1, 25, 1, 1]
strides = [1, 43, 63, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_55")

ori_shape0 = [143, 126, 225, 5]
ori_shape1 = [143, 3, 23, 5]
ksize = [1, 27, 8, 1]
strides = [1, 58, 10, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_56")

ori_shape0 = [14, 20, 181, 362]
ori_shape1 = [14, 1, 15, 362]
ksize = [1, 16, 5, 1]
strides = [1, 36, 12, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_57")

ori_shape0 = [14, 24, 200, 188]
ori_shape1 = [14, 1, 5, 188]
ksize = [1, 8, 6, 1]
strides = [1, 61, 44, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_58")

ori_shape0 = [156, 71, 352, 14]
ori_shape1 = [156, 2, 6, 14]
ksize = [1, 14, 6, 1]
strides = [1, 55, 59, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_59")

ori_shape0 = [157, 138, 8, 28]
ori_shape1 = [157, 5, 2, 28]
ksize = [1, 1, 6, 1]
strides = [1, 32, 6, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_60")

ori_shape0 = [15, 109, 13, 76]
ori_shape1 = [15, 2, 1, 76]
ksize = [1, 11, 12, 1]
strides = [1, 60, 55, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_61")

ori_shape0 = [15, 46, 383, 86]
ori_shape1 = [15, 1, 8, 86]
ksize = [1, 25, 10, 1]
strides = [1, 49, 50, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_62")

ori_shape0 = [161, 45, 90, 21]
ori_shape1 = [161, 1, 4, 21]
ksize = [1, 27, 8, 1]
strides = [1, 31, 21, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_63")

ori_shape0 = [167, 2, 190, 11]
ori_shape1 = [167, 1, 15, 11]
ksize = [1, 2, 20, 1]
strides = [1, 60, 13, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_64")

ori_shape0 = [16, 194, 20, 115]
ori_shape1 = [16, 4, 1, 115]
ksize = [1, 11, 10, 1]
strides = [1, 54, 50, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_65")

ori_shape0 = [16, 212, 141, 1]
ori_shape1 = [16, 4, 3, 1]
ksize = [1, 19, 1, 1]
strides = [1, 60, 47, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_66")

ori_shape0 = [16, 52, 56, 2]
ori_shape1 = [16, 1, 1, 2]
ksize = [1, 24, 9, 1]
strides = [1, 45, 58, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_67")

ori_shape0 = [180, 22, 49, 10]
ori_shape1 = [180, 1, 1, 10]
ksize = [1, 10, 24, 1]
strides = [1, 58, 58, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_68")

ori_shape0 = [181, 128, 78, 2]
ori_shape1 = [181, 3, 2, 2]
ksize = [1, 11, 8, 1]
strides = [1, 57, 53, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_69")

ori_shape0 = [181, 14, 102, 1]
ori_shape1 = [181, 1, 2, 1]
ksize = [1, 7, 7, 1]
strides = [1, 63, 58, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_70")

ori_shape0 = [184, 84, 185, 5]
ori_shape1 = [184, 2, 13, 5]
ksize = [1, 28, 6, 1]
strides = [1, 46, 14, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_71")

ori_shape0 = [187, 25, 65, 165]
ori_shape1 = [187, 1, 1, 165]
ksize = [1, 17, 7, 1]
strides = [1, 56, 59, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_72")

ori_shape0 = [1, 175, 228, 92]
ori_shape1 = [1, 4, 4, 92]
ksize = [1, 26, 1, 1]
strides = [1, 56, 57, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_73")

ori_shape0 = [202, 63, 178, 9]
ori_shape1 = [202, 2, 11, 9]
ksize = [1, 3, 28, 1]
strides = [1, 58, 17, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_74")

ori_shape0 = [209, 34, 7, 343]
ori_shape1 = [209, 1, 1, 343]
ksize = [1, 6, 7, 1]
strides = [1, 54, 62, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_75")


ori_shape0 = [240, 15, 60, 160]
ori_shape1 = [240, 1, 1, 160]
ksize = [1, 9, 10, 1]
strides = [1, 44, 61, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_77")

ori_shape0 = [25, 124, 86, 31]
ori_shape1 = [25, 3, 2, 31]
ksize = [1, 8, 18, 1]
strides = [1, 60, 63, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_78")

ori_shape0 = [25, 64, 385, 15]
ori_shape1 = [25, 2, 77, 15]
ksize = [1, 29, 8, 1]
strides = [1, 56, 5, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_79")

ori_shape0 = [277, 125, 38, 12]
ori_shape1 = [277, 2, 1, 12]
ksize = [1, 20, 2, 1]
strides = [1, 60, 62, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_80")

ori_shape0 = [283, 50, 41, 11]
ori_shape1 = [283, 1, 1, 11]
ksize = [1, 7, 3, 1]
strides = [1, 53, 55, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_81")

ori_shape0 = [28, 78, 48, 9]
ori_shape1 = [28, 1, 1, 9]
ksize = [1, 25, 5, 1]
strides = [1, 54, 45, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_82")

ori_shape0 = [298, 46, 104, 7]
ori_shape1 = [298, 1, 2, 7]
ksize = [1, 2, 4, 1]
strides = [1, 56, 58, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_83")

ori_shape0 = [2, 352, 274, 94]
ori_shape1 = [2, 6, 6, 94]
ksize = [1, 28, 1, 1]
strides = [1, 57, 47, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_84")

ori_shape0 = [2, 367, 362, 15]
ori_shape1 = [2, 8, 7, 15]
ksize = [1, 3, 3, 1]
strides = [1, 51, 53, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_85")

ori_shape0 = [31, 15, 115, 64]
ori_shape1 = [31, 1, 3, 64]
ksize = [1, 2, 11, 1]
strides = [1, 48, 56, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_86")

ori_shape0 = [322, 68, 62, 3]
ori_shape1 = [322, 2, 2, 3]
ksize = [1, 16, 9, 1]
strides = [1, 47, 59, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_87")

ori_shape0 = [339, 23, 73, 107]
ori_shape1 = [339, 1, 2, 107]
ksize = [1, 19, 10, 1]
strides = [1, 54, 59, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_88")

ori_shape0 = [379, 98, 3, 94]
ori_shape1 = [379, 2, 1, 94]
ksize = [1, 17, 2, 1]
strides = [1, 59, 63, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_89")

ori_shape0 = [38, 167, 5, 156]
ori_shape1 = [38, 4, 1, 156]
ksize = [1, 12, 5, 1]
strides = [1, 52, 61, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_90")

ori_shape0 = [39, 23, 249, 84]
ori_shape1 = [39, 1, 23, 84]
ksize = [1, 14, 10, 1]
strides = [1, 42, 11, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_91")

ori_shape0 = [3, 13, 69, 146]
ori_shape1 = [3, 1, 2, 146]
ksize = [1, 13, 2, 1]
strides = [1, 59, 61, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_92")


ori_shape0 = [43, 10, 33, 30]
ori_shape1 = [43, 1, 1, 30]
ksize = [1, 7, 20, 1]
strides = [1, 62, 56, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_94")

ori_shape0 = [43, 40, 7, 154]
ori_shape1 = [43, 1, 1, 154]
ksize = [1, 14, 1, 1]
strides = [1, 49, 63, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_95")

ori_shape0 = [4, 33, 91, 3]
ori_shape1 = [4, 1, 2, 3]
ksize = [1, 21, 6, 1]
strides = [1, 57, 53, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_96")

ori_shape0 = [4, 39, 74, 336]
ori_shape1 = [4, 1, 2, 336]
ksize = [1, 6, 3, 1]
strides = [1, 55, 52, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_97")

ori_shape0 = [4, 41, 323, 171]
ori_shape1 = [4, 2, 11, 171]
ksize = [1, 8, 30, 1]
strides = [1, 25, 30, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_98")

ori_shape0 = [4, 56, 56, 75]
ori_shape1 = [4, 1, 5, 75]
ksize = [1, 8, 12, 1]
strides = [1, 51, 10, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_99")

ori_shape0 = [4, 87, 282, 3]
ori_shape1 = [4, 2, 5, 3]
ksize = [1, 12, 1, 1]
strides = [1, 44, 63, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_100")

ori_shape0 = [52, 59, 83, 128]
ori_shape1 = [52, 2, 2, 128]
ksize = [1, 9, 8, 1]
strides = [1, 48, 59, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_101")

ori_shape0 = [59, 10, 139, 248]
ori_shape1 = [59, 1, 3, 248]
ksize = [1, 4, 30, 1]
strides = [1, 53, 60, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_102")

ori_shape0 = [60, 45, 166, 112]
ori_shape1 = [60, 1, 3, 112]
ksize = [1, 30, 1, 1]
strides = [1, 58, 60, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_103")

ori_shape0 = [60, 50, 87, 68]
ori_shape1 = [60, 1, 8, 68]
ksize = [1, 3, 9, 1]
strides = [1, 63, 12, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_104")

ori_shape0 = [63, 76, 8, 59]
ori_shape1 = [63, 2, 1, 59]
ksize = [1, 12, 5, 1]
strides = [1, 55, 60, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_105")

ori_shape0 = [64, 284, 3, 1]
ori_shape1 = [64, 5, 1, 1]
ksize = [1, 17, 1, 1]
strides = [1, 57, 48, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_106")

ori_shape0 = [67, 45, 29, 114]
ori_shape1 = [67, 1, 1, 114]
ksize = [1, 18, 5, 1]
strides = [1, 60, 60, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_107")

ori_shape0 = [70, 360, 8, 278]
ori_shape1 = [70, 6, 1, 278]
ksize = [1, 15, 3, 1]
strides = [1, 58, 56, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_108")

ori_shape0 = [72, 9, 131, 9]
ori_shape1 = [72, 1, 3, 9]
ksize = [1, 3, 23, 1]
strides = [1, 48, 56, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_109")

ori_shape0 = [73, 85, 34, 1]
ori_shape1 = [73, 2, 5, 1]
ksize = [1, 22, 3, 1]
strides = [1, 33, 7, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_110")

ori_shape0 = [76, 124, 306, 13]
ori_shape1 = [76, 3, 5, 13]
ksize = [1, 19, 6, 1]
strides = [1, 59, 62, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_111")

ori_shape0 = [77, 22, 220, 3]
ori_shape1 = [77, 1, 5, 3]
ksize = [1, 21, 12, 1]
strides = [1, 63, 47, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_112")

ori_shape0 = [7, 29, 90, 24]
ori_shape1 = [7, 1, 2, 24]
ksize = [1, 23, 7, 1]
strides = [1, 63, 57, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_113")

ori_shape0 = [85, 62, 24, 83]
ori_shape1 = [85, 1, 1, 83]
ksize = [1, 29, 4, 1]
strides = [1, 62, 56, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_114")

ori_shape0 = [86, 14, 155, 118]
ori_shape1 = [86, 1, 4, 118]
ksize = [1, 3, 10, 1]
strides = [1, 59, 46, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_115")

ori_shape0 = [8, 103, 167, 2]
ori_shape1 = [8, 2, 4, 2]
ksize = [1, 5, 24, 1]
strides = [1, 55, 51, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_116")

ori_shape0 = [8, 268, 144, 153]
ori_shape1 = [8, 5, 3, 153]
ksize = [1, 12, 10, 1]
strides = [1, 54, 61, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_117")

ori_shape0 = [8, 79, 190, 35]
ori_shape1 = [8, 1, 5, 35]
ksize = [1, 30, 2, 1]
strides = [1, 61, 45, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_118")

ori_shape0 = [96, 15, 154, 54]
ori_shape1 = [96, 1, 3, 54]
ksize = [1, 1, 10, 1]
strides = [1, 49, 63, 1]
paddings = "SAME"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_119")

ori_shape0 = [14, 13, 13, 34]
ori_shape1 = [14, 1, 1, 34]
ksize = [1, 13, 13, 1]
strides = [1, 12, 51, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_120")

ori_shape0 = [24, 5, 5, 5]
ori_shape1 = [24, 1, 1, 5]
ksize = [1, 5, 5, 1]
strides = [1, 22, 46, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_121")

ori_shape0 = [22, 64, 45, 67*16]
ori_shape1 = [22, 1, 4, 67*16]
ksize = [1, 42, 6, 1]
strides = [1, 32, 12, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_122")

ori_shape0 = [47, 72, 51, 42*16]
ori_shape1 = [47, 1, 1, 42*16]
ksize = [1, 13, 19, 1]
strides = [1, 60, 39, 1]
paddings = "VALID"
data_format = "NHWC"
do_case(ori_shape0, ori_shape1, ksize, strides, paddings, data_format, "case_123")
# if __name__ == '__main__':
#     ut_case.run("Ascend910")
#     exit(0)
