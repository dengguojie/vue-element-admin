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


# if __name__ == '__main__':
#     ut_case.run("Ascend910")
#     exit(0)
