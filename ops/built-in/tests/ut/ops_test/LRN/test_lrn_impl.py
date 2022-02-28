#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# import tensorflow as tf
import numpy as np
from op_test_frame.ut import OpUT

ut_case = OpUT("LRN", "impl.lrn", "lrn")


def nchw2nc1hwc0(input_tensor):
    dst_type = input_tensor.dtype
    c0 = 16
    if dst_type == np.int8 or dst_type == np.uint8:
        c0 = 32

    input_shape = input_tensor.shape
    dim_c = input_shape[1]
    pad_value = (c0 - dim_c % c0) % c0
    tmp_tensor = np.pad(input_tensor, ((0, 0), (0, pad_value), (0, 0), (0, 0)),
                        mode="constant", constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    dim_c1 = (dim_c + pad_value) // c0
    tmp_tensor = tmp_tensor.reshape((input_shape[0], dim_c1, c0, input_shape[2], input_shape[3]))
    tmp_tensor = tmp_tensor.transpose(0, 1, 3, 4, 2)
    return tmp_tensor

def nhwc2nc1hwc0(input_tensor):
    dst_type = input_tensor.dtype
    c0 = 16
    if dst_type == np.int8 or dst_type == np.uint8:
        c0 = 32

    input_shape = input_tensor.shape
    dim_c = input_shape[3]
    pad_value = (c0 - dim_c % c0) % c0
    tmp_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, 0),(0, pad_value)),
                        mode="constant", constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    dim_c1 = (dim_c + pad_value) // c0
    tmp_tensor = tmp_tensor.reshape((input_shape[0], input_shape[2], input_shape[3]),dim_c1, c0)
    tmp_tensor = tmp_tensor.transpose(0, 3, 1, 2, 4)
    return tmp_tensor


def nc1hwc02nchw(input_tensor):
    # nc1hwc0 -> nc1c0hw
    tmp_input = np.transpose(input_tensor, [0, 1, 4, 2, 3])
    shape = tmp_input.shape
    return np.reshape(tmp_input, (shape[0], shape[1] * shape[2],
                                  shape[3], shape[4]))

def nc1hwc02nhwc(input_tensor):
    tmp_input = np.transpose(input_tensor, [0, 3, 4, 1, 2])
    shape = tmp_input.shape
    return np.reshape(tmp_input, (shape[0], shape[3], shape[4], -1))

def calc_expect_func(input_x, output_y, depth_radius=5, bias=1, alpha=1,
                     beta=0.5, norm_region="ACROSS_CHANNELS",
                     kernel_name="lrn", impl_mode="high_performance"):

    # sqr_sum[a, b, c, d] =
    # sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
    # output = input / (bias + alpha * sqr_sum) ** beta
    fmt = input_x["format"]
    if fmt == "NC1HWC0":
        input_tensor = nc1hwc02nhwc(input_x.get("value"))
    elif fmt == "NCHW":
        input_tensor = input_x.get("value")
        input_tensor = input_tensor.transpose(0, 2, 3, 1)
    shape = input_tensor.shape
    out_tensor = np.zeros(shape, dtype=input_tensor.dtype)
    for n in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                for c in range(shape[3]):
                    sum_start_idx = max(0, c - depth_radius)
                    sum_end_idx = min(shape[3] - 1, depth_radius)
                    for i in range(sum_start_idx, sum_end_idx+1):
                        out_tensor[n, h, w, c] += out_tensor[n, h, w, i]
                    pass
    out_tensor = input_tensor / (bias + alpha * out_tensor) ** beta
    out_fmt =  output_y.get("format")
    if out_fmt == "NCHW":
        out_tensor = out_tensor.transpose(0, 2, 3, 1)
    if out_fmt == "NC1HWC0":
        out_tensor = nhwc2nc1hwc0(out_tensor)
    return out_tensor


def gen_lrn_case(shape, dtype, expect, depth_radius=5, bias=1.0, alpha=1.0,
                 beta=0.5, fmt="NCHW", norm_region="ACROSS_CHANNELS",
                 kernel_name="lrn", impl_mode="high_performance"):
    return {"params": [{"dtype": dtype, "shape": shape, "format": fmt,
                        "ori_shape": shape, "ori_format": "NCHW", },
                       {"dtype": dtype, "shape": shape, "format": fmt,
                        "ori_shape": shape, "ori_format": "NCHW"},
                       depth_radius, bias, alpha, beta, norm_region],
            "case_name": kernel_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def gen_lrn_precision_case(shape, dtype, expect="success", depth_radius=5,
                           bias=1.0, alpha=1.0, beta=0.5, fmt="NCHW",
                           norm_region="ACROSS_CHANNELS", kernel_name="lrn",
                           impl_mode="high_performance"):
    return {"params": [{"dtype": dtype, "shape": shape, "format": fmt,
                        "ori_shape": shape, "ori_format": "NCHW",
                        "param_type": "input", "value_range": [-10, 10]},
                       {"dtype": dtype, "shape": shape, "format": fmt,
                        "ori_shape": shape, "ori_format": "NCHW",
                        "param_type": "output", "value_range": [-10, 10]},
                       depth_radius, bias, alpha, beta, norm_region],
            "case_name": kernel_name,
            "expect": expect,
            "calc_expect_func": calc_expect_func}


ut_case.add_case("all",
                 gen_lrn_case((1, 32, 5, 5, 16),
                              "float16", "success", 2, 1.0,
                              0.5, 0.75, "NC1HWC0",
                              kernel_name="lrn_099"))

ut_case.add_case("all",
                 gen_lrn_case((4, 1, 16, 32, 16),
                              "float16", "success", 2, 1.0,
                              0.0002, 0.75, "NC1HWC0",
                              kernel_name="lrn_100"))

ut_case.add_case("all",
                 gen_lrn_case((1, 8, 16, 32, 16),
                              "float16", "success", 2, 1.0,
                              0.5, 0.75, "NC1HWC0",
                              kernel_name="lrn_101"))


ut_case.add_case("all",
                 gen_lrn_case((1, 12, 56, 56, 16),
                              "float16", "success", 2, 1.0,
                              0.5, 0.75, "NC1HWC0",
                              kernel_name="lrn_103"))

# ut_case.add_case("all",
#                  gen_lrn_case((1, 4, 258, 256, 16),
#                               "float16", "success", 2, 1.0,
#                               0.5, 0.75, "NC1HWC0",
#                               kernel_name="lrn_104"))

ut_case.add_case("all",
                 gen_lrn_case((1, 128, 28, 28), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_001"))

ut_case.add_case("all",
                 gen_lrn_case((1, 32, 8, 8), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_002", "high_precision"))

ut_case.add_case("all",
                 gen_lrn_case((1, 6, 28, 28, 16), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NC1HWC0",
                              "ACROSS_CHANNELS", "lrn_003"))

ut_case.add_case("all",
                 gen_lrn_case((3, 1, 16, 16, 16), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NC1HWC0",
                              "ACROSS_CHANNELS", "lrn_004"))

ut_case.add_case("all",
                 gen_lrn_case((1, 16, 4, 16, 16), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NC1HWC0",
                              "ACROSS_CHANNELS", "lrn_005"))

ut_case.add_case("all",
                 gen_lrn_case((1, 20, 16, 16, 16), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NC1HWC0",
                              "ACROSS_CHANNELS", "lrn_006"))

ut_case.add_case("all",
                 gen_lrn_case((1, 4, 28, 28, 16), "float16", "success",
                              5, 1.0, 0.00002, 0.5, "NC1HWC0",
                              "ACROSS_CHANNELS", "lrn_007"))

ut_case.add_case("all",
                 gen_lrn_case((1, 256, 28, 28), "float16", "success",
                              5, 1.0, 0.00002, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_008"))

ut_case.add_case("all",
                 gen_lrn_case((1, 64, 5, 5, 16), "float16", "success",
                              5, 1.0, 0.00002, 0.5, "NC1HWC0",
                              "ACROSS_CHANNELS", "lrn_009"))

ut_case.add_case("all",
                 gen_lrn_case((1, 16, 16, 16), "float16", ValueError,
                              5, 1.0, 0.00002, 0.5, "NCHW",
                              "ACROSS_CHANNE", "lrn_100"))

ut_case.add_case("all",
                 gen_lrn_case((1, 16, 16, 16), "float16", ValueError,
                              64, 1.0, 0.00002, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_101"))

ut_case.add_case("all",
                 gen_lrn_case((1, 16, 16, 16), "float16", ValueError,
                              -1, 1.0, 0.00002, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_102"))

ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_001"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_002", "high_precision"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_003", "high_precision"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_004"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_005", "high_precision"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_006", "high_precision"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_007", "high_precision"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_008"))
ut_case.add_case(["all"],
                 gen_lrn_case((1, 192, 28, 28), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_009"))
ut_case.add_case("all",
                 gen_lrn_case((1, 2555, 255, 255), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_010"))

ut_case.add_case("all",
                 gen_lrn_case((1, 255, 255, 255), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_011"))

ut_case.add_case("all",
                 gen_lrn_case((1, 255, 3, 3), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_012"))


ut_case.add_case("all",
                 gen_lrn_case((1, 2555, 3, 3), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_013"))

ut_case.add_case("all",
                 gen_lrn_case((2555, 25, 25, 25), "float16", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_014"))

ut_case.add_case(["all"],
                 gen_lrn_case((1, 1, 1, 8), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_015"))

ut_case.add_case(["all"],
                 gen_lrn_case((2, 1, 1, 16), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_016"))

ut_case.add_case(["all"],
                 gen_lrn_case((36, 1, 1, 16), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_017"))

ut_case.add_case(["all"],
                 gen_lrn_case((2, 1, 1, 15), "float32", "success",
                              5, 1.0, 1.0, 0.5, "NCHW",
                              "ACROSS_CHANNELS", "lrn_018"))

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
