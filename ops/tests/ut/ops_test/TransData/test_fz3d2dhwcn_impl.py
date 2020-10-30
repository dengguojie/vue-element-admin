#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", None, None)


def calc_expect_func(src, dst, src_format, dst_format):
    input_shape = src.get("shape")
    input_tensor = src.get("value")
    dst_shape = dst.get("shape")
    axis_d = dst_shape[0]
    axis_h = dst_shape[1]
    axis_w = dst_shape[2]
    axis_c = dst_shape[3]
    axis_n = dst_shape[4]
    axis_dc1hw = input_shape[0]
    axis_no = input_shape[1]
    axis_ni = input_shape[2]
    axis_c0 = input_shape[3]
    axis_c1 = axis_dc1hw // axis_d // axis_h // axis_w
    n_pad = None if axis_no * axis_ni == axis_n else axis_n - axis_no * axis_ni
    c_pad = None if axis_c1 * axis_c0 == axis_c else axis_c - axis_c1 * axis_c0

    tmp_input_tensor = input_tensor.reshape(axis_d, axis_c1, axis_h, axis_w,
                                            axis_no, axis_ni, axis_c0)
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1, 6, 4, 5))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_d, axis_h, axis_w,
                                                axis_c1 * axis_c0, axis_no * axis_ni)
    output_tensor = tmp_input_tensor[:, :, :, :c_pad, :n_pad].reshape(dst_shape)

    return output_tensor


err1 = {"params": [{"shape": (120, 4, 16, 16), "dtype": "float16",
                    "ori_shape": (120, 4, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (3, 4, 5, 32, 64), "dtype": "float32",
                    "ori_shape": (3, 4, 5, 32, 64), "format": "DHWCN", "ori_format": "DHWCN"},
                   "FRACTAL_Z_3D", "DHWCN"],
        "expect": RuntimeError,
        "format_expect": ["DHWCN"],
        "support_expect": False}

err2 = {"params": [{"shape": (120, 4, 32, 16), "dtype": "float16",
                    "ori_shape": (120, 4, 32, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (3, 4, 5, 32, 64), "dtype": "float16",
                    "ori_shape": (3, 4, 5, 32, 64), "format": "DHWCN", "ori_format": "DHWCN"},
                   "FRACTAL_Z_3D", "DHWCN"],
        "expect": RuntimeError,
        "format_expect": ["DHWCN"],
        "support_expect": False}

err3 = {"params": [{"shape": (120, 4, 16, 32), "dtype": "float16",
                    "ori_shape": (120, 4, 16, 32), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (3, 4, 5, 32, 64), "dtype": "float16",
                    "ori_shape": (3, 4, 5, 32, 64), "format": "DHWCN", "ori_format": "DHWCN"},
                   "FRACTAL_Z_3D", "DHWCN"],
        "expect": RuntimeError,
        "format_expect": ["DHWCN"],
        "support_expect": False}

err4 = {"params": [{"shape": (120, 3, 16, 16), "dtype": "float16",
                    "ori_shape": (120, 3, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (3, 4, 5, 32, 64), "dtype": "float16",
                    "ori_shape": (3, 4, 5, 32, 64), "format": "DHWCN", "ori_format": "DHWCN"},
                   "FRACTAL_Z_3D", "DHWCN"],
        "expect": RuntimeError,
        "format_expect": ["DHWCN"],
        "support_expect": False}

err5 = {"params": [{"shape": (100, 4, 16, 16), "dtype": "float16",
                    "ori_shape": (100, 4, 16, 16), "format": "FRACTAL_Z_3D",
                    "ori_format": "FRACTAL_Z_3D"},
                   {"shape": (3, 4, 5, 32, 64), "dtype": "float16",
                    "ori_shape": (3, 4, 5, 32, 64), "format": "DHWCN", "ori_format": "DHWCN"},
                   "FRACTAL_Z_3D", "DHWCN"],
        "expect": RuntimeError,
        "format_expect": ["DHWCN"],
        "support_expect": False}

case1 = {"params": [{"shape": (120, 4, 16, 16), "dtype": "float16",
                     "ori_shape": (120, 4, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (3, 4, 5, 31, 64), "dtype": "float16",
                     "ori_shape": (3, 4, 5, 31, 64), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case2 = {"params": [{"shape": (60, 4, 16, 16), "dtype": "float16",
                     "ori_shape": (60, 4, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (3, 4, 5, 1, 64), "dtype": "float16",
                     "ori_shape": (3, 4, 5, 1, 64), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case3 = {"params": [{"shape": (72, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (72, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (4, 3, 2, 48, 16), "dtype": "float16",
                     "ori_shape": (4, 3, 2, 48, 16), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case4 = {"params": [{"shape": (32, 16, 16, 16), "dtype": "float16",
                     "ori_shape": (32, 16, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (2, 2, 4, 17, 256), "dtype": "float16",
                     "ori_shape": (2, 2, 4, 17, 256), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case5 = {"params": [{"shape": (24, 128, 16, 16), "dtype": "float16",
                     "ori_shape": (24, 128, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (2, 1, 4, 48, 2048), "dtype": "float16",
                     "ori_shape": (2, 1, 4, 48, 2048), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case6 = {"params": [{"shape": (16, 510, 16, 16), "dtype": "float16",
                     "ori_shape": (16, 510, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (2, 1, 4, 32, 8160), "dtype": "float16",
                     "ori_shape": (2, 1, 4, 32, 8160), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case7 = {"params": [{"shape": (12, 3, 16, 16), "dtype": "float16",
                     "ori_shape": (12, 3, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (3, 1, 2, 30, 46), "dtype": "float16",
                     "ori_shape": (3, 1, 2, 30, 46), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case8 = {"params": [{"shape": (6, 1, 16, 16), "dtype": "float16",
                     "ori_shape": (6, 1, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (3, 1, 2, 14, 15), "dtype": "float16",
                     "ori_shape": (3, 1, 2, 14, 15), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case9 = {"params": [{"shape": (45, 4, 16, 16), "dtype": "float16",
                     "ori_shape": (45, 4, 16, 16), "format": "FRACTAL_Z_3D",
                     "ori_format": "FRACTAL_Z_3D"},
                    {"shape": (3, 1, 5, 45, 61), "dtype": "float16",
                     "ori_shape": (3, 1, 5, 45, 61), "format": "DHWCN", "ori_format": "DHWCN"},
                    "FRACTAL_Z_3D", "DHWCN"],
         "expect": "success",
         "format_expect": ["DHWCN"],
         "support_expect": True}

case10 = {"params": [{"shape": (48, 4, 16, 16), "dtype": "float16",
                      "ori_shape": (48, 4, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2, 1, 3, 125, 61), "dtype": "float16",
                      "ori_shape": (2, 1, 3, 125, 61), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case11 = {"params": [{"shape": (528, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (528, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2, 1, 3, 1394, 3), "dtype": "float16",
                      "ori_shape": (2, 1, 3, 1394, 3), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case12 = {"params": [{"shape": (576, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (576, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (3, 1, 2, 1531, 1), "dtype": "float16",
                      "ori_shape": (3, 1, 2, 1531, 1), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case13 = {"params": [{"shape": (18, 21, 16, 16), "dtype": "float16",
                      "ori_shape": (18, 21, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2, 1, 3, 46, 334), "dtype": "float16",
                      "ori_shape": (2, 1, 3, 46, 334), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case14 = {"params": [{"shape": (6, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (6, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (3, 1, 2, 2, 3), "dtype": "float16",
                      "ori_shape": (3, 1, 2, 2, 3), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case15 = {"params": [{"shape": (831, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (831, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (3, 1, 277, 3, 3), "dtype": "float16",
                      "ori_shape": (3, 1, 277, 3, 3), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case16 = {"params": [{"shape": (8925, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (8925, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (35, 1, 5, 816, 1), "dtype": "float16",
                      "ori_shape": (35, 1, 5, 816, 1), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case17 = {"params": [{"shape": (8925, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (8925, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (35, 1, 5, 813, 16), "dtype": "float16",
                      "ori_shape": (35, 1, 5, 813, 16), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case18 = {"params": [{"shape": (1542, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (1542, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (3, 1, 2, 4112, 16), "dtype": "float16",
                      "ori_shape": (3, 1, 2, 4112, 16), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case19 = {"params": [{"shape": (6, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (6, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (3, 1, 2, 2, 4), "dtype": "float16",
                      "ori_shape": (3, 1, 2, 2, 4), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case20 = {"params": [{"shape": (5152, 3, 16, 16), "dtype": "float16",
                      "ori_shape": (5152, 3, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2576, 1, 1, 19, 48), "dtype": "float16",
                      "ori_shape": (2576, 1, 1, 19, 48), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case21 = {"params": [{"shape": (30912, 3, 16, 16), "dtype": "float16",
                      "ori_shape": (30912, 3, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2576, 2, 3, 19, 48), "dtype": "float16",
                      "ori_shape": (2576, 2, 3, 19, 48), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case22 = {"params": [{"shape": (10304, 3, 16, 16), "dtype": "float16",
                      "ori_shape": (10304, 3, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2, 2576, 1, 19, 48), "dtype": "float16",
                      "ori_shape": (2, 2576, 1, 19, 48), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case23 = {"params": [{"shape": (5152, 3, 16, 16), "dtype": "float16",
                      "ori_shape": (5152, 3, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2576, 1, 1, 19, 33), "dtype": "float16",
                      "ori_shape": (2576, 1, 1, 19, 33), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case24 = {"params": [{"shape": (10304, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (10304, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2576, 2, 1, 19, 15), "dtype": "float16",
                      "ori_shape": (2576, 2, 1, 19, 15), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case25 = {"params": [{"shape": (90, 17, 16, 16), "dtype": "float16",
                      "ori_shape": (90, 17, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (3, 1, 10, 48, 272), "dtype": "float16",
                      "ori_shape": (3, 1, 10, 48, 272), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case26 = {"params": [{"shape": (45, 17, 16, 16), "dtype": "float16",
                      "ori_shape": (45, 17, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (3, 1, 5, 48, 272), "dtype": "float16",
                      "ori_shape": (3, 1, 5, 48, 272), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case27 = {"params": [{"shape": (5055, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (5055, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (337, 3, 5, 1, 1), "dtype": "float16",
                      "ori_shape": (337, 3, 5, 1, 1), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case28 = {"params": [{"shape": (41472, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (41472, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (36, 96, 12, 1, 1), "dtype": "float16",
                      "ori_shape": (36, 96, 12, 1, 1), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case29 = {"params": [{"shape": (1248, 136, 16, 16), "dtype": "float16",
                      "ori_shape": (1248, 136, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (8, 3, 52, 1, 2162), "dtype": "float16",
                      "ori_shape": (8, 3, 52, 1, 2162), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case30 = {"params": [{"shape": (1, 2097152, 16, 16), "dtype": "float16",
                      "ori_shape": (1, 2097152, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (1, 1, 1, 3, 33554431), "dtype": "float16",
                      "ori_shape": (1, 1, 1, 3, 33554431), "format": "DHWCN",
                      "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case31 = {"params": [{"shape": (2, 1, 16, 16), "dtype": "float32",
                      "ori_shape": (2, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (1, 1, 1, 32, 16), "dtype": "float32",
                      "ori_shape": (1, 1, 1, 32, 16), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case32 = {"params": [{"shape": (9, 1, 16, 16), "dtype": "float32",
                      "ori_shape": (9, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (1, 3, 3, 1, 7), "dtype": "float32",
                      "ori_shape": (1, 3, 3, 1, 7), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case33 = {"params": [{"shape": (18, 2, 16, 16), "dtype": "float32",
                      "ori_shape": (18, 2, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (1, 3, 3, 32, 32), "dtype": "float32",
                      "ori_shape": (1, 3, 3, 32, 32), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case34 = {"params": [{"shape": (288, 16, 16, 16), "dtype": "float32",
                      "ori_shape": (288, 16, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2, 3, 3, 256, 256), "dtype": "float32",
                      "ori_shape": (2, 3, 3, 256, 256), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case35 = {"params": [{"shape": (24, 1, 16, 16), "dtype": "float32",
                      "ori_shape": (24, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (2, 3, 4, 1, 1), "dtype": "float32",
                      "ori_shape": (2, 3, 4, 1, 1), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case36 = {"params": [{"shape": (18, 2, 16, 16), "dtype": "float32",
                      "ori_shape": (18, 2, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (1, 3, 3, 32, 31), "dtype": "float32",
                      "ori_shape": (1, 3, 3, 32, 31), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

case37 = {"params": [{"shape": (18, 1, 16, 16), "dtype": "float32",
                      "ori_shape": (18, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D"},
                     {"shape": (1, 3, 3, 17, 1), "dtype": "float32",
                      "ori_shape": (1, 3, 3, 17, 1), "format": "DHWCN", "ori_format": "DHWCN"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "format_expect": ["DHWCN"],
          "support_expect": True}

ut_case.add_case(["Ascend910"], err1)
ut_case.add_case(["Ascend910"], err2)
ut_case.add_case(["Ascend910"], err3)
ut_case.add_case(["Ascend910"], err4)
ut_case.add_case(["Ascend910"], err5)
ut_case.add_case(["Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
ut_case.add_case(["Ascend910"], case4)
ut_case.add_case(["Ascend910"], case5)
ut_case.add_case(["Ascend910"], case6)
ut_case.add_case(["Ascend910"], case7)
ut_case.add_case(["Ascend910"], case8)
ut_case.add_case(["Ascend910"], case9)
ut_case.add_case(["Ascend910"], case10)
ut_case.add_case(["Ascend910"], case11)
ut_case.add_case(["Ascend910"], case12)
ut_case.add_case(["Ascend910"], case13)
ut_case.add_case(["Ascend910"], case14)
ut_case.add_case(["Ascend910"], case15)
ut_case.add_case(["Ascend910"], case16)
ut_case.add_case(["Ascend910"], case17)
ut_case.add_case(["Ascend910"], case18)
ut_case.add_case(["Ascend910"], case19)
ut_case.add_case(["Ascend910"], case20)
ut_case.add_case(["Ascend910"], case21)
ut_case.add_case(["Ascend910"], case22)
ut_case.add_case(["Ascend910"], case23)
ut_case.add_case(["Ascend910"], case24)
ut_case.add_case(["Ascend910"], case25)
ut_case.add_case(["Ascend910"], case26)
ut_case.add_case(["Ascend910"], case27)
ut_case.add_case(["Ascend910"], case28)
ut_case.add_case(["Ascend910"], case29)
ut_case.add_case(["Ascend910"], case30)
ut_case.add_case(["Ascend910"], case31)
ut_case.add_case(["Ascend910"], case32)
ut_case.add_case(["Ascend910"], case33)
ut_case.add_case(["Ascend910"], case34)
ut_case.add_case(["Ascend910"], case35)
ut_case.add_case(["Ascend910"], case36)
ut_case.add_case(["Ascend910"], case37)

# add precision case
case_1 = {"params": [{"shape": (120, 4, 16, 16), "dtype": "float16",
                      "ori_shape": (120, 4, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (3, 4, 5, 31, 64), "dtype": "float16",
                      "ori_shape": (3, 4, 5, 31, 64), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_2 = {"params": [{"shape": (60, 4, 16, 16), "dtype": "float16",
                      "ori_shape": (60, 4, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (3, 4, 5, 1, 64), "dtype": "float16",
                      "ori_shape": (3, 4, 5, 1, 64), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_3 = {"params": [{"shape": (72, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (72, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (4, 3, 2, 48, 16), "dtype": "float16",
                      "ori_shape": (4, 3, 2, 48, 16), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_4 = {"params": [{"shape": (32, 16, 16, 16), "dtype": "float16",
                      "ori_shape": (32, 16, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (2, 2, 4, 17, 256), "dtype": "float16",
                      "ori_shape": (2, 2, 4, 17, 256), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_5 = {"params": [{"shape": (24, 128, 16, 16), "dtype": "float16",
                      "ori_shape": (24, 128, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (2, 1, 4, 48, 2048), "dtype": "float16",
                      "ori_shape": (2, 1, 4, 48, 2048), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_6 = {"params": [{"shape": (16, 510, 16, 16), "dtype": "float16",
                      "ori_shape": (16, 510, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (2, 1, 4, 32, 8160), "dtype": "float16",
                      "ori_shape": (2, 1, 4, 32, 8160), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_7 = {"params": [{"shape": (12, 3, 16, 16), "dtype": "float16",
                      "ori_shape": (12, 3, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (3, 1, 2, 30, 46), "dtype": "float16",
                      "ori_shape": (3, 1, 2, 30, 46), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_8 = {"params": [{"shape": (6, 1, 16, 16), "dtype": "float16",
                      "ori_shape": (6, 1, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (3, 1, 2, 14, 15), "dtype": "float16",
                      "ori_shape": (3, 1, 2, 14, 15), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_9 = {"params": [{"shape": (45, 4, 16, 16), "dtype": "float16",
                      "ori_shape": (45, 4, 16, 16), "format": "FRACTAL_Z_3D",
                      "ori_format": "FRACTAL_Z_3D",
                      "param_type": "input", "value_range": [-10.0, 10.0]},
                     {"shape": (3, 1, 5, 45, 61), "dtype": "float16",
                      "ori_shape": (3, 1, 5, 45, 61), "format": "DHWCN", "ori_format": "DHWCN",
                      "param_type": "output"},
                     "FRACTAL_Z_3D", "DHWCN"],
          "expect": "success",
          "calc_expect_func": calc_expect_func,
          "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_10 = {"params": [{"shape": (48, 4, 16, 16), "dtype": "float16",
                       "ori_shape": (48, 4, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2, 1, 3, 125, 61), "dtype": "float16",
                       "ori_shape": (2, 1, 3, 125, 61), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_11 = {"params": [{"shape": (528, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (528, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2, 1, 3, 1394, 3), "dtype": "float16",
                       "ori_shape": (2, 1, 3, 1394, 3), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_12 = {"params": [{"shape": (576, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (576, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (3, 1, 2, 1531, 1), "dtype": "float16",
                       "ori_shape": (3, 1, 2, 1531, 1), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_13 = {"params": [{"shape": (18, 21, 16, 16), "dtype": "float16",
                       "ori_shape": (18, 21, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2, 1, 3, 46, 334), "dtype": "float16",
                       "ori_shape": (2, 1, 3, 46, 334), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_14 = {"params": [{"shape": (6, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (6, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (3, 1, 2, 2, 3), "dtype": "float16",
                       "ori_shape": (3, 1, 2, 2, 3), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_15 = {"params": [{"shape": (831, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (831, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (3, 1, 277, 3, 3), "dtype": "float16",
                       "ori_shape": (3, 1, 277, 3, 3), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_16 = {"params": [{"shape": (8925, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (8925, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (35, 1, 5, 816, 1), "dtype": "float16",
                       "ori_shape": (35, 1, 5, 816, 1), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_17 = {"params": [{"shape": (8925, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (8925, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (35, 1, 5, 813, 16), "dtype": "float16",
                       "ori_shape": (35, 1, 5, 813, 16), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_18 = {"params": [{"shape": (1542, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (1542, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (3, 1, 2, 4112, 16), "dtype": "float16",
                       "ori_shape": (3, 1, 2, 4112, 16), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_19 = {"params": [{"shape": (6, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (6, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (3, 1, 2, 2, 4), "dtype": "float16",
                       "ori_shape": (3, 1, 2, 2, 4), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_20 = {"params": [{"shape": (5152, 3, 16, 16), "dtype": "float16",
                       "ori_shape": (5152, 3, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2576, 1, 1, 19, 48), "dtype": "float16",
                       "ori_shape": (2576, 1, 1, 19, 48), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_21 = {"params": [{"shape": (30912, 3, 16, 16), "dtype": "float16",
                       "ori_shape": (30912, 3, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2576, 2, 3, 19, 48), "dtype": "float16",
                       "ori_shape": (2576, 2, 3, 19, 48), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_22 = {"params": [{"shape": (10304, 3, 16, 16), "dtype": "float16",
                       "ori_shape": (10304, 3, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2, 2576, 1, 19, 48), "dtype": "float16",
                       "ori_shape": (2, 2576, 1, 19, 48), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_23 = {"params": [{"shape": (5152, 3, 16, 16), "dtype": "float16",
                       "ori_shape": (5152, 3, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2576, 1, 1, 19, 33), "dtype": "float16",
                       "ori_shape": (2576, 1, 1, 19, 33), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_24 = {"params": [{"shape": (10304, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (10304, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2576, 2, 1, 19, 15), "dtype": "float16",
                       "ori_shape": (2576, 2, 1, 19, 15), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_25 = {"params": [{"shape": (90, 17, 16, 16), "dtype": "float16",
                       "ori_shape": (90, 17, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (3, 1, 10, 48, 272), "dtype": "float16",
                       "ori_shape": (3, 1, 10, 48, 272), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_26 = {"params": [{"shape": (45, 17, 16, 16), "dtype": "float16",
                       "ori_shape": (45, 17, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (3, 1, 5, 48, 272), "dtype": "float16",
                       "ori_shape": (3, 1, 5, 48, 272), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_27 = {"params": [{"shape": (5055, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (5055, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (337, 3, 5, 1, 1), "dtype": "float16",
                       "ori_shape": (337, 3, 5, 1, 1), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_28 = {"params": [{"shape": (41472, 1, 16, 16), "dtype": "float16",
                       "ori_shape": (41472, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (36, 96, 12, 1, 1), "dtype": "float16",
                       "ori_shape": (36, 96, 12, 1, 1), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_29 = {"params": [{"shape": (1248, 136, 16, 16), "dtype": "float16",
                       "ori_shape": (1248, 136, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (8, 3, 52, 1, 2162), "dtype": "float16",
                       "ori_shape": (8, 3, 52, 1, 2162), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_30 = {"params": [{"shape": (1, 2097152, 16, 16), "dtype": "float16",
                       "ori_shape": (1, 2097152, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (1, 1, 1, 3, 33554431), "dtype": "float16",
                       "ori_shape": (1, 1, 1, 3, 33554431), "format": "DHWCN",
                       "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_31 = {"params": [{"shape": (2, 1, 16, 16), "dtype": "float32",
                       "ori_shape": (2, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (1, 1, 1, 32, 16), "dtype": "float32",
                       "ori_shape": (1, 1, 1, 32, 16), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_32 = {"params": [{"shape": (9, 1, 16, 16), "dtype": "float32",
                       "ori_shape": (9, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (1, 3, 3, 1, 7), "dtype": "float32",
                       "ori_shape": (1, 3, 3, 1, 7), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_33 = {"params": [{"shape": (18, 2, 16, 16), "dtype": "float32",
                       "ori_shape": (18, 2, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (1, 3, 3, 32, 32), "dtype": "float32",
                       "ori_shape": (1, 3, 3, 32, 32), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_34 = {"params": [{"shape": (288, 16, 16, 16), "dtype": "float32",
                       "ori_shape": (288, 16, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2, 3, 3, 256, 256), "dtype": "float32",
                       "ori_shape": (2, 3, 3, 256, 256), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_35 = {"params": [{"shape": (24, 1, 16, 16), "dtype": "float32",
                       "ori_shape": (24, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (2, 3, 4, 1, 1), "dtype": "float32",
                       "ori_shape": (2, 3, 4, 1, 1), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_36 = {"params": [{"shape": (18, 2, 16, 16), "dtype": "float32",
                       "ori_shape": (18, 2, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (1, 3, 3, 32, 31), "dtype": "float32",
                       "ori_shape": (1, 3, 3, 32, 31), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

case_37 = {"params": [{"shape": (18, 1, 16, 16), "dtype": "float32",
                       "ori_shape": (18, 1, 16, 16), "format": "FRACTAL_Z_3D",
                       "ori_format": "FRACTAL_Z_3D",
                       "param_type": "input", "value_range": [-10.0, 10.0]},
                      {"shape": (1, 3, 3, 17, 1), "dtype": "float32",
                       "ori_shape": (1, 3, 3, 17, 1), "format": "DHWCN", "ori_format": "DHWCN",
                       "param_type": "output"},
                      "FRACTAL_Z_3D", "DHWCN"],
           "expect": "success",
           "calc_expect_func": calc_expect_func,
           "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_precision_case(["Ascend910"], case_1)
ut_case.add_precision_case(["Ascend910"], case_2)
ut_case.add_precision_case(["Ascend910"], case_3)
ut_case.add_precision_case(["Ascend910"], case_4)
ut_case.add_precision_case(["Ascend910"], case_5)
ut_case.add_precision_case(["Ascend910"], case_6)
ut_case.add_precision_case(["Ascend910"], case_7)
ut_case.add_precision_case(["Ascend910"], case_8)
ut_case.add_precision_case(["Ascend910"], case_9)
ut_case.add_precision_case(["Ascend910"], case_10)
ut_case.add_precision_case(["Ascend910"], case_11)
ut_case.add_precision_case(["Ascend910"], case_12)
ut_case.add_precision_case(["Ascend910"], case_13)
ut_case.add_precision_case(["Ascend910"], case_14)
ut_case.add_precision_case(["Ascend910"], case_15)
# ut_case.add_precision_case(["Ascend910"], case_16)
# ut_case.add_precision_case(["Ascend910"], case_17)
# ut_case.add_precision_case(["Ascend910"], case_18)
# ut_case.add_precision_case(["Ascend910"], case_19)
# ut_case.add_precision_case(["Ascend910"], case_20)
# ut_case.add_precision_case(["Ascend910"], case_21)
# ut_case.add_precision_case(["Ascend910"], case_22)
# ut_case.add_precision_case(["Ascend910"], case_23)
# ut_case.add_precision_case(["Ascend910"], case_24)
# ut_case.add_precision_case(["Ascend910"], case_25)
# ut_case.add_precision_case(["Ascend910"], case_26)
# ut_case.add_precision_case(["Ascend910"], case_27)
# ut_case.add_precision_case(["Ascend910"], case_28)
# ut_case.add_precision_case(["Ascend910"], case_29)
# ut_case.add_precision_case(["Ascend910"], case_30)
ut_case.add_precision_case(["Ascend910"], case_31)
ut_case.add_precision_case(["Ascend910"], case_32)
ut_case.add_precision_case(["Ascend910"], case_33)
ut_case.add_precision_case(["Ascend910"], case_34)
ut_case.add_precision_case(["Ascend910"], case_35)
ut_case.add_precision_case(["Ascend910"], case_36)
ut_case.add_precision_case(["Ascend910"], case_37)

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    exit(0)
