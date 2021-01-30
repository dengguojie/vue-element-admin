#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Deconvolution", "impl.dynamic.deconvolution",
               "deconvolution")

def gen_dynamic_deconvolution_case(shape_filter, shape_dedy, shape_dx,
                                           dtype_filter, dtype_dedy, dtype_dx,
                                           format_filter, format_dedy, format_dx,
                                           range_dedy, range_dx,
                                           strides, pads, dilations, groups,
                                           data_format, kernel_name_val, expect):
    return {"params": [
        # out_backprop
        {"ori_shape": shape_dedy,  "ori_format": format_dedy,
         "dtype": dtype_dedy, "range": range_dedy},
        # filter
        {"ori_shape": shape_filter, "ori_format": format_filter,
         "dtype": dtype_filter},
        None, None,
        # y
        {"ori_shape": shape_dx, "ori_format": format_dx,
         "dtype": dtype_dx, "range": range_dx},
        strides, pads, dilations, groups, data_format],
        "case_name": kernel_name_val,
        "expect": expect
        }

# opti, dynamic_hw
ut_case.add_case(
    "all",
    gen_dynamic_deconvolution_case([128, 128, 3, 3], [1, 128, -1, -1], [1, 128, -1, -1],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 1), (128, 128), (1, 128), (1, 128)),
                                        ((1, 1), (128, 128), (4, 128), (4, 128)),
                                        (1, 1), [1, 1, 1, 1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_deconvolution_case1",
                                        "success"))

# opti, dynamic_hw
ut_case.add_case(
    "all",
    gen_dynamic_deconvolution_case([128, 128, 3, 3], [1, 128, -1, -1], [1, 128, -1, -1],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 1), (128, 128), (1, None), (1, None)),
                                        ((1, 1), (128, 128), (4, None), (4, None)),
                                        (1, 1), [1, 1, 1, 1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_deconvolution_case1",
                                        "success"))

# general, dynamic_hw
ut_case.add_case(
    "all",
    gen_dynamic_deconvolution_case([32, 16, 3, 3], [1, 32, -1, -1], [1, 16, -1, -1],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 1), (32, 32), (4, 24), (4, 24)),
                                        ((1, 1), (16, 16), (6, 26), (6, 26)),
                                        (1, 1), [0,0,0,0], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_deconvolution_case2",
                                        "success"))

# general, dynamic_hw
ut_case.add_case(
    "all",
    gen_dynamic_deconvolution_case([32, 16, 3, 3], [1, 32, -1, -1], [1, 16, -1, -1],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 1), (32, 32), (4, None), (4, None)),
                                        ((1, 1), (16, 16), (6, None), (6, None)),
                                        (1, 1), [0,0,0,0], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_deconvolution_case2",
                                        "success"))

# opti, dynamic_batch
ut_case.add_case(
    "all",
    gen_dynamic_deconvolution_case([32, 16, 1, 1], [-1, 32, 8, 8], [-1, 16, 16, 16],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 10), (32, 32), (8, 8), (16, 16)),
                                        ((1, 10), (16, 16), (8, 8), (16, 16)),
                                        (2, 2), [0,0,0,0], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_deconvolution_case3",
                                        "success"))
# general, dynamic_batch
ut_case.add_case(
    "all",
    gen_dynamic_deconvolution_case([128, 128, 3, 3], [-1, 128,28, 28], [-1, 128, 28, 28],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 10), (128, 128), (28, 28), (28, 28)),
                                        ((1, 10), (128, 128), (28, 28), (28, 28)),
                                        (1, 1), [1, 1, 1, 1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_deconvolution_case4",
                                        "success"))

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910A")
    exit(0)