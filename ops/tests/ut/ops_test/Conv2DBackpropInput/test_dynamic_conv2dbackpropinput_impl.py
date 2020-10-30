#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2DBackpropInput", "impl.dynamic.conv2d_backprop_input",
               "conv2d_backprop_input")

def gen_dynamic_conv2d_backprop_input_case(shape_filter, shape_dedy, shape_dx,
                                           dtype_filter, dtype_dedy, dtype_dx,
                                           format_filter, format_dedy, format_dx,
                                           range_dedy, range_dx,
                                           strides, pads, dilations, groups,
                                           data_format, kernel_name_val, expect):
    return {"params": [
        # dx
        {"ori_shape": shape_dx, "ori_format": format_dx,
         "dtype": dtype_dx, "range": range_dx},
        # filter
        {"ori_shape": shape_filter, "ori_format": format_filter,
         "dtype": dtype_filter},
        # out_backprop
        {"ori_shape": shape_dedy,  "ori_format": format_dedy,
         "dtype": dtype_dedy, "range": range_dedy},
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
    gen_dynamic_conv2d_backprop_input_case([32, 16, 1, 1], [1, 32, -1, -1], [1, 16, -1, -1],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 1), (32, 32), (6, 26), (6, 26)),
                                        ((1, 1), (16, 16), (6, 26), (6, 26)),
                                        (1, 1, 1, 1), "SAME", (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case1",
                                        "success"))
# general, dynamic_hw
# ut_case.add_case(
#     "all",
#     gen_dynamic_conv2d_backprop_input_case([32, 16, 3, 3], [1, 32, -1, -1], [1, 16, -1, -1],
#                                         "float16", "float16", "float16",
#                                         "NCHW", "NCHW", "NCHW",
#                                         ((1, 1), (32, 32), (4, 24), (4, 24)),
#                                         ((1, 1), (16, 16), (6, 26), (6, 26)),
#                                         (1, 1, 1, 1), "VALID", (1, 1, 1, 1), 1, "NCHW",
#                                         "dynamic_conv2d_backprop_input_case2",
#                                         "success"))

# opti, dynamic_batch
# ut_case.add_case(
#     "all",
#     gen_dynamic_conv2d_backprop_input_case([32, 16, 1, 1], [-1, 32, 8, 8], [-1, 16, 16, 16],
#                                         "float16", "float16", "float16",
#                                         "NCHW", "NCHW", "NCHW",
#                                         ((1, 10), (32, 32), (8, 8), (16, 16)),
#                                         ((1, 10), (16, 16), (8, 8), (16, 16)),
#                                         (1, 1, 2, 2), "VALID", (1, 1, 1, 1), 1, "NCHW",
#                                         "dynamic_conv2d_backprop_input_case3",
#                                         "success"))
# general, dynamic_batch
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_input_case([32, 16, 3, 3], [-1, 32, 8, 8], [-1, 16, 16, 16],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 10), (32, 32), (8, 8), (16, 16)),
                                        ((1, 10), (16, 16), (8, 8), (16, 16)),
                                        (1, 1, 2, 2), "SAME", (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case4",
                                        "success"))

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)