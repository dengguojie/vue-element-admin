#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2DTranspose", "impl.dynamic.conv2d_transpose",
               "conv2d_transpose")

def gen_dynamic_conv2d_transpose_case(shape_filter, shape_dedy, shape_dx, shape_inputsize,
                                           dtype_filter, dtype_dedy, dtype_dx,
                                           format_filter, format_dedy, format_dx,
                                           range_dedy, range_dx, range_inputsize,
                                           strides, pads, dilations, groups,
                                           data_format, kernel_name_val, expect):
    return {"params": [
        # dx
        {"ori_shape": shape_inputsize, "ori_format": format_dx,
         "dtype": dtype_dx, "range": range_inputsize},
        # out_backprop
        {"ori_shape": shape_dedy,  "ori_format": format_dedy,
         "dtype": dtype_dedy, "range": range_dedy},
        # filter
        {"ori_shape": shape_filter, "ori_format": format_filter,
         "dtype": dtype_filter},
        #bias
        None,
        #offset_w
        None,
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
    gen_dynamic_conv2d_transpose_case([32, 16, 1, 1], [1, 32, -1, -1], [1, 16, -1, -1],[4],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 1), (32, 32), (6, 26), (6, 26)),
                                        ((1, 1), (16, 16), (6, 26), (6, 26)),
                                        ((4, 4)),
                                        (1, 1, 1, 1), [-1, -1, -1, -1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case1",
                                        "success"))

# opti, dynamic_hw, range(n,None)
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_transpose_case([32, 16, 1, 1], [1, 32, -1, -1], [1, 16, -1, -1],[4],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 1), (32, 32), (6, None), (6, None)),
                                        ((1, 1), (16, 16), (6, None), (6, None)),
                                        ((4, 4)),
                                        (1, 1, 1, 1), [-1, -1, -1, -1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case1",
                                        "success"))

# general, dynamic_batch
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_transpose_case([32, 16, 3, 3], [-1, 32, 8, 8], [-1, 16, 16, 16],[4],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 10), (32, 32), (8, 8), (16, 16)),
                                        ((1, 10), (16, 16), (8, 8), (16, 16)),
                                        ((4, 4)),
                                        (1, 1, 2, 2), [-1, -1, -1, -1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case4",
                                        "success"))

# general, dynamic_batch,y and data_format are not the same
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_transpose_case([32, 16, 3, 3], [-1, 32, 8, 8], [-1, 16, 16, 16],[4],
                                        "float16", "float16", "float16",
                                        "NCHW", "NHWC", "NCHW",
                                        ((1, 10), (32, 32), (8, 8), (16, 16)),
                                        ((1, 10), (16, 16), (8, 8), (16, 16)),
                                        ((4, 4)),
                                        (1, 1, 2, 2), [-1, -1, -1, -1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case4",
                                        "failed"))

# general, dynamic_batch,N dim's range is (1, None)
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_transpose_case([32, 16, 3, 3], [-1, 32, 8, 8], [-1, 16, 16, 16],[4],
                                        "float16", "float16", "float16",
                                        "NCHW", "NHWC", "NCHW",
                                        ((1, None), (32, 32), (8, 8), (16, 16)),
                                        ((1, None), (16, 16), (8, 8), (16, 16)),
                                        ((4, 4)),
                                        (1, 1, 2, 2), [-1, -1, -1, -1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case4",
                                        "failed"))


ut_case.add_case(
    "all",
    gen_dynamic_conv2d_transpose_case([32, 16, 3, 3], [-1, 32, 8, 8], [-1, 16, 16, 16],[-2],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((1, 10), (32, 32), (8, 8), (16, 16)),
                                        ((1, 10), (16, 16), (8, 8), (16, 16)),
                                        ((-2, -2)),
                                        (1, 1, 2, 2), [-1, -1, -1, -1], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case4",
                                        "failed"))

# general, dynamic_hw,hw dim's range is (2000, 2047)
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_transpose_case([32, 16, 5, 5], [4, 32, -1, -1], [4, 16, -1, -1],[4],
                                        "float16", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        ((4, 4), (32, 32), (2000, 2047), (2000, 2046)),
                                        ((4, 4), (16, 16), (4000, 4097), (4000, 4096)),
                                        ((4, 4)),
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1), 1, "NCHW",
                                        "dynamic_conv2d_backprop_input_case5",
                                        "success"))

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)