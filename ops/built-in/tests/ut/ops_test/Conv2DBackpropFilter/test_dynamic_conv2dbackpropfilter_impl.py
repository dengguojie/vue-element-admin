#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2DBackpropFilter", "impl.dynamic.conv2d_backprop_filter",
               "conv2d_backprop_filter")

def gen_dynamic_conv2d_backprop_filter_case(shape_filter, shape_dedy, shape_x,
                                            dtype_filter, dtype_dedy, dtype_x,
                                            format_filter, format_dedy, format_x,
                                            range_filter, range_dedy, range_x,
                                            strides, pads, dilations,
                                            kernel_name_val, expect):
    return {"params": [
        # x
        {"ori_shape": shape_x, "ori_format": format_x,
         "dtype": dtype_x, "range": range_x, "shape": shape_x, "format": format_x},
        # filter_size
        {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), 'dtype': "float16", "format": 'NC1HWC0', "ori_format": 'NCHW'},
        # dedy
        {"ori_shape": shape_dedy, "ori_format": format_dedy,
         "dtype": dtype_dedy, "range": range_dedy, "shape": shape_dedy, "format": format_dedy},
        # filter
        {"ori_shape": shape_filter, "ori_format": format_filter,
         "dtype": dtype_filter, "range": range_filter, "shape": shape_filter, "format": format_filter},
        strides, pads, dilations, 1, "NCHW"],
        "case_name": kernel_name_val,
        "expect": expect
        }

# dynamic_hw
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1",
                                        "success"))

#dynamic_batch
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case((64, 3, 3, 3), (-1, 64, 112, 112), (-1, 3, 112, 112),
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, 158), (64, 64), (112, 112), (112, 112)],
                                        [(128, 158), (3, 3), (112, 112), (112, 112)],
                                        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case2",
                                        "success"))

# dynamic_rank
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [-2], [-2],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [],
                                        [],
                                        (1, 1, 2, 2), [-1, -1, -1, -1], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case3",
                                        "success"))

#dynamic_nwc
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 3, 3, 3], [-1, -1, 112, -1], [-1, -1, 112, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, 158), (64, 128), (112, 112), (112, 142)],
                                        [(128, 158), (1, 16), (112, 112), (112, 142)],
                                        (1, 1, 1, 1), (-1, -1, -1, -1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case4",
                                        "success"))

# dynamic_hw, x shape is not list or tuple
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], "[128, 64, -1, -1]",
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_1",
                                        RuntimeError))

# dynamic_hw, dedy shape is less than 4
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_2",
                                        RuntimeError))

# dynamic_hw, dedy format is not in ["NHWC", "NCHW"]
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "HWCN", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_3",
                                        RuntimeError))

# dynamic_hw, dedy[0] is not equal to x[0]
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [64, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_4",
                                        RuntimeError))

# dynamic_hw, x[2] is not equal to -1
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, 1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_5",
                                        RuntimeError))

# dynamic_hw, stride dim is not 4
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_6",
                                        RuntimeError))

# dynamic_hw, pad_up greater than filter_h_dilation
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 2, 2], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_8",
                                        RuntimeError))

# dynamic_hw, pad_left greater than filter_w_dilation
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [2, 2, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_9",
                                        RuntimeError))

# dynamic_hw, dilationh greater than 256
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 256, 1),
                                        "dynamic_conv2d_backprop_filter_case1_10",
                                        RuntimeError))

# dynamic_hw, x hw upper range's product greater than 2^63-1
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 4294967296), (112, 4294967296)],
                                        (1, 1, 256, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_11",
                                        RuntimeError))

# dynamic_hw, dilationN greater than 1
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (256, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_12",
                                        RuntimeError))

# dynamic_hw, filter_h_dilation > upper_fmap_h_padding
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 145, 1], [128, 66, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (145, 145), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_13",
                                        RuntimeError))

# dynamic_hw, filter_w_dilation > upper_fmap_w_padding
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 145], [128, 66, -1, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (145, 145)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_14",
                                        RuntimeError))
# dynamic_hw, pads error
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([456, 85, 10, 10], [17, 456, -1, -1], [17, 85, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(456, 456), (85, 85), (10, 10), (10, 10)],
                                        [(17, 17), (456, 456), (74, 98), (4, 20)],
                                        [(17, 17), (85, 85), (336, 406), (4, 32)],
                                        (1, 1, 4, 4), [3, 3, 3, 3], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1_15",
                                        RuntimeError))

# dynamic_batch, x[2] is equal to -1
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 3, 3, 3], [-1, 64, 112, 112], [-1, 3, -1, 112],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, 158), (64, 64), (112, 112), (112, 112)],
                                        [(128, 158), (3, 3), (112, 112), (112, 112)],
                                        (1, 1, 1, 1), [-1, -1, -1, -1], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case2_1",
                                        RuntimeError))

# dynamic_h in x, dynamic_hw in dedy
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, -1, 120],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (120, 120)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case3_1",
                                        "success"))

# dynamic_w in x, dynamic_hw in dedy
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [128, 64, 120, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (120, 120), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case3_2",
                                        "success"))

# dynamic_hw in x, dynamic_w in dedy
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, 60, -1], [128, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (60, 60), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case3_3",
                                        "success"))

# dynamic_nhw in x, dynamic_hw in dedy
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 64, 1, 1], [128, 64, -1, -1], [-1, 64, -1, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [(128, 128), (64, 64), (56, 71), (56, 71)],
                                        [(128, 128), (64, 64), (112, 142), (112, 142)],
                                        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case3_3",
                                        "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
