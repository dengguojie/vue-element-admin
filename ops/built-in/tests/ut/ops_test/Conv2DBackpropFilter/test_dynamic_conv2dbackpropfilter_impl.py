#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.dynamic.conv2d_backprop_filter import get_op_support_info
ut_case = OpUT("Conv2DBackpropFilter", "impl.dynamic.conv2d_backprop_filter",
               "conv2d_backprop_filter")

def gen_dynamic_conv2d_backprop_filter_case(shape_filter, shape_dedy, shape_x,
                                            dtype_filter, dtype_dedy, dtype_x,
                                            format_filter, format_dedy, format_x,
                                            range_filter, range_dedy, range_x,
                                            strides, pads, dilations,
                                            kernel_name_val, expect):
    import sys
    import importlib
    modulename = sys.modules.get('impl.dynamic.conv2d_backprop_filter')
    importlib.reload(modulename)
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
        strides, pads, dilations, 1, format_x],
        "case_name": kernel_name_val,
        "expect": expect
        }

# dynamic_nhw VALID NHWC 
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([3, 3, 64, 64], [-1, -1, -1, 64], [-1, -1, -1, 64],
                                        "float32", "float16", "float16",
                                        "HWCN", "NHWC", "NHWC",
                                        [(1, 1), (1, 1), (64, 64), (64, 64)],
                                        [(64, 128), (50, 71), (50, 71), (64, 64)],
                                        [(64, 128), (56, 142), (112, 142), (64, 64)],
                                        (1, 1, 2, 1), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1",
                                        "success"))

#dynamic_batch SAME NCHW
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case((64, 3, 3, 3), (-1, 64, 1, 2), (-1, 3, 1, 2),
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, 158), (64, 64), (1, 1), (2, 2)],
                                        [(128, 158), (3, 3), (1, 1), (2, 2)],
                                        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case2",
                                        "success"))

# w_one_case dynamic_batch SAME NCHW
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case((64, 3, 3, 3), (-1, 64, 3, 1), (-1, 3, 3, 1),
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, 158), (64, 64), (3, 3), (1, 1)],
                                        [(128, 158), (3, 3), (3, 3), (1, 1)],
                                        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case_w_one_case",
                                        "success"))

# w_one_case range None
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case((64, 3, 3, 3), (-1, 64, 3, 1), (-1, 3, 3, 1),
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, None), (64, 64), (3, 3), (1, 1)],
                                        [(128, None), (3, 3), (3, 3), (1, 1)],
                                        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case_w_one_case_range_none",
                                        "success"))

# dynamic_rank
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([1, 1, 64, 64], [-2], [-2],
                                        "float32", "float16", "float16",
                                        "HWCN", "NHWC", "NHWC",
                                        [(64, 64), (64, 64), (1, 1), (1, 1)],
                                        [],
                                        [],
                                        (1, 1, 2, 2), [-1, -1, -1, -1], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case3",
                                        "success"))

#dynamic_nwc SAME NCHW range None
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([64, 3, 3, 3], [-1, -1, 112, -1], [-1, -1, 112, -1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, None), (64, 128), (112, 112), (112, None)],
                                        [(128, None), (1, 16), (112, 112), (112, None)],
                                        (1, 1, 1, 1), (-1, -1, -1, -1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case4",
                                        "success"))

# dynamic hw SAME NHWC
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([3, 3, 64, 64], [7, -1, -1, 64], [7, -1, -1, 64],
                                        "float32", "float16", "float16",
                                        "HWCN", "NHWC", "NHWC",
                                        [(3, 3), (3, 3), (64, 64), (64, 64)],
                                        [(7, 7), (1, 71), (1, 71), (64, 64)],
                                        [(7, 7), (1, 142), (1, 142), (64, 64)],
                                        (1, 1, 2, 1), [-1, -1, -1, -1], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case5",
                                        "success"))

# dynamic h VALID NHWC 
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([2, 2, 64, 64], [7, -1, 2, 64], [7, -1, 3, 64],
                                        "float32", "float16", "float16",
                                        "HWCN", "NHWC", "NHWC",
                                        [(2, 2), (2, 2), (64, 64), (64, 64)],
                                        [(7, 7), (2, 71), (2, 2), (64, 64)],
                                        [(7, 7), (2, 142), (3, 3), (64, 64)],
                                        (1, 2, 1, 1), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case6",
                                        "success"))

# dynamic_nh hw VALID NHWC 
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([3, 3, 64, 64], [-1, -1, 50, 64], [64, -1, -1, 64],
                                        "float32", "float16", "float16",
                                        "HWCN", "NHWC", "NHWC",
                                        [(1, 1), (1, 1), (64, 64), (64, 64)],
                                        [(64, 128), (50, 71), (50, 71), (64, 64)],
                                        [(64, 128), (56, 142), (112, 142), (64, 64)],
                                        (1, 1, 2, 1), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case7",
                                        "success"))

#dynamic_w hw SAME NCHW
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case((64, 3, 3, 3), (128, 64, 1, -1), (128, 3, -1, -1),
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, 158), (64, 64), (1, 1), (2, 4)],
                                        [(128, 158), (3, 3), (1, 10), (2, 6)],
                                        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case8",
                                        "success"))

#dynamic_hw w SAME NCHW
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case((64, 3, 3, 3), (128, 64, -1, -1), (128, 3, 1, -1),
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(64, 64), (3, 3), (3, 3), (3, 3)],
                                        [(128, 158), (64, 64), (1, 10), (2, 4)],
                                        [(128, 158), (3, 3), (1, 1), (2, 8)],
                                        (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case9",
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

ut_case.add_case(
    "all",
    {"params": [
        # x
        {"ori_shape": [-1, 64, -1, -1], "ori_format": 'NCHW',
         "dtype": "float16", "range": [(128, 128), (4, 4), (112, 142), (112, 142), (16, 16)], "shape": [-1, 4, -1, -1, 16], "format": 'NC1HWC0'},
        # filter_size
        {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), 'dtype': "float16", "format": 'NC1HWC0', "ori_format": 'NCHW'},
        # dedy
        {"ori_shape": [128, 64, -1, -1], "ori_format": 'NCHW',
         "dtype": "float16", "range": [(128, 128), (4, 4), (56, 71), (56, 71), (16, 16)], "shape": [128, 4, -1, -1, 16], "format": 'NC1HWC0'},
        # filter
        {"ori_shape": [64, 64, 1, 1], "ori_format": 'NCHW',
         "dtype": 'float32', "range": [(64, 64), (64, 64), (1, 1), (1, 1)], "shape": [64, 64, 1, 1], "format": 'NCHW'},
        (1, 1, 2, 2), [0, 0, 0, 0], (1, 1, 1, 1), 1, 'NCHW'],
        "case_name": 'dynamic_conv2d_backprop_filter_case3_4',
        "expect": 'success'
        })

# h = 1 and w = 1
ut_case.add_case(
    "all",
    gen_dynamic_conv2d_backprop_filter_case([1024, 1024, 1, 1], [-1, 1024, 1, 1], [-1, 1024, 1, 1],
                                        "float32", "float16", "float16",
                                        "NCHW", "NCHW", "NCHW",
                                        [(1024, 1024), (1024, 1024), (1, 1), (1, 1)],
                                        [(1, 16384), (1024, 1024), (1, 1), (1, 1)],
                                        [(1, 16384), (1024, 1024), (1, 1), (1, 1)],
                                        (1, 1, 1, 1), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case_hw1",
                                        "success"))

def test_conv2d_backprop_filter_fuzz_build_generalization(test_arg):
    from impl.dynamic.conv2d_backprop_filter import conv2d_bp_filter_generalization
    input_list = [
        {
            'shape': (16, 1, 16, 16, 16),
            'ori_shape': (16, 3, 16, 16),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            "range": ((16, 32), (1, 1), (16, 32), (16, 32), (16, 16)),
            "ori_range": ((16, 32), (3, 3), (16, 32), (16, 32))
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32',
            "range": [[4, 4]],
            "ori_range": ()
        }, {
            'shape': (16, 3, 14, 12, 16),
            'ori_shape': (16, 33, 14, 12),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            "range": ((16, 32), (3, 3), (4, 16), (4, 16), (16, 16)),
            "ori_range": ((16, 32), (33, 33), (4, 16), (4, 16))
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            "range": [[33, 33], [3, 3], [3, 3], [5, 5]],
            "ori_range": ()
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 'conv2d_backprop_filter_fuzz_build_generalization']
    conv2d_bp_filter_generalization(*input_list)
print("adding conv2d test_conv2d_backprop_filter_fuzz_build_generalization testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_backprop_filter_fuzz_build_generalization)


# modify max grade point, when exceed L1 size
def test_conv2d_backprop_filter_fuzz_build_generalization_1(test_arg):
    from impl.dynamic.conv2d_backprop_filter import conv2d_bp_filter_generalization
    input_list = [
        {
            'shape': (2, 1, 768, 1280, 16),
            'ori_shape': (2, 3, 768, 1280),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            "range": ((2, 4), (1, 1), (768, 1024), (1024, 4096), (16, 16)),
            "ori_range": ((2, 4), (3, 3), (768, 1024), (1024, 4096))
        }, {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32',
            "range": [[4, 4]],
            "ori_range": ()
        }, {
            'shape': (2, 4, 384, 640, 16),
            'ori_shape': (2, 64, 384, 640),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16',
            "range": ((2, 4), (4, 4), (256, 512), (512, 768), (16, 16)),
            "ori_range": ((2, 4), (64, 64), (256, 512), (512, 768))
        }, {
            'ori_shape': (64, 3, 7, 7),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16',
            "range": [[64, 64], [3, 3], [7, 7], [7, 7]],
            "ori_range": ()
        }, (1, 1, 2, 2), (2, 3, 2, 3), (1, 1, 1, 1), 1, 'NCHW', 'conv2d_backprop_filter_fuzz_build_generalization_1']
    conv2d_bp_filter_generalization(*input_list)
print("adding conv2d test_conv2d_backprop_filter_fuzz_build_generalization_1 testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_backprop_filter_fuzz_build_generalization_1)

def test_conv2d_backprop_filter_fuzz_build_tilingcase(test_arg):
    import json
    from impl.dynamic.conv2d_backprop_filter import conv2d_backprop_filter
    from tbe.common.context import get_context
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        get_context().set_build_type("fuzzily_build")
        get_context().add_addition("max_kernel_id", -1)
        missing_info = [{
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [16, 32],
                                        [3, 3],
                                        [256, 512],
                                        [256, 512]
                                    ],
                                    "shape": [-1, 3, -1, -1]
                                }]
                            }]
                        }, {
                            "inputs": [{
                                "index": 0,
                                "tensor": [{
                                    "range": [
                                        [16, 32],
                                        [3, 3],
                                        [512, 1024],
                                        [512, 1024]
                                    ],
                                    "shape": [-1, 3, -1, -1]
                                }]
                            }]
                        }]
        get_context().add_addition("missing_support_info", json.dumps(missing_info))
        input_list = [
            {
                'shape': (-1, 1, -1, -1, 16),
                'ori_shape': (-1, 3, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16',
                'range': ((16, 32), (1, 1), (256, 1024), (256, 1024), (16, 16))
            }, {
                'shape': (4,),
                'ori_shape': (4,),
                'ori_format': 'ND',
                'format': 'ND',
                'dtype': 'int32',
                'range': ()
            }, {
                'shape': (-1, 3, -1, -1, 16),
                'ori_shape': (-1, 33, -1, -1),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16',
                'range': ((16, 32), (3, 3), (256, 1024), (256, 1024), (16, 16))
            }, {
                'shape': (33, 3, 3, 5),
                'ori_shape': (33, 3, 3, 5),
                'ori_format': 'NCHW',
                'format': 'NCHW',
                'dtype': 'float32'
            }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW', 'test_conv2d_backprop_filter_fuzz_build_tilingcase']
        conv2d_backprop_filter(*input_list)
print("adding conv2d_backprop_filter test_conv2d_backprop_filter_fuzz_build_tilingcase testcase")
ut_case.add_cust_test_func(test_func=test_conv2d_backprop_filter_fuzz_build_tilingcase)


gen_dynamic_conv2d_backprop_filter_case([3, 3, 64, 64], [-1, -1, -1, 64], [-1, -1, -1, 64],
                                        "float32", "float16", "float16",
                                        "HWCN", "NHWC", "NHWC",
                                        [(1, 1), (1, 1), (64, 64), (64, 64)],
                                        [(64, 128), (50, 71), (50, 71), (64, 64)],
                                        [(64, 128), (56, 142), (112, 142), (64, 64)],
                                        (1, 1, 2, 1), [0, 0, 0, 0], (1, 1, 1, 1),
                                        "dynamic_conv2d_backprop_filter_case1",
                                        "success")

def test_get_op_support_info_dynamic_dw(test_arg):
    x = {"shape": (-1, 4, -1, -1, 16), 'ori_shape': (-1, -1, -1, 64),
         "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
         "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
        }
    out_backprop = {"shape":  (-1, 4, -1, -1, 16), 'ori_shape':(-1, -1, -1, 64),
                    "ori_format": "NHWC", "format": "NC1HWC0", "dtype": "float16",
                    "range": ((2, 4), (4, 4), (4, 8), (4, 8), (16, 16))
                   }
    y = {"shape":  (36, 4, 16, 16), 'ori_shape':(3, 3, 64, 64),
         "ori_format": "NHWC", "format": "FRATAL_NZ", "dtype": "float16",
         "range": ((36, 36), (4, 4), (16, 16), (16, 16))
        }
    get_op_support_info(x, (3, 3, 64, 64), out_backprop, y, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1))
ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_dw)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
