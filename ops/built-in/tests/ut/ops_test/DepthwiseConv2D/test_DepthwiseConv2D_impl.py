#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.depthwise_conv2d import get_op_support_info
ut_case = OpUT("DepthwiseConv2d", "impl.depthwise_conv2d", "depthwise_conv2d")

dp_conv2d_op_testcase = [
    # fp16 without bias
    ((32, 960, 7, 7), (3, 3, 960, 1), None, (32, 960, 7, 7), 1, 1, (1, 1, 1, 1), "NCHW", 0, "float16", "success"),
    ((16, 192, 28, 28), (5, 5, 192, 2), None, (16, 192, 14, 14), 2, 2, (2, 2, 2, 2), "NCHW", 0, "float16",
     RuntimeError),
    ((16, 192, 28, 28), (5, 5, 192, 1), None, (16, 192, 14, 14), 2, 2, (2, 2, 2, 2), "NCHW", 0, "float16",
     RuntimeError),
    # ((8, 96, 56, 56), (3, 3, 96, 1), None, (8, 96, 56, 56), 1, 1, (1, 1, 1, 1),
    #  "NCHW", 0, "float16", "success"),
    # ((1, 32, 112, 112), (3, 3, 32, 1), None, (1, 32, 56, 56), 2, 1,
    #  (1, 1, 1, 1), "NCHW", 0, "float16", "success"),

    # ((2, 336, 504, 256), (3, 3, 256, 1), None, (2, 334, 502, 256), 1, 1,
    #  (0, 0, 0, 0), "NHWC", 0, "float16", "success"),
    # ((648, 4, 5, 2048), (3, 3, 2048, 1), None, (648, 2, 3, 2048), 1, 1,
    #  (0, 0, 0, 0), "NHWC", 0, "float16", "success"),
    # ((2, 28, 28, 240), (5, 5, 240, 1), None, (2, 14, 14, 240), 2, 1,
    #  (2, 2, 2, 2), "NHWC", 0, "float16", "success"),

    # fp16 with bias
    ((32, 960, 7, 7), (3, 3, 960, 1), (1, 960, 1, 1), (32, 960, 7, 7), 1, 1, (1, 1, 1, 1), "NCHW", 0, "float16",
     "success"),
    # ((16, 192, 28, 28), (5, 5, 192, 1), (1, 192, 1, 1), (16, 192, 14, 14), 2, 1,
    #  (2, 2, 2, 2), "NCHW", 0, "float16", "success"),
    # ((8, 96, 56, 56), (3, 3, 96, 1), (1, 96, 1, 1), (8, 96, 56, 56), 1, 1,
    #  (1, 1, 1, 1), "NCHW", 0, "float16", "success"),
    # ((1, 32, 112, 112), (3, 3, 32, 1), (1, 32, 1, 1), (1, 32, 56, 56), 2, 1,
    #  (1, 1, 1, 1), "NCHW", 0, "float16", "success"),

    # ((2, 336, 504, 256), (3, 3, 256, 1), (1, 1, 1, 256), (2, 334, 502, 256), 1,
    #  1, (0, 0, 0, 0), "NHWC", 0, "float16", "success"),
    # ((648, 4, 5, 2048), (3, 3, 2048, 1), (1, 1, 1, 2048), (648, 2, 3, 2048), 1,
    #  1, (0, 0, 0, 0), "NHWC", 0, "float16", "success"),
    # ((2, 28, 28, 240), (5, 5, 240, 1), (1, 1, 1, 240), (2, 14, 14, 240), 2, 1,
    #  (2, 2, 2, 2), "NHWC", 0, "float16", "success"),

    # int8 without bias
    ((32, 960, 7, 7), (3, 3, 960, 1), None, (32, 960, 7, 7), 1, 1, (1, 1, 1, 1), "NCHW", 0, "int8", "success"),
    # ((16, 192, 28, 28), (5, 5, 192, 1), None, (16, 192, 14, 14), 2, 1,
    #  (2, 2, 2, 2), "NCHW", 0, "int8", "success"),
    # ((8, 96, 56, 56), (3, 3, 96, 1), None, (8, 96, 56, 56), 1, 1, (1, 1, 1, 1),
    #  "NCHW", 0, "int8", "success"),
    # ((1, 32, 112, 112), (3, 3, 32, 1), None, (1, 32, 56, 56), 2, 1,
    #  (1, 1, 1, 1), "NCHW", 0, "int8", "success"),

    # ((2, 336, 504, 256), (3, 3, 256, 1), None, (2, 334, 502, 256), 1, 1,
    #  (0, 0, 0, 0), "NHWC", 0, "int8", "success"),
    # ((648, 4, 5, 2048), (3, 3, 2048, 1), None, (648, 2, 3, 2048), 1, 1,
    #  (0, 0, 0, 0), "NHWC", 0, "int8", "success"),
    # ((2, 28, 28, 240), (5, 5, 240, 1), None, (2, 14, 14, 240), 2, 1,
    #  (2, 2, 2, 2), "NHWC", 0, "int8", "success"),
]

get_op_support_info_case=[
    ((32, 960, 7, 7), (3, 3, 960, 1), None, (32, 960, 7, 7), 1, 1, (1, 1, 1, 1), "NCHW", 0, "float16", "success"),
    ((32, 960, 7, 7), (3, 3, 960, 1), (1, 960, 1, 1), (32, 960, 7, 7), 1, 1, (1, 1, 1, 1), "NCHW", 0, "float16",
     "success"),
]
def _get_kernel_name(x_shape, filter_shape, bias_shape, stride, dilation, pads, dtype, offset_x):
    bias_shape = bias_shape if bias_shape else []
    kernel_name = 'dp_conv2d_' + '_'.join(map(str, x_shape)) + '_' + '_'.join(map(str, filter_shape)) + '_' + '_'.join(
        map(str, bias_shape)) + '_' + str(stride) + '_' + str(dilation) + "_" + '_'.join(map(
            str, pads)) + '_' + dtype + '_' + str(offset_x)
    return kernel_name


def _shape_to_NC1HWC0(shape, data_format, dtype):
    if data_format.upper() == "NCHW":
        n, c, h, w = shape
    else:  # NCHW
        n, h, w, c = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (n, c1, h, w, c0)


def _shape_to_C1HWNCoC0(shape, data_format, dtype):
    if data_format.upper() == "HWCN":
        h, w, c, n = shape
    else:  # NCHW
        n, c, h, w = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (c1, h, w, n, c0, c0)


def _gen_trans_data_case(param):
    x_shape, filter_shape, bias_shape, y_shape, stride, dilation, pads, data_format, offset_x, dtype, expect_result = param

    data_format = data_format.upper()
    dtype = dtype.lower()
    bias_dtype = "float32" if dtype == "float16" else "int32"
    y_dtype = "float16" if dtype == "float16" else "int32"

    x = {
        "shape": _shape_to_NC1HWC0(x_shape, data_format, dtype),
        "format": "NC1HWC0",
        "ori_shape": x_shape,
        "ori_format": data_format,
        "dtype": dtype
    }
    filter = {
        "shape": _shape_to_C1HWNCoC0(filter_shape, "HWCN", dtype),
        "ori_shape": filter_shape,
        "ori_format": "HWCN",
        "format": "C1HWNCoC0",
        "dtype": dtype
    }
    bias = {
        "shape": _shape_to_NC1HWC0(bias_shape, data_format, bias_dtype),
        "format": "NC1HWC0",
        "ori_shape": bias_shape,
        "ori_format": data_format,
        "dtype": bias_dtype
    } if bias_shape else None
    offset_w = None
    y = {
        "shape": _shape_to_NC1HWC0(y_shape, data_format, y_dtype),
        "format": "NC1HWC0",
        "ori_shape": y_shape,
        "ori_format": data_format,
        "dtype": y_dtype
    }
    strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]
    dilations = [1, dilation, dilation, 1] if data_format == "NHWC" else [1, 1, dilation, dilation]

    kernel_name = _get_kernel_name(x_shape, filter_shape, bias_shape, stride, dilation, pads, dtype, offset_x)
    return {
        "params": [x, filter, bias, offset_w, y, strides, dilations, pads, data_format, offset_x],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


# ============ auto test get_op_support_info and depthwise_compute api =====================
# get_op_support_info, depthwise_compute
def _test_other_api():
    from impl.depthwise_conv2d import depthwise_compute
    from impl.depthwise_conv2d import get_op_support_info
    for case in dp_conv2d_op_testcase:
        res = _gen_trans_data_case(case)
        x = res["params"][0]
        weights = res["params"][1]
        bias = res["params"][2]
        offset_w = res["params"][3]
        outputs = res["params"][4]
        strides = res["params"][5]
        dilations = res["params"][6]
        pads = res["params"][7]
        data_format = res["params"][8]
        offset_x = res["params"][9]
        kernel_name = res["case_name"]
        groups = 1
        _ = get_op_support_info(x, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format,
                                offset_x, kernel_name)
        _ = depthwise_compute(x, weights, bias, offset_w, outputs, strides, dilations, pads, data_format, offset_x,
                              kernel_name)

def _test_get_op_support_info(test_arg):
    for test_case in get_op_support_info_case:
        formatted_case = _gen_trans_data_case(test_case)
        params = formatted_case["params"]
        get_op_support_info(*params)

# ============ auto gen ["Ascend310"] test cases start ===============
for case in dp_conv2d_op_testcase:
    ut_case.add_case(["Ascend310"], _gen_trans_data_case(case))
# ============ auto gen ["Ascend310"] test cases end =================

# ============ auto gen ["Ascend910"] test cases start ===============
for case in dp_conv2d_op_testcase:
    ut_case.add_case(["Ascend910"], _gen_trans_data_case(case))
# ============ auto gen ["Ascend910"] test cases end =================
print("run depthwiseconv2d get_op_support_info test_case")
ut_case.add_cust_test_func(test_func=_test_get_op_support_info)

def test_depthwiseconv2d_split_info_unknown_shape(test_arg):
    input_list = [
        {
            'ori_shape': (-2,),
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding depthwiseconv2d test_depthwiseconv2d_split_info_unknown_shape testcase")
ut_case.add_cust_test_func(test_func=test_depthwiseconv2d_split_info_unknown_shape)

def test_depthwiseconv2d_split_info_normal(test_arg):
    input_list = [
        {
            'ori_shape': (-1, 64, 32, 32),
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, None, None, {
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    get_op_support_info(*input_list)
print("adding depthwiseconv2d test_depthwiseconv2d_split_info_normal testcase")
ut_case.add_cust_test_func(test_func=test_depthwiseconv2d_split_info_normal)

if __name__ == '__main__':
    # ut_case.run("Ascend910")
    ut_case.run()
    # test depthwise_conv2d other apis
    ut_case.add_cust_test_func(test_func=_test_other_api)
    exit(0)
