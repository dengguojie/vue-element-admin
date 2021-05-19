#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DTransposeD
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DTransposeD", "impl.conv3d_transpose_d",
               "conv3d_transpose_d")


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}

# base case
def _run_api_end_with_d(out_backprop=None,
                        filters=None,
                        bias=None,
                        offset_w=None,
                        y_input=None,
                        input_sizes=(1, 16, 120, 176, 32),
                        strides=(1, 2, 2, 2, 1),
                        pads=[0, 0, 0, 0, 0, 0],
                        dilations=(1, 1, 1, 1, 1),
                        groups=1,
                        data_format="NDHWC",
                        output_padding=[0, 0, 0, 0, 0],
                        offset_x=0):
    if not out_backprop:
        out_backprop = {
            'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    if not filters:
        filters = {
            'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
    if not y_input:
        y_input = {
            'ori_shape': (1, 16, 120, 176, 32),
            'shape': (1, 16, 120, 176, 32),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    return [out_backprop, filters, bias, offset_w, y_input, input_sizes,
            strides, pads, dilations, groups, data_format, output_padding,
            offset_x]


def test_op_check_supported(test_arg):
    from impl.conv3d_transpose_d import check_supported
    input_sizes = (2, 16, 120, 176, 32)
    (out_backprop, filters, bias, offset_w, y_input, input_sizes,
        strides, pads, dilations, groups, data_format, output_padding,
        offset_x) = _run_api_end_with_d(input_sizes=input_sizes)
    check_supported(out_backprop, filters, bias, offset_w, y_input, input_sizes,
                    strides, pads, dilations, groups, data_format, output_padding,
                    offset_x)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

# test_conv3d_transpose_succ_d
case1 = _run_api_end_with_d()

# test_conv3d_transpose_stride_one
out_backprop = {
    'ori_shape': (1, 15, 119, 175, 64), 'shape': (1, 15, 119, 175, 64),
    'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
case2 = _run_api_end_with_d(out_backprop=out_backprop, strides=strides)

# test_conv3d_transpose_success_NCDHW
out_backprop = {
    'ori_shape': (1, 64, 8, 60, 88), 'shape': (1, 64, 8, 60, 88),
    'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
filters = {
    'ori_shape': (64, 32, 2, 2, 2), 'shape': (64, 32, 2, 2, 2),
    'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
y_input = {
    'ori_shape': (1, 32, 16, 120, 176),
    'shape': (1, 32, 16, 120, 176),
    'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
strides = (1, 1, 2, 2, 2)
input_sizes = (1, 32, 16, 120, 176)
case3 = _run_api_end_with_d(out_backprop=out_backprop,
                            filters=filters,
                            y_input=y_input, input_sizes=input_sizes,
                            strides=strides, data_format='NCDHW')

# test_conv3d_transpose_pad_same_failed
pads = "SAME"
case4 = _run_api_end_with_d(pads=pads)

# Invalid padding
pads = [0, 0, 0, 0]
case5 = _run_api_end_with_d(pads=pads)

# test_conv3d_transpose_invalid_dilations
dilations = [1, 0, 1, 0]
case6 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_transpose_invalid_shape
# fmap_channel != filter_channel
input_sizes = (1, 16, 120, 176, 16)
case7 = _run_api_end_with_d(input_sizes=input_sizes)

# dedy_channel != filter_batch
out_backprop = {
    'ori_shape': (1, 16, 60, 88, 16), 'shape': (1, 16, 60, 88, 16),
    'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
case8 = _run_api_end_with_d(out_backprop=out_backprop)

# fmap_batch != dedy_batch
input_sizes = (2, 16, 120, 176, 32)
case9 = _run_api_end_with_d(input_sizes=input_sizes)

# Filter with NDHWC but failed.
filters = {
    'ori_shape': (64, 32, 2, 2, 2), 'shape': (64, 32, 2, 2, 2),
    'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
input_sizes = (1, 16, 120, 176, 16)
case10 = _run_api_end_with_d(filters=filters, input_sizes=input_sizes)

# Filter with NCDHW but failed.
filters = {'ori_shape': (64, 2, 2, 2, 32), 'shape': (64, 2, 2, 2, 32),
           'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
input_sizes = (1, 16, 120, 176, 16)
case11 = _run_api_end_with_d(filters=filters, input_sizes=input_sizes)

# Wrong data format and failed.
out_backprop = {
    'ori_shape': (1, 64, 60, 88, 8), 'shape': (1, 64, 60, 88, 8),
    'ori_format': 'NCHWD', 'format': 'NCHWD', 'dtype': 'float16'}
case12 = _run_api_end_with_d(out_backprop=out_backprop)

filters = {
    'ori_shape': (64, 32, 2, 2, 2), 'shape': (64, 32, 2, 2, 2),
    'ori_format': 'NWCDH', 'format': 'NWCDH', 'dtype': 'float16'}
case13 = _run_api_end_with_d(filters=filters)

y_input = {
    'ori_shape': (1, 32, 16, 120, 176),
    'shape': (1, 32, 16, 120, 176),
    'ori_format': 'NWCDH', 'format': 'NWCDH', 'dtype': 'float16'}
case14 = _run_api_end_with_d(y_input=y_input)

# Add test Cases
# Params is the input params of the operator. Fro example, [fmap,filter,bias...]
ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case1, "success", "case1", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case2, "success", "case2", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case3, "success", "case3", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case4, RuntimeError, "case4", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case5, RuntimeError, "case5", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case6, ValueError, "case6", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case7, RuntimeError, "case7", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case8, RuntimeError, "case8", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case9, RuntimeError, "case9", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case10, RuntimeError, "case10", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case11, RuntimeError, "case11", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case12, RuntimeError, "case12", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case13, RuntimeError, "case13", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case14, RuntimeError, "case14", True))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
