#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInput Dynamic
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DTranspose", "impl.dynamic.conv3d_transpose", "conv3d_transpose")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api(
        input_size={'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        x={'ori_shape': (-1, -1, -1, -1, 64), 'ori_format': 'NDHWC', 'dtype': 'float16', 
           'range': ((1, 3), (12, 12), (4, 4), (5, 40), (9, 9), (16, 16)), 'format': 'NDC1HWC0'},
        filter={'ori_shape': (1, 1, 1, 256, 64), 'ori_format': 'DHWCN', 'dtype': 'float16'},
        bias=None,
        offset_w=None,
        y={'ori_shape': (-1, -1, -1, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16',
           'range': ((1, 2), (8, 9), (16, 17), (56, 57), (1, 75), (16, 16))},
        strides=(1, 1, 1, 1, 1),
        pads=[0, 0, 0, 0, 0, 0],
        dilations=(1, 1, 1, 1, 1),
        groups=1,
        data_format="NDHWC",
        output_padding=(0, 0, 0, 0, 0),
        offset_x=0):
    return [input_size, x, filter, bias, offset_w, y, strides, pads, dilations,
            groups, data_format, output_padding, offset_x]

def _test_op_get_op_support_info_succ(test_arg):
    from impl.dynamic.conv3d_transpose import get_op_support_info

    [input_size, x, filter, bias, offset_w, y, strides, pads, dilations,
    groups, data_format, output_padding, offset_x] = _run_api()
    bias = {'ori_shape': (256,), 'ori_format': 'ND', 'dtype': 'float16'}
    get_op_support_info(
        input_size, x, filter, bias, offset_w, y, strides, pads, dilations,
        groups, data_format, output_padding, offset_x)

def _test_op_get_op_support_info_wrong_input(test_arg):
    from impl.dynamic.conv3d_transpose import get_op_support_info

    [input_size, x, filter, bias, offset_w, y, strides, pads, dilations,
    groups, data_format, output_padding, offset_x] = _run_api()
    input_size = {'ori_shape': (-2,), 'ori_format': 'ND', 'dtype': 'int32'}
    get_op_support_info(
        input_size, x, filter, bias, offset_w, y, strides, pads, dilations,
        groups, data_format, output_padding, offset_x)

ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info_succ)
ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info_wrong_input)


# test_conv3d_transpose_succ_dynamic
case1 = _run_api()

# test_conv3d_transpose_succ_dynamic_same_pad_depthwise
filter = {'ori_shape': (3, 6, 17, 1, 16), 'ori_format': 'DHWCN', 'dtype': 'float16'}
x = {'ori_shape': (-1, 16, 12, -1, 9), 'ori_format': 'NCDHW', 'dtype': 'float16',
     'range': ((1, 3), (12, 12), (2, 2), (5, 40), (9, 9), (16, 16))}
y = {'ori_shape': (-1, 8, 24, -1, 26), 'ori_format': 'NCDHW', 'dtype': 'float16',
     'range': ((1, 3), (24, 24), (1, 1), (16, 68), (26, 26), (16, 16))}
strides = (1, 1, 2, 3, 3)
pads = [-1, -1, -1, -1, -1, -1]
groups = 8
data_format="NCDHW"

case2 = _run_api(filter=filter, x=x, y=y, strides=strides, pads=pads, groups=groups,
                 data_format=data_format)

# test_error_output_padding_d
output_padding =[1, 100, 1, 1, 1]

case3 = _run_api(output_padding=output_padding)

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_case1", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_case2_same_pad_depthwise", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case3, RuntimeError, "dynamic_case3_error_output_padding_d", True))

if __name__ == '__main__':
    ut_case.run()
