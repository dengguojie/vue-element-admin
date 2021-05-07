#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInput Dynamic
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropInput", "impl.dynamic.conv3d_backprop_input", "conv3d_backprop_input")
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
        filter={'ori_shape': (1, 1, 1, 256, 64), 'ori_format': 'DHWCN', 'dtype': 'float16'},
        out_backprop={'ori_shape': (1, 8, 56, -1, 64), 'ori_format': 'NDHWC', 'dtype': 'float16',
                      'range': ((1, 1), (8, 8), (4, 4), (56, 56), (1, 75), (16, 16))},
        y={'ori_shape': (1, 8, 56, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16',
           'range': ((1, 1), (8, 8), (16, 16), (56, 56), (1, 75), (16, 16))},
        strides=(1, 1, 1, 1, 1),
        pads=[0, 0, 0, 0, 0, 0],
        dilations=(1, 1, 1, 1, 1),
        groups=1,
        data_format="NDHWC"):
    return [input_size, filter, out_backprop, y, strides, pads, dilations, groups, data_format]


def test_conv3d_backprop_input_fuzz_build_generalization(test_arg):
    from impl.dynamic.conv3d_backprop_input import conv3d_backprop_input_generalization
    input_list = [
        {
            'shape': (5,),
            'ori_shape': (5,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (16, 4, 16, 16),
            'ori_format': 'DHWCN',
            'format': 'FRACTAL_Z_3D',
            'dtype': 'float16'
        }, {
            'shape': (1, 8, 4, 56, 56, 16),
            'ori_shape': (1, 8, 56, 56, 64),
            'ori_format': 'NDHWC',
            'format': 'NDC1HWC0',
            'dtype': 'float16',
            'range': [(1, 1), (8, 15), (4, 4), (32, 63), (32, 63), (16, 16)]
        }, {
            'shape': (1, 8, 16, 56, 56, 16),
            'ori_shape': (1, 8, 56, 56, 256),
            'ori_format': 'NDHWC',
            'format': 'NDC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 'conv3d_backprop_input_generalization']
    conv3d_backprop_input_generalization(*input_list)


# test_conv3dbp_succ_dynamic
case1 = _run_api()

# test_conv3dbp_succ_dynamic_same_pad_depthwise
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (3, 6, 17, 1, 16), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 16, -1, -1, 9), 'ori_format': 'NCDHW', 'dtype': 'float16',
                'range': ((1, 3), (12, 12), (2, 2), (5, 40), (9, 9), (16, 16))}
y = {'ori_shape': (-1, 8, -1, -1, 26), 'ori_format': 'NCDHW', 'dtype': 'float16',
     'range': ((1, 3), (24, 24), (1, 1), (16, 68), (26, 26), (16, 16))}
strides = (1, 1, 2, 3, 3)
pads = [-1, -1, -1, -1, -1, -1]
groups = 8
data_format="NCDHW"

case2 = _run_api(input_size, filter, out_backprop, y, strides, pads, groups=groups, data_format=data_format)

# test_conv3dbp_exception_error_pad
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (6, 3, 10, 6, 72), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 85, 25, 5, 72), 'ori_format': 'NDHWC', 'dtype': 'float16',
                'range': ((1, 2), (85, 85), (5, 5), (25, 25), (5, 5), (16, 16))}
y = {'ori_shape': (-1, 423, 73, 16, 6), 'ori_format': 'NDHWC', 'dtype': 'float16',
     'range': ((1, 2), (73, 73), (1, 1), (73, 73), (16, 16), (16, 16))}
strides = (1, 5, 3, 5, 1)
pads = [2, 10, 1, 1, 4, 5]
case3 = _run_api(input_size, filter, out_backprop, y, strides, pads)

# test_conv3dbp_exception_error_dilation
dilations = [1, 2, 1, 1, 1]
case4 = _run_api(dilations=dilations)

# test_conv3dbp_exception_-2
input_size = {'ori_shape': (-2,), 'ori_format': 'ND', 'dtype': 'int32'}
case5 = _run_api(input_size=input_size)

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_case1", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_case2_same_pad_depthwise", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case3, RuntimeError, "dynamic_case3_error_padding_d", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case4, RuntimeError, "dynamic_case4_error_dilation", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case5, RuntimeError, "dynamic_case5_error_minus_2", True))
# test_conv3d_backprop_input_fuzz_build_generalization
print("adding conv3d test_conv3d_backprop_input_fuzz_build_generalization testcase")
ut_case.add_cust_test_func(test_func=test_conv3d_backprop_input_fuzz_build_generalization)

if __name__ == '__main__':
    ut_case.run()
