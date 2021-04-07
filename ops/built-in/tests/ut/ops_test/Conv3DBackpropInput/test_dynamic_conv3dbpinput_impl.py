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
        out_backprop={'ori_shape': (1, 8, 56, -1, 64), 'ori_format': 'NDHWC', 'dtype': 'float16', 'range': ((1, 1), (8, 8), (4, 4), (56, 56), (1, 75), (16, 16))},
        y={'ori_shape': (1, 8, 56, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16', 'range': ((1, 1), (8, 8), (16, 16), (56, 56), (1, 75), (16, 16))},
        strides=(1, 1, 1, 1, 1),
        pads=[0, 0, 0, 0, 0, 0],
        dilations=(1, 1, 1, 1, 1),
        groups=1,
        data_format="NDHWC"):
    return [input_size, filter, out_backprop, y, strides, pads, dilations, groups, data_format]


# test_conv3dbp_succ_dynamic
case1 = _run_api()

# test_conv3dbp_succ_dynamic_same_pad_depthwise
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (3, 6, 17, 1, 16), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 12, -1, 9, 16), 'ori_format': 'NDHWC', 'dtype': 'float16', 'range': ((1, 3), (12, 12), (2, 2), (5, 40), (9, 9), (16, 16))}
y = {'ori_shape': (-1, 24, -1, 26, 8), 'ori_format': 'NDHWC', 'dtype': 'float16', 'range': ((1, 3), (24, 24), (1, 1), (16, 68), (26, 26), (16, 16))}
strides = (1, 2, 3, 3, 1)
pads = [-1, -1, -1, -1, -1, -1]
groups = 8

case2 = _run_api(input_size, filter, out_backprop, y, strides, pads, groups=groups)


# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_case1", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_case2_same_pad_depthwise", True))

if __name__ == '__main__':
    ut_case.run()
