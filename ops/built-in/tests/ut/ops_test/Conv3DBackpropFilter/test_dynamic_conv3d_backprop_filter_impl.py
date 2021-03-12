#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInput Dynamic
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropFilter", "impl.dynamic.conv3d_backprop_filter", "conv3d_backprop_filter")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api(x=None,
             filter_size=None,
             out_backprop=None,
             filter=None,
             strides=(1, 1, 1, 1, 1),
             pads=[0, 0, 0, 0, 0, 0],
             dilations=(1, 1, 1, 1, 1),
             groups=1,
             data_format="NDHWC"):
    if x is None:
        x = {'ori_shape': (1, -1, -1, -1, 256), 'shape': (1, -1, -1, -1, 256),
             'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', 
             "range": [(1, 1), (16, 453), (1, 160), (160, 160), (256,256)]}
    if filter_size is None or filter is None:
        filter = {'ori_shape': (1, 1, 1, 256, 64), 'shape': (1, 1, 1, 256, 64),
                  'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32', 
                  "range": [(1, 1), (1, 1), (1, 1), (256, 256), (64,64)]}
        filter_size = filter
    if out_backprop is None:
        out_backprop = {'ori_shape': (1, -1, -1, -1, 64), 'shape': (1, -1, -1, -1, 64),
                        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', 
                        "range": [(1, 1), (1, 47), (1, 1), (256, 256), (64, 64)]}

    return [x, filter_size, out_backprop, filter, strides, pads, dilations, groups, data_format]


# test_conv3dbp_succ_dynamic
case1 = _run_api()

# test_dynamic_dw_success_SAME_padding
x = {'ori_shape': (1, -1, -1, -1, 256), 'shape': (1, -1, -1, -1, 256),
     'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', 
     "range": [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)]}
filter = {'ori_shape': (3, 3, 3, 256, 256), 'shape': (3, 3, 3, 256, 256),
                  'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32', 
                  "range": [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)]}
out_backprop = {'ori_shape': (1, -1, -1, -1, 256), 'shape': (1, -1, -1, -1, 256),
                        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16', 
                        "range": [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)]}
filter_size = filter
strides = [1,2,2,2,1]
pads = [-1,-1,-1,-1,-1,-1]
case2 = _run_api(x=x, filter_size=filter_size, out_backprop=out_backprop, filter=filter,
                 strides=strides, pads=pads)


# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_pass_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_SAME_strides_2_2_2_case", True))

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run()
    exit(0)

