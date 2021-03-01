#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
import tbe

ut_case = OpUT("Conv2D", "impl.dynamic.conv2d",
               "conv2d")

def gen_trans_data_case(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x, expect):
    return {"params": [inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x],
            "case_name": "dynamic_conv2d_case",
            "expect": expect
            }

print("adding Conv2D dyanmic op testcases")
for test_case  in tc.conv2D_dynamic_ut_testcase:
    ut_case.add_case(test_case[0], gen_trans_data_case(*test_case[1:]))

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
