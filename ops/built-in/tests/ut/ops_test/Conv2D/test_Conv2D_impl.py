#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def gen_kernel_name(input_shape, weights_shape):
    dedy_shape_info = '_'.join([str(i) for i in input_shape])
    w_shape_info = '_'.join([str(i) for i in weights_shape])

    kernel_name = 'conv2d_x_{}_w_{}'.format(
        dedy_shape_info, w_shape_info)
    return kernel_name

def gen_trans_data_case(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, expect):

    input_shape = inputs.get('ori_shape')
    weights_shape = weights.get('ori_shape')
    kernel_name = gen_kernel_name(input_shape, weights_shape)
    return {"params": [inputs, weights, bias, offset_w, outputs, strides, pads, dilations],
            "case_name": kernel_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

print("adding Conv2D op testcases")
for test_case  in tc.conv2D_ut_testcase:
    ut_case.add_case(test_case[0], gen_trans_data_case(*test_case[1:]))

if __name__ == '__main__':
    ut_case.run(["Ascend910", "Ascend310"])
    exit(0)
