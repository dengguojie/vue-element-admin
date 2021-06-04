#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_multi_conv2d(test_arg):
    from impl.conv2d import conv2d_compute
    from tbe import tvm
    # case name: ((fm_shape), (weight_shape), (paddings), (strides), data_flow, bias_flag)
    testcase = {
        "conv_bias_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], True),
        "conv_nobias_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], False)
    }

    def conv_v200_fusion(fm_shape, filter, pads, strides, bias_flag, kernel_name):
        batch, channel, height, weight = fm_shape
        C0 = 16
        block_size_k = 16
        block_size_n = 16
        C1 = (channel + C0 - 1) // C0
        shape_in_5hd = (batch, C1, height, weight, C0)
        shape_in_4hd = (batch, C1, height * weight, C0)
        out_channel = filter[0]
        in_channel_weight = ((filter[1] + block_size_k - 1) // block_size_k) * block_size_k
        filter_h = filter[2]
        filter_w = filter[3]
        c_out = (out_channel + block_size_n - 1) // block_size_n
        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                c_out, block_size_n, block_size_k)
        # conv2d
        dilations = [1, 1, 1, 1]
        shape_c = (1, c_out, 1, 1, 16)
        fm = tvm.placeholder(shape_in_4hd, name='fm', dtype='float16', attrs={'ori_format': 'NCHW', "current_shape":shape_in_5hd})
        filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='float16',
                            attrs={'ori_shape': filter, 'ori_format': 'NCHW'})
        if bias_flag:
            bias_tensor = tvm.placeholder((c_out*16,), name='bias', dtype='int32')
        else:
            bias_tensor = None
        conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations)
    
    for key, value in testcase.items():
        print("test multi conv2d case:",key)
        conv_v200_fusion(*value, key)


print("test multi Conv2D running")
ut_case.add_cust_test_func(test_func=test_multi_conv2d)

# if __name__ == '__main__':
#     ut_case.run("Ascend310")
#     ut_case.run("Ascend910A")
#     ut_case.run("Ascend710")
#     exit(0)
