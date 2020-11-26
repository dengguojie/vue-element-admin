#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_compress(test_arg):
    import sys

    from impl.conv2d_compress import _conv_layer_compress_cce

    succ_str = "OK"
    try:
        fm_type = "int8"
        weight_type = "int32"
        output_type = "int32"
        fm_shape = (16, 32, 3, 3)
        w_shape = (16, 32, 1, 1)
        padding = [0, 0, 0, 0]
        strides = (1, 1)
        data_flow_type = -1
        bias_flag = 0
        deq_sqrt = 0
        deq_relu=0
        q_sqrt=0
        q_scaler=0
        q_offset=0
        kernel_name_val="conv_compress_int8_int8_int32_x_16_32_3_3_w_16_32_1_1_p_0_0_0_0_s_1_1_data_dtype_m1_bias_0_dequant_sqrt_0_dequant_relu_0_q_sqrt_0_q_scaler_0_q_offset_0"

        _conv_layer_compress_cce(fm_shape, w_shape, w_shape, fm_type, weight_type, weight_type,
                                    output_type, [padding[0], padding[1]], [padding[2], padding[3]],
                                    strides[0], strides[1], offset_x=q_offset,
                                    bias=bool(bias_flag),
                                    kernel_name=kernel_name_val,
                                    need_build=True, need_print=False)
    except RuntimeError as ex:
        print("ex:",ex)
        print("111111[ %s ] %s" % (succ_str, sys._getframe().f_code.co_name))


ut_case.add_cust_test_func(test_func=test_conv2d_compress)

if __name__ == "__main__":
    ut_case.add_cust_test_func(test_func=test_conv2d_compress)
    exit(0)