#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_ub_cal_fix(test_arg):
    import tbe
    from impl.conv2d import conv2d_compute
    from impl.mul import mul_compute
    from impl.accumulate_nv2 import _accumulate_nv2_compute
    from impl.relu import relu_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.tanh import tanh_compute
    from impl.conv2d_data_rm import conv2d_data_rm_compute
    from topi import generic
    from te import platform as cceconf
    from te import tvm

    def test_conv2d_mul_scalar_num():
        cceconf.te_set_version("Ascend910A")
        with tvm.target.cce():
            inputs = tvm.placeholder((24, 4, 128, 128, 16), name="fmap", dtype="float16")
            weights = tvm.placeholder((36, 4, 16, 16), name="weights", dtype="float16",
                                      attrs={"ori_shape": [3, 3, 64, 64], "ori_format": "HWCN"})
            bias = None
            offset_w = None
            outputs = None
            strides = [1, 1, 1, 1]
            pads = [1, 1, 1, 1]
            dilations = [1, 1, 1, 1]
            conv2d_res = conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations)

            input_x = tvm.placeholder((1,), name="input_x", dtype="float16")
            input_y = conv2d_res
            output_data = None
            res = mul_compute(input_x, input_y, output_data)

            tensor_list = [inputs, weights, input_x, res]
            sch = generic.auto_schedule(res)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "conv2d_fusion",
            "tensor_list": tensor_list
        }
        tbe.dsl.build(sch, config)

    def test_conv2d_accumulate_nv2_relu():
        cceconf.te_set_version("Ascend710")
        with tvm.target.cce():
            inputs = tvm.placeholder((1, 4, 56, 56, 16), name="fmap", dtype="float16")
            weights = tvm.placeholder((4, 16, 16, 16), name="weights", dtype="float16", attrs={"ori_shape": [256, 64, 1, 1], "ori_format": "NCHW"})
            bias = tvm.placeholder((256,), name="bias", dtype="float16")
            offset_w = None
            outputs = None
            strides = [1, 1, 1, 1]
            pads = [0, 0, 0, 0]
            dilations = [1, 1, 1, 1]
            conv2d_res = conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations)

            params_3 = tvm.placeholder((1, 16, 3136, 16), name="params_3", dtype="float16")
            tensor_list = [params_3, conv2d_res]
            y = {'shape': [1, 16, 56, 56, 16], 'dtype': 'float16', 'format': 'NC1HWC0'}
            num = 2
            accumulate_nv2_res = _accumulate_nv2_compute(tensor_list, y, num)

            x = accumulate_nv2_res
            y = None
            res = relu_compute(x, y)

            tbe_tensor_list = [inputs, weights, bias, params_3, res]
            sch = generic.auto_schedule(res)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "conv2d_fusion",
            "tensor_list": tbe_tensor_list
        }
        tbe.dsl.build(sch, config)


    def test_conv2d_dequant_tanh():
        cceconf.te_set_version("Ascend710")
        with tvm.target.cce():
            inputs = tvm.placeholder((1, 4, 256, 256, 32), name="fmap", dtype="int8")
            weights = tvm.placeholder((64, 1, 16, 32), name="weights", dtype="int8",
                                      attrs={"ori_shape": [4, 4, 128, 3], "ori_format": "HWCN"})
            bias = tvm.placeholder((3,), name="bias", dtype="int32")
            offset_w = None
            outputs = None
            strides = [1, 1, 1, 1]
            pads = [1, 1, 1, 1]
            dilations = [1, 1, 1, 1]
            conv2d_res = conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations)

            x = conv2d_res
            deq_scale = tvm.placeholder((1, 1, 1, 1, 16), name="deq_scale", dtype="uint64", attrs={"ori_shape": [3]})
            y = None
            dequant_res = ascend_dequant_compute(x, deq_scale, y)

            input_x = dequant_res
            output_y = None
            tanh_res = tanh_compute(input_x, output_y)

            res = conv2d_data_rm_compute(tanh_res)

            tbe_tensor_list = [inputs, weights, bias, deq_scale, res]
            sch = generic.auto_schedule(res)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "conv2d_fusion",
            "tensor_list": tbe_tensor_list
        }
        tbe.dsl.build(sch, config)

    test_conv2d_mul_scalar_num()
    test_conv2d_accumulate_nv2_relu()
    test_conv2d_dequant_tanh()

print("adding Conv2D ub cal fix testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_ub_cal_fix)