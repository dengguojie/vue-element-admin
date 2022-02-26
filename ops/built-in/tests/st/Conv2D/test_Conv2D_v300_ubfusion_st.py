#!/usr/bin/env python
# -*- coding: UTF-8 -*-
def test_conv2d_v300_ubfusion():
    import tbe
    import json
    from impl.conv2d import _conv_layer_cce
    from impl.aipp import aipp_compute
    from impl.conv2d import conv2d
    from impl.conv2d import conv2d_compute
    from impl.trans_data import trans_data_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_requant import ascend_requant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.sigmoid import sigmoid_compute
    from impl.prelu import prelu_compute
    from impl.mul import mul_compute
    from impl.add import add_compute
    from impl.abs import abs_compute
    from impl.round import round_compute
    from impl.rsqrt import rsqrt_compute
    from impl.square import square_compute
    from tbe.common.context import op_context
    from te import platform as cceconf
    from te import tvm
    from te.platform.cce_build import build_config
    from te.platform.cce_build import build_config_update
    from topi import generic
    from tbe.dsl import auto_schedule
    from tbe.common.platform.platform_info import get_soc_spec


    case_list = [
    # dataflow,
    # conv_type,
    # shape_in,
    # shape_w,
    # pads,
    # strides,
    # groups,
    # bias_flag,
    # quant_scale=0,
    # quant_offset=0

    ("conv2d_add", "float16", [1, 64, 32, 32], [64, 64, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 1, True,),
    ("conv2d_mul", "float16", [1, 64, 32, 32], [64, 64, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 1, True,),
    ("conv2d_sigmoid_mul", "float16", [1, 64, 32, 32], [64, 64, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 1, True,),
    ("conv2d_abs", "float16", [1, 64, 32, 32], [64, 64, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 1, True,),
    ("conv2d_round", "float16", [1, 64, 32, 32], [64, 64, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 1, True,),
    ("conv2d_rsqrt_square", "float16", [1, 64, 32, 32], [64, 64, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 1, True,),
    ("conv2d_quant", "float16", [1, 64, 32, 32], [64, 64, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], 1, True,),
    ]

    def conv_v300_fusion_case(dataflow,
                             conv_type,
                             shape_in,
                             shape_w,
                             pads,
                             strides,
                             groups,
                             bias_flag,
                             quant_scale=0,
                             quant_offset=0):
        Ni, Ci, Hi, Wi = shape_in
        Co, _, Hk, Wk = shape_w

        Ci0_dict = {
            "float32": 8,
            "float16": 16,
            "int8": 32,
            "bfloat16": 16
        }
        Ci0 = Ci0_dict[conv_type]
        Ci1 = (Ci + Ci0 - 1) // Ci0

        Co0 = 16
        Co1 = (Co + Co0 - 1) // Co0

        shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
        shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)

        shape_scale = (1, Co1, 1, 1, 16)
        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]

        bias_dtype_dict = {
            "float32": "float32",
            "float16": "float32",
            "bfloat16": "float32",
            "int8": "int32"
        }
        bias_dtype = bias_dtype_dict[conv_type]

        with tvm.target.cce():
            fmap = tvm.placeholder(shape_in_5HD, name='fmap', dtype=conv_type)
            weight = tvm.placeholder(shape_w_fracz,
                                     name='weight',
                                     dtype=conv_type,
                                     attrs={
                                         'ori_shape': shape_w,
                                         'ori_format': 'NCHW'
                                     })
            bias = tvm.placeholder((Co1*Co0,), name='bias', dtype=bias_dtype) if bias_flag else None
            conv_res = conv2d_compute(fmap,
                                      weight,
                                      bias,
                                      None,
                                      None,
                                      strides,
                                      pads,
                                      dilations,
                                      offset_x=0)
            vdeq = tvm.placeholder(shape_scale,
                                   name='vreq_reg',
                                   dtype='uint64',
                                   attrs={'ori_shape': [Co1 * Co0]})

            if dataflow == "conv2d_mul":
                input_y = tvm.placeholder(conv_res.shape, name="input_y", dtype="float16")
                out = mul_compute(conv_res, input_y, None)
            elif dataflow == "conv2d_sigmoid_mul":
                res_sigmoid = sigmoid_compute(conv_res, None)
                input_y = tvm.placeholder(res_sigmoid.shape, name="input_y", dtype="float16")
                out = mul_compute(res_sigmoid, input_y, None)
            elif dataflow == "conv2d_add":
                input_y = tvm.placeholder((1,), name="input_y", dtype="float16")
                output_z = None
                is_scene_1d = False
                broadcast_flag = True
                out = add_compute(conv_res, input_y, output_z, is_scene_1d, broadcast_flag)
            elif dataflow == "conv2d_abs":
                out = abs_compute(conv_res, None)
            elif dataflow == "conv2d_round":
                out = round_compute(conv_res, None)
            elif dataflow == "conv2d_rsqrt_square":
                out = rsqrt_compute(conv_res, None)
                out = square_compute(out, None)
            elif dataflow == "conv2d_quant":
                out = ascend_quant_compute(conv_res, None, scale=0.5, offset=0.5, sqrt_mode=False)

            if dataflow in ("conv2d_mul", "conv2d_sigmoid_mul", "conv2d_add"):
                tensor_list = [fmap, weight, input_y, out]
            else:
                tensor_list = [fmap, weight, out]

            if bias_flag:
                tensor_list.insert(2, bias)

            sch = generic.auto_schedule(out)


    cceconf.te_set_version('Ascend320')
    with op_context.OpContext():
        for x in case_list:
            dataflow, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag = x
            conv_v300_fusion_case(dataflow, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag)

if __name__ == '__main__':
    test_conv2d_v300_ubfusion()