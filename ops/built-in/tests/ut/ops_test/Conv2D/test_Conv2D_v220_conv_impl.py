#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_v220(test_arg):

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
    from impl.prelu import prelu_compute
    from tbe.common.context import op_context
    from te import platform as cceconf
    from te import tvm
    from te.platform.cce_build import build_config
    from te.platform.cce_build import build_config_update
    from topi import generic
    from tbe.common.platform.platform_info import get_soc_spec


    v220_case = [
        #          dataflow,
        #          conv_type,
        #          shape_in,
        #          shape_w,
        #          pads,
        #          strides,
        #          groups,
        #          bias_flag,
        #          (relu_mode), "prelu/relu/None", prelu only allows fp16 and fp32 dtype as input.
        #          (quant_scale),
        #          (quant_offset),
        #          (sqrt_mode)  # only for dequant

        # ("conv2d_relu", "float16", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),

        # ("conv2d_relu", "float32", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # 有问题

        # ("conv2d_relu", "bfloat16", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # vrelu数据类型不支持

        # ('conv2d_quant', 'float16', (4, 128, 32, 32), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, None, 0.5, 0.5), # 有问题

        # ("conv2d_quant", "float16", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, None, 0.5, 0.5),

        # ("conv2d_quant", "float16", (1, 64, 32, 32), (48, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, None, 0.5, 0.5),

        # ("conv2d_quant", "float16", (1, 64, 32, 32), (112, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, None, 0.5, 0.5),

        # ("conv2d_dequant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "None", False),

        # ("conv2d_dequant", "int8", (1, 64, 32, 32), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "None", False),

        # ("conv2d_dequant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "relu", False),

        # ("conv2d_dequant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "leaky_relu", False),

        # ("conv2d_dequant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "prelu", False),

        # ("conv2d_requant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "None"),

        # ("conv2d_requant", "int8", (2, 2048, 49, 49), (480, 2048, 1, 1), (0, 0, 0, 0), (1, 1), 1, False, "None"), # 有问题

        # ("conv2d_requant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "relu"),

        # ("conv2d_requant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, "leaky_relu"), # vrelu数据类型不支持
    ]

    v220_single_op_case = [
        #          dataflow,
        #          conv_type,
        #          shape_in,
        #          shape_w,
        #          pads,
        #          strides,
        #          groups,
        #          bias_flag
        # ("conv2d_single_op", "float16", (1, 9, 2, 2), (36, 9, 2, 2), (0, 0, 0, 0), (2, 2), 1, False), # 连续轴限制

        ("conv2d_single_op", "float16", (1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),
        ("conv2d_single_op", "float16", (1, 64, 49, 49), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),

        ('conv2d_single_op', 'float16', (2, 2048, 49, 49), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),

        # drv platform
        ("conv2d_single_op", "float16", (2, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),

        ("conv2d_single_op", "float16", (2, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),

        ("conv2d_single_op", "float16", (2, 64, 64, 48), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 1, False),

        # single op
        # load2d
        ("conv2d_single_op", "float16", (1, 64, 28, 28), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 1, False), # √

        # strideh优化
        ("conv2d_single_op", "float16", (1, 64, 14, 14), (64, 64, 1, 1), (0, 0, 0, 0), (2, 2), 1, False), # √

        ("conv2d_single_op", "float16", (1, 64, 28, 28), (64, 64, 1, 1), (0, 0, 0, 0), (2, 2), 1, False), # √

        ("conv2d_single_op", "int8", (1, 64, 28, 28), (64, 64, 1, 1), (0, 0, 0, 0), (2, 2), 1, False), # √
        ("conv2d_single_op", "int8", (1, 64, 138, 138), (64, 64, 1, 1), (0, 0, 0, 0), (2, 2), 1, False),

        # 正常case
        ("conv2d_single_op", "float16", (1, 24, 28, 28), (64, 24, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
        ("conv2d_single_op", "float16", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
        ("conv2d_single_op", "float16", (1, 128, 32, 32), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # √

        ("conv2d_single_op", "int8", (1, 128, 32, 32), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # √

        # ("conv2d_single_op", "float32", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # 后续蓝黄同步再上
        # ("conv2d_single_op", "float32", (1, 128, 32, 32), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # 后续蓝黄同步再上

        # ("conv2d_single_op", "bfloat16", (1, 128, 32, 32), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),

        # c0 = 4 single_op 支持各种数据类型，验证非首层fp16以外的组合
        ("conv2d_single_op", "float16", (1, 3, 32, 32), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
        # ("conv2d_single_op", "int8", (1, 3, 32, 32), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # tiling有问题
        # ("conv2d_single_op", "float32", (1, 3, 32, 32), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # tiling有问题

        # N*1
        ("conv2d_single_op", "float16", (1, 16, 16, 16), (16, 16, 1, 16), (0, 0, 0, 0), (1, 1), 1, False),

        # ===========================group卷积用例=============================================
        ("conv2d_single_op", "float16", (1, 120, 14, 14), (360, 40, 1, 1), (0, 0, 0, 0), (1, 1), 30, False),
        ("conv2d_single_op", "float16", (1, 600, 14, 14), (360, 40, 1, 1), (0, 0, 0, 0), (1, 1), 15, False),
    ]

    v220_input_nd2nz_case = [
        #          dataflow,
        #          conv_type,
        #          shape_in,
        #          shape_w,
        #          pads,
        #          strides,
        #          groups,
        #          bias_flag,
        #          (relu_mode), "prelu/relu/None"
        #          (quant_scale),
        #          (quant_offset),
        #          (sqrt_mode)  # only for dequant

        # ===================================ND2NZ  fmap输入为NHWC格式========================================
        # ("nd2nz_conv2d_relu", "float16", (1, 32, 32, 64), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
        ("nd2nz_conv2d", "float16", (1, 28, 28, 64), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
        ("nd2nz_conv2d", "float16", (1, 28, 28, 24), (64, 24, 3, 3), (1, 1, 1, 1), (1, 1), 1, False), # C补pad

        # ("nd2nz_conv2d", "float32", (1, 32, 32, 64), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
        # ("nd2nz_conv2d", "bfloat16", (1, 32, 32, 64), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
        ("nd2nz_conv2d", "int8", (1, 28, 28, 64), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
    ]

    v220_output_nz2nd_case = [
        #          dataflow,
        #          conv_type,
        #          shape_in,
        #          shape_w,
        #          cout_real,
        #          pads,
        #          strides,
        #          groups,
        #          bias_flag,
        #          (relu_mode), "prelu/relu/None"
        #          (quant_scale),
        #          (quant_offset),
        #          (sqrt_mode)  # only for dequant

        #====================================NZ2ND======================================
        # ("conv2d_relu_nz2nd", "float32", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False), # 有问题
        # ("conv2d_relu_nz2nd", "bfloat16", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False),
        # ("conv2d_relu_nz2nd", "int8", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False),

        # ("conv2d_quant_nz2nd", "float16", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False, None, 0.5, 0.5),
        # ("conv2d_dequant_nz2nd", "int8", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False, "relu", False),
        # ("conv2d_requant_nz2nd", "int8", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False, "relu", False),

        ("conv2d_nz2nd", "float16", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False),
        # ("conv2d_nz2nd", "float32", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False),
        # ("conv2d_nz2nd", "bfloat16", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False),
        ("conv2d_nz2nd", "int8", (1, 64, 32, 32), (32, 64, 3, 3), 32, (1, 1, 1, 1), (1, 1), 1, False),

        #==============C方向需要剔数据==============================
    ]

    # AIPP generates a FP16 output
    v220_aipp_case = [
        #          dataflow,
        #          conv_type,
        #          shape_in,
        #          shape_w,
        #          pads,
        #          strides,
        #          groups,
        #          bias_flag,
        #          crop_flag,
        #          c04_flag,
        #          aipp_format,
        #          (quant_scale),
        #          (quant_offset),

        # ("aipp_conv2d", "float16", (1, 3, 24, 24), (16, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, False, True, "yuv"),
        # ("aipp_conv2d", "float16", (1, 3, 1024, 1824), (32, 3, 3, 3), (1, 1, 1, 1), (2, 2), 1, False, True, False, "yuv"), # 非c04
        # ("aipp_conv2d", "float16", (16, 3, 228, 240), (64, 3, 3, 3), (0, 0, 0, 0), (2, 2), 1, False, True, False, "rgb"), # 非c04
        # ("aipp_conv2d", "float16", (1, 4, 24, 24), (16, 4, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, False, True, "xrgb"),

        # ("aipp_conv2d_quant", "float16", (1, 3, 416, 416), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, False, False, True, "yuv", 0.5, 0.5),
    ]

    def is_support_v100():
        """
        Check if Ascend310/Ascend910 version.
        """
        soc_version = get_soc_spec("SOC_VERSION")
        if soc_version in ("Ascend310", "Ascend910"):
            return True
        return False


    def is_support_v200():
        """
        Check if Ascend610/Ascend615/Ascend710/Hi3796CV300CS version.
        """
        soc_version = get_soc_spec("SOC_VERSION")
        if soc_version in ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS",
                           "SD3403"):
            return True
        return False


    def conv_v220_fusion_case(dataflow,
                              conv_type,
                              in_nd2nz_flag,
                              out_nz2nd_flag,
                              shape_in,
                              shape_w,
                              pads,
                              strides,
                              groups,
                              bias_flag,
                              relu_mode=None,
                              quant_scale=0,
                              quant_offset=0,
                              sqrt_mode=False,
                              cout_real=0):
        if in_nd2nz_flag:
            Ni, Hi, Wi, Ci = shape_in
        else:
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
            if in_nd2nz_flag:
                fmap_ori = tvm.placeholder(shape_in,
                                           name='fmap_ori',
                                           dtype=conv_type)
                print("fmap_ori", fmap_ori)
                fmap = trans_data_compute(fmap_ori,
                                          None,
                                          src_format="NHWC",
                                          dst_format="NC1HWC0")
            else:
                fmap = tvm.placeholder(shape_in_5HD, name='fmap', dtype=conv_type)

            weight = tvm.placeholder(shape_w_fracz,
                                     name='weight',
                                     dtype=conv_type,
                                     attrs={
                                         'ori_shape': shape_w,
                                         'ori_format': 'NCHW'
                                     })
            bias = tvm.placeholder(
                (Co1 *
                 Co0, ), name='bias', dtype=bias_dtype) if bias_flag else None
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

            relu_flag = True if relu_mode == "relu" else False

            if dataflow == "conv2d":
                out = conv_res
            if dataflow == "conv2d_relu":
                out = leaky_relu_compute(conv_res, None)
            elif dataflow == "conv2d_quant":
                out = ascend_quant_compute(conv_res,
                                           None,
                                           scale=quant_scale,
                                           offset=quant_offset,
                                           sqrt_mode=False)
            elif dataflow == "conv2d_dequant":
                out = ascend_dequant_compute(conv_res,
                                             vdeq,
                                             None,
                                             sqrt_mode=False,
                                             relu_flag=relu_flag)
            elif dataflow == "conv2d_requant":
                out = ascend_requant_compute(conv_res,
                                             vdeq,
                                             None,
                                             relu_flag=relu_flag)

            if relu_mode == "prelu":
                weight_input = tvm.placeholder(
                    (1, out.shape[-3], 1, out.shape[-1]),
                    name='weight_input',
                    dtype=out.dtype,
                    attrs={'ori_shape': [Co1 * Co0]})
                out = prelu_compute(out, weight_input, None)
            elif relu_mode == "leaky_relu":
                out = leaky_relu_compute(out, None, negative_slope=0.1)

            if out_nz2nd_flag:
                src_n, src_c1, src_hw, src_c0 = tuple(i.value for i in out.shape)
                out_nhwc_shape = (src_n, src_hw, src_c1 *
                                  src_c0) if cout_real == 0 else (src_n, src_hw,
                                                                  cout_real)
                out = trans_data_compute(out, {"shape": out_nhwc_shape},
                                         src_format="NC1HWC0",
                                         dst_format="NHWC")

            if dataflow in ("conv2d", "conv2d_relu", "conv2d_quant"):
                tensor_list = [fmap, weight, out]
            elif dataflow in ("conv2d_dequant", "conv2d_requant"):
                tensor_list = [fmap, weight, vdeq, out]

            if bias_flag:
                tensor_list.insert(2, bias)
            if relu_mode == "prelu":
                tensor_list.insert(-1, weight_input)
            if in_nd2nz_flag:
                tensor_list[0] = fmap_ori

            sch = generic.auto_schedule(out)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "conv2d_fusion",
            "tensor_list": tensor_list
        }
        tbe.dsl.build(sch, config)


    def conv_v220_single_op_case(conv_type,
                                 in_nd2nz_flag,
                                 out_nz2nd_flag,
                                 shape_in,
                                 shape_w,
                                 pads,
                                 strides,
                                 groups,
                                 bias_flag,
                                 c04_flag=False):

        Ni, Ci, Hi, Wi = shape_in
        Co, w_Ci, Hk, Wk = shape_w

        Co0 = 16
        Co1 = (Co + Co0 - 1) // Co0

        Ci0_dict = {
            "float32": 8,
            "float16": 16,
            "int8": 32,
            "bfloat16": 16
        }
        Ci0 = Ci0_dict[conv_type]
        Ci1 = (Ci + Ci0 - 1) // Ci0

        if w_Ci == 3:  # c0 = 4
            c04_flag = True

            w_Ci0 = 16
            k1_w_fracz = (Hk * Wk * 4 + w_Ci0 - 1) // w_Ci0
            Ci0 = 4 if is_support_v200() else Ci0

            shape_in = (Ni, 4, Hi, Wi)
            shape_in_5HD = (Ni, 1, Hi, Wi, Ci0)
            shape_w_fracz = (k1_w_fracz, Co1, Co0, w_Ci0)
        else:
            Ci0_dict = {
                "float32": 8,
                "float16": 16,
                "int8": 32,
                "bfloat16": 16
            }
            Ci0 = Ci0_dict[conv_type]
            Ci1 = (Ci + Ci0 - 1) // Ci0
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
        w_format = "FRACTAL_Z_C04" if c04_flag else "FRACTAL_Z"

        res_dtype_dict = {
            "float32": "float32",
            "float16": "float16",
            "bfloat16": "bfloat16",
            "int8": "int32"
        }

        #===================config conv2d parameters====================
        inputs = {
            "ori_shape": shape_in,
            "ori_format": "NCHW",
            "shape": shape_in_5HD,
            "format": "NC1HWC0_C04" if is_support_v200() else "NC1HWC0",
            "dtype": conv_type,
            "is_first_layer": False
        }

        weights = {
            "ori_shape": shape_w,
            "ori_format": "NCHW",
            "shape": shape_w_fracz,
            "format": w_format,
            "dtype": conv_type,
        }

        bias = {
            "ori_shape": (Co1 * Co0),
            "dtype": bias_dtype
        } if bias_flag else None
        offset_w = None
        outputs = {"dtype": res_dtype_dict[conv_type]}

        # data_format决定了strides和dilations怎么取， 默认"NCHW"

        print("============conv2d inputs==============", inputs)
        print("============conv2d weights==============", weights)
        conv2d(inputs,
               weights,
               bias,
               offset_w,
               outputs,
               strides,
               pads,
               dilations,
               groups=groups,
               offset_x=0,
               kernel_name="conv2d")


    def conv_v220_aipp_case(dataflow,
                            conv_type,
                            shape_in,
                            shape_w,
                            pads,
                            strides,
                            groups,
                            bias_flag,
                            crop_flag,
                            c04_flag,
                            aipp_format,
                            quant_scale=0.5,
                            quant_offset=0.5):
        Ni, Ci, Hi, Wi = shape_in
        Co, _, Hk, Wk = shape_w

        Ci0 = 4 if c04_flag else 16
        Co0 = 16
        Ci1 = 1
        Co1 = (Co + Co0 - 1) // Co0

        shape_in_5HD = (Ni, (Ci + Ci0 - 1) // Ci0, Hi, Wi, Ci0)

        if c04_flag:
            shape_w_fracz = ((Hk * Wk * Ci1 + Ci0 - 1) // Ci0, Co1, Co0, Ci0 * 4)
            weight_format = "FRACTAL_Z_C04"
        else:
            shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)
            weight_format = "FRACTAL_Z"

        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]
        bias_dtype = "float32"
        aipp_format_dict = {
            "yuv": "YUV420SP_U8",
            "xrgb": "XRGB8888_U8",
            "rgb": "RGB888_U8"
        }
        aipp_input_format = aipp_format_dict[aipp_format]

        with tvm.target.cce():
            fmap = tvm.placeholder(shape_in,
                                   name="params_0",
                                   dtype="uint8",
                                   attrs={
                                       "ori_shape": shape_in,
                                       "format": "NCHW",
                                       "ori_format": "NHWC"
                                   })
            if crop_flag:
                h_after_crop = Hi - 4
                w_after_crop = Wi - 4
            else:
                h_after_crop = Hi
                w_after_crop = Wi

            if c04_flag:
                output_data = {
                    'shape': [Ni, 1, h_after_crop, w_after_crop, 4],
                    'ori_shape': [Ni, 3, h_after_crop, w_after_crop],
                    'format': 'NC1HWC0_C04',
                    'ori_format': 'NCHW',
                    'dtype': 'float16',
                    'addr_type': 0,
                    'valid_shape': (),
                    'slice_offset': (),
                    'use_L1_workspace': 0,
                    'L1_workspace_size': -1,
                    'L1_fusion_type': -1,
                    'L1_addr_offset': 0,
                    'total_shape': (),
                    'split_index': 0
                }
            else:
                output_data = {
                    'shape': [Ni, 1, h_after_crop, w_after_crop, 16],
                    'ori_shape': [Ni, 3, h_after_crop, w_after_crop],
                    'format': 'NC1HWC0',
                    'ori_format': 'NCHW',
                    'dtype': 'float16',
                    'addr_type': 0,
                    'valid_shape': (),
                    'slice_offset': (),
                    'use_L1_workspace': 0,
                    'L1_workspace_size': -1,
                    'L1_fusion_type': -1,
                    'L1_addr_offset': 0,
                    'total_shape': (),
                    'split_index': 0
                }

            aipp_config_dict = {
                "cce_product": "2.1",
                "out_dtype": "float16",
                "out_format": "NC1HWC0",
                "aipp_mode": "static",
                "related_input_rank": 0,
                "input_format": aipp_input_format,
                "src_image_size_n": shape_in[0],
                "src_image_size_c": shape_in[1],
                "src_image_size_h": shape_in[2],
                "src_image_size_w": shape_in[3],
                "crop": crop_flag,
                "load_start_pos_h": 2,
                "load_start_pos_w": 2,
                "crop_size_h": h_after_crop,
                "crop_size_w": w_after_crop,
                "src_image_h": shape_in[2],
                "src_image_w": shape_in[3],
                "resize": 0,
                "resize_model": 0,
                "resize_output_h": 32,
                "resize_output_w": 32,
                "padding": 0,
                "left_padding_size": 7,
                "right_padding_size": 7,
                "top_padding_size": 0,
                "bottom_padding_size": 4,
                "csc_switch": 1,
                "rbuv_swap_switch": 0,
                "ax_swap_switch": 0,
                "matrix_r0c0": 298,
                "matrix_r0c1": 516,
                "matrix_r0c2": 0,
                "matrix_r1c0": 298,
                "matrix_r1c1": -100,
                "matrix_r1c2": -208,
                "matrix_r2c0": 298,
                "matrix_r2c1": 0,
                "matrix_r2c2": 409,
                "input_bias_0": 16,
                "input_bias_1": 128,
                "input_bias_2": 128,
                "mean_chn_0": 1,
                "mean_chn_1": 1,
                "mean_chn_2": 1,
                "mean_chn_3": 1,
                "var_reci_chn_0": 1.0,
                "var_reci_chn_1": 1.0,
                "var_reci_chn_2": 1.0,
                "min_chn_0": 0,
                "min_chn_1": 0,
                "min_chn_2": 0,
                "min_chn_3": 0
            }

            aipp_config_dict_json = json.dumps(aipp_config_dict)
            aipp_res = aipp_compute(fmap,
                                    None,
                                    output_data,
                                    aipp_config_dict_json,
                                    kernel_name="aipp")
            aipp_res.op.attrs["is_first_layer"] = True

            weight = tvm.placeholder(shape_w_fracz,
                                     name="weight",
                                     dtype="float16",
                                     attrs={
                                         "ori_shape": shape_w,
                                         "ori_format": "NCHW",
                                         "format": weight_format
                                     })

            bias = tvm.placeholder(
                (Co1 *
                 Co0, ), name='bias', dtype=bias_dtype) if bias_flag else None

            conv_res = conv2d_compute(aipp_res, weight, bias, None, None, strides,
                                      pads, dilations)
            # out = leaky_relu_compute(conv_res, None, negative_slope=0)
            out = conv_res
            sch = generic.auto_schedule(out)

        kernel_name = "aipp_fusion"
        tensor_list = [fmap, weight, out]
        if bias_flag:
            tensor_list.insert(2, bias)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": kernel_name,
            "tensor_list": tensor_list
        }
        tbe.dsl.build(sch, config)


    def run_testcase(config_dict):
        for i in config_dict:
            print("=" * 150)
            print("case {}".format(i))
            print()

            in_nd2nz_flag = False
            out_nz2nd_flag = False
            cout_real = 0

            if i[0].startswith("nd2nz_"):
                in_nd2nz_flag = True
                dataflow = i[0][6:]
            elif i[0].endswith("_nz2nd"):
                out_nz2nd_flag = True
                dataflow = i[0][:-6]
                cout_real = i[4]
                i = i[:4] + i[5:]
            else:
                dataflow = i[0]

            if dataflow == "aipp_conv2d":
                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag, crop_flag, c04_flag, aipp_format = i
                conv_v220_aipp_case(dataflow, conv_type, shape_in, shape_w, pads,
                                    strides, groups, bias_flag, crop_flag,
                                    c04_flag, aipp_format)
            elif dataflow == "aipp_conv2d_quant":
                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag, crop_flag, c04_flag, aipp_format, quant_scale, quant_offset = i
                conv_v220_aipp_case(dataflow, conv_type, shape_in, shape_w, pads,
                                    strides, groups, bias_flag, crop_flag,
                                    c04_flag, aipp_format, quant_scale,
                                    quant_offset)
            elif dataflow == "conv2d_single_op":
                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag = i
                conv_v220_single_op_case(conv_type, in_nd2nz_flag, out_nz2nd_flag,
                                         shape_in, shape_w, pads, strides, groups,
                                         bias_flag)
            elif dataflow == "conv2d_quant":
                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag, _, quant_scale, quant_offset = i
                conv_v220_fusion_case(dataflow,
                                      conv_type,
                                      in_nd2nz_flag,
                                      out_nz2nd_flag,
                                      shape_in,
                                      shape_w,
                                      pads,
                                      strides,
                                      groups,
                                      bias_flag,
                                      None,
                                      quant_scale=quant_scale,
                                      quant_offset=quant_offset,
                                      cout_real=cout_real)
            elif dataflow == "conv2d_dequant":
                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag, relu_mode, sqrt_mode = i
                conv_v220_fusion_case(dataflow,
                                      conv_type,
                                      in_nd2nz_flag,
                                      out_nz2nd_flag,
                                      shape_in,
                                      shape_w,
                                      pads,
                                      strides,
                                      groups,
                                      bias_flag,
                                      relu_mode,
                                      sqrt_mode=sqrt_mode,
                                      cout_real=cout_real)
            elif dataflow == "conv2d_requant":
                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag, relu_mode = i
                conv_v220_fusion_case(dataflow,
                                      conv_type,
                                      in_nd2nz_flag,
                                      out_nz2nd_flag,
                                      shape_in,
                                      shape_w,
                                      pads,
                                      strides,
                                      groups,
                                      bias_flag,
                                      relu_mode,
                                      cout_real=cout_real)
            else:
                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag = i
                conv_v220_fusion_case(dataflow,
                                      conv_type,
                                      in_nd2nz_flag,
                                      out_nz2nd_flag,
                                      shape_in,
                                      shape_w,
                                      pads,
                                      strides,
                                      groups,
                                      bias_flag,
                                      cout_real=cout_real)


    cceconf.te_set_version('Ascend920A')
    with op_context.OpContext():
        run_testcase(v220_case)
        run_testcase(v220_single_op_case)
        run_testcase(v220_input_nd2nz_case)
        run_testcase(v220_output_nz2nd_case)
        run_testcase(v220_aipp_case)


print("adding Conv2D v220 ut testcases")
# ut_case.add_cust_test_func(test_func=test_conv2d_v220)
