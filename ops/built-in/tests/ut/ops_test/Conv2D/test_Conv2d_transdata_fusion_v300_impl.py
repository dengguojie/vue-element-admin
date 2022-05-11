#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")


def test_conv2d_transdata_fusion_v300(test_arg):
    import tbe
    from impl.conv2d import conv2d_compute
    from impl.trans_data import trans_data_compute
    from tbe.common.context import op_context
    from te import platform as cceconf
    from te import tvm
    from topi import generic

    input_nd2nz_case = [
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
        ("nd2nz_conv2d", "float16", (1, 32, 32, 3), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, False),
    ]


    def conv_fusion_case(dataflow,
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
            "float16": "float16",
            "bfloat16": "float16",
            "int8": "int32"
        }
        bias_dtype = bias_dtype_dict[conv_type]

        with tvm.target.cce():
            fmap_ori = tvm.placeholder(shape_in,
                                       name='fmap_ori',
                                       dtype=conv_type)
            fmap = trans_data_compute(fmap_ori,
                                      None,
                                      src_format="NHWC",
                                      dst_format="NC1HWC0")

            weight = tvm.placeholder(shape_w_fracz,
                                     name='weight',
                                     dtype=conv_type,
                                     attrs={
                                         'ori_shape': shape_w,
                                         'ori_format': 'NCHW',
                                         'format': 'FRACTAL_Z_C04'
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

            sch = auto_schedule(out)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "conv2d_fusion",
            "tensor_list": tensor_list
        }
        tbe.dsl.build(sch, config)

    def run_testcase(config_dict):
        for i in config_dict:
            try:
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

                _, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag = i
                conv_fusion_case(
                    dataflow,
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
            except Exception as e:
                print("=" * 60 + "[run failed!]" + "=" * 60)
                print("case {}".format(i))
                print(e)

    def test_op_select_format():
        from impl.util import util_conv2d
        inputs = {"is_first_layer": False}
        weights = {}
        shape_fm = [1, 32, 16, 16]
        c0_optim_flag = True
        util_conv2d.v220_gen_param(inputs, weights, shape_fm, c0_optim_flag)
        util_conv2d.is_support_v300()

    cceconf.te_set_version("Ascend320")
    with op_context.OpContext():
        run_testcase(input_nd2nz_case)
        test_op_select_format()

print("adding Conv2D v300 transdata fusion testcases")
ut_case.add_cust_test_func('Ascend320', test_func=test_conv2d_transdata_fusion_v300)
ut_case.run(['Ascend320'])
