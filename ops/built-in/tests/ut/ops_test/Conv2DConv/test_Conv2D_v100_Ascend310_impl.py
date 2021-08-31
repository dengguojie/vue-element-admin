#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_v100(test_arg):
    """
    v200 ut case.
    """
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from tbe.dsl import auto_schedule
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.prelu import prelu_compute
    from impl.eltwise import eltwise_compute
    from impl.ascend_dequant_s16 import ascend_dequant_s16_compute
    from impl.ascend_requant_s16 import ascend_requant_s16_compute
    from impl.ascend_requant import ascend_requant_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    testcases = {
        "op_name": "conv_v200",
        "all": {
            # case name: ((fm_shape), (weight_shape), (paddings), (strides), data_flow, bias_flag, relu_flag, vector_flag)
            # data_flow
            # FP16 UB FUSION
            #   21:conv2d+relu
            #   22:Conv2d+Eltwise(Add)
            #   23:Conv2d+Eltwise(Add)+ReLU
            #   24:conv2d+LeakyRelu
            #   25:Conv2d+LeakyReLU+Eltwise(Add)
            #quant fusion
            # 6 :（1）conv2d+AscendDequant(relu)
            # 30 ：Conv2d+AscendDequant+AscendQuant
            # 31 :（4）Conv2d+AscendDequant+Eltwise+ReLU+AscendQuant
            # 32 :（5）Conv2d+AscendDequant+Eltwise+ReLU
            # 33 :（7）Conv2d+AscendDequant+Eltwise+ReLU(FP16)+AscendQuant double out
            # 34 :（2）Conv2d+AscendDequant+Eltwise+AscendQuant
            # 35 :（3）Conv2d+AscendDequant+Eltwise
            # 36 :（6）Conv2d+AscendDequant+Eltwise+AscendQuant double out

            # 37 :（1）Conv2d+AscendDequant+LeakyReLU+Eltwise
            # 38 :（2）conv2d+AscendDequant+LeakyRelu(FP16)+AscendQuant double out
            # 39 :（3）Conv2d+AscendDequant+LeakyReLU+Eltwise(FP16)+AscendQuant double out
            # 40 :（4）conv2d+AscendDequant+LeakyRelu
            # 41 :（5）Conv2d+AscendDequant+LeakyReLU+Eltwise+AscendQuant
            # 42 :（6）conv2d+AscendDequant+LeakyRelu+AscendQuant

            # 43 :（1）Conv2d+AscendDequant+Prelu+Eltwise
            # 44 :（2）conv2d+AscendDequant+Prelu(FP16)+AscendQuant double out
            # 45 :（3）Conv2d+AscendDequant+Prelu+Eltwise+AscendQuant double out
            # 46 :（4）conv2d+AscendDequant+Prelu
            # 47 :（5）Conv2d+AscendDequant+Prelu+Eltwise+AscendQuant
            # 48 :（6）conv2d+AscendDequant+Prelu+AscendQuant

            "conv_v100_bias_1_flow_21": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 21, 1, 0, 0),
            "conv_v100_bias_1_flow_22": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 22, 1, 0, 0),
            "conv_v100_bias_1_flow_23": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 23, 1, 0, 0),
            "conv_v100_bias_1_flow_24": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 24, 1, 0, 0),
            "conv_v100_bias_1_flow_25": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 25, 1, 0, 0),

            # relu
            "conv_v100_quant_bias_1_flow_6": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 6, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_30": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 30, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_31": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 31, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_34": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 34, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_35": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 35, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_32": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 32, 1, 0, 0),

            # leakyrelu
            "conv_v100_quant_bias_1_flow_37": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 37, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_40": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 40, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_41": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 41, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_42": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 42, 1, 0, 0),

            # prelu
            "conv_v100_quant_bias_1_flow_43": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 43, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_46": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 46, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_47": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 47, 1, 0, 0),
            "conv_v100_quant_bias_1_flow_48": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 48, 1, 0, 0),

            "conv_v100_bias_0_flow_21": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 21, 0, 0, 0),
            "conv_v100_bias_0_flow_22": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 22, 0, 0, 0),
            "conv_v100_bias_0_flow_23": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 23, 0, 0, 0),
            "conv_v100_bias_0_flow_24": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 24, 0, 0, 0),
            "conv_v100_bias_0_flow_25": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 25, 0, 0, 0),

            # relu
            "conv_v100_quant_bias_0_flow_6": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 6, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_30": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 30, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_31": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 31, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_34": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 34, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_35": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 35, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_32": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 32, 0, 0, 0),

            # leakyrelu
            "conv_v100_quant_bias_0_flow_37": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 37, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_40": ((1, 32, 7, 7), (32, 32, 2, 2),\
            [0, 0, 0, 0], [1, 1, 1, 1], 40, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_41": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 41, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_42": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 42, 0, 0, 0),

            # prelu
            "conv_v100_quant_bias_0_flow_43": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 43, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_46": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 46, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_47": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 47, 0, 0, 0),
            "conv_v100_quant_bias_0_flow_48": ((1, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 48, 0, 0, 0),
        }
    }


    def conv_v200_fusion_case(shape_in, shape_w, pads, strides, c_out, \
        orig_shape_w, data_flow, bias_flag, relu_flag, vector_flag):
        data_flow_fution_type_map_bias_true = {
            "unknown": 100,
            "0": 27,
            "6": 10,
            "21": 4,
            "22": 23,
            "23": 8,
            "24": 18,
            "25": 25,
            "30": 12,
            "31": 14,
            "32": 14,
            "34": 29,
            "35": 31,
            "37": 35,
            "40": 21,
            "41": 39,
            "42": 14,
            # 100 0000 1100 0010 0000 0000 0000  pattern_value:4,oplist_num:3-4,2-3
            "43": 67903488,
            # 100 0000 1000 0010 0000 0000 0000  pattern_value:4,oplist_num:2-4,2-3
            "46": 67641344,
            # 110 0000 1100 0010 0000 0000 0000 pattern_value:6,oplist_num:3-4,2-3
            "47": 101457920,
            # 110 0000 1000 0010 0000 0000 0000 pattern_value:6,oplist_num:2-4,2-3
            "48": 101195776
            }

        data_flow_fution_type_map_bias_false = {
            "unknown": 100,
            "0": 26,
            "6": 9,
            "21": 3,
            "22": 22,
            "23": 7,
            "24": 5,
            "25": 24,
            "30": 11,
            "31": 13,
            "32": 17,
            "34": 28,
            "35": 30,
            "37": 34,
            "40": 17,
            "41": 38,
            "42": 13,
            # 11 0000 1100 0010 0000 0000 0000  pattern_value:3,oplist_num:3-4,2-3
            "43": 51126272,
            # 11 0000 1000 0010 0000 0000 0000  pattern_value:3,oplist_num:2-4,2-3
            "46": 50864128,
            # 101 0000 1100 0010 0000 0000 0000 pattern_value:5,oplist_num:3-4,2-3
            "47": 84680704,
            # 101 0000 1000 0010 0000 0000 0000 pattern_value:5,oplist_num:2-4,2-3
            "48": 84418560
            }
        with tvm.target.cce():
            # conv2d
            dilations = [1, 1, 1, 1]
            if vector_flag:
                shape_req = (1, c_out, 1, 1, 16)
            else:
                shape_req = (1, 1, 1, 1, 1)
            shape_c = (1, c_out, 1, 1, 16)
            if data_flow in (21, 22, 23, 24,25):
                fm = tvm.placeholder(shape_in, name='fm', dtype='float16', attrs={'ori_format': 'NCHW'})
                filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='float16',
                                    attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
                if bias_flag:
                    bias_tensor = tvm.placeholder((c_out*16,), name='bias', dtype='float16')
                else:
                    bias_tensor = None
            else:
                fm = tvm.placeholder(shape_in, name='fm', dtype='int8', attrs={'ori_format': 'NCHW'})
                filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='int8',
                                        attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
                if bias_flag:
                    bias_tensor = tvm.placeholder((c_out*16,), name='bias', dtype='int32')
                else:
                    bias_tensor = None
            conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations)

            if data_flow in (6, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, \
                42, 43, 44, 45, 46, 47, 48):
                # conv + dequant
                deq16_reg = tvm.placeholder(shape_req, name='deq_reg', dtype='float16',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                if data_flow in (37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48):
                    relu_flag = False
                else:
                    relu_flag = True
                out = ascend_dequant_compute(conv_res, deq16_reg, None, sqrt_mode=False, relu_flag=relu_flag)

                fm2 = tvm.placeholder(out.shape, name='fmap2', dtype="float16", attrs={'ori_format': 'NCHW'})
                tensor_list = [fm, filter_w, deq16_reg]
                if data_flow == 30:
                    out = ascend_quant_compute(out, None, scale=0.1, offset=0.2, sqrt_mode=True)
                if data_flow in (31, 33):
                    res_add = eltwise_compute([fm2, out], None)
                    res_relu = leaky_relu_compute(res_add, None)
                    res_quant = ascend_quant_compute(res_relu, None, scale=0.1, offset=0.2, sqrt_mode=True)
                    out = res_quant
                    tensor_list.append(fm2)
                    if data_flow == 33:
                        out = [res_relu, res_quant]
                if data_flow == 32:
                    res_add = eltwise_compute([fm2, out], None)
                    res_relu = leaky_relu_compute(res_add, None)
                    out = res_relu
                    tensor_list.append(fm2)
                if data_flow in (34, 35, 36):
                    res_add = eltwise_compute([fm2, out], None)
                    out = res_add
                    tensor_list.append(fm2)
                    if data_flow in (34, 36):
                        res_quant = ascend_quant_compute(out, None, scale=0.1, offset=0.2, sqrt_mode=True)
                        out = res_quant
                    if data_flow == 36:
                        out = [res_add, res_quant]
                if data_flow in (37, 38, 39, 40, 41, 42):
                    out = leaky_relu_compute(out, None, negative_slope=0.1)
                    if data_flow == 37:
                        out = eltwise_compute([fm2, out], None)
                        tensor_list.append(fm2)
                    if data_flow in (38, 42):
                        res_quant = ascend_quant_compute(out, None, scale=0.1, offset=0.2, sqrt_mode=True)
                        if data_flow == 38:
                            out = [out, res_quant]
                        else:
                            out = res_quant
                    if data_flow in (39, 41):
                        res_add = eltwise_compute([fm2, out], None)
                        res_quant = ascend_quant_compute(res_add, None, scale=0.1, offset=0.2, sqrt_mode=True)
                        if data_flow == 39:
                            out = [out, res_quant]
                        else:
                            out = res_quant
                        tensor_list.append(fm2)
                if data_flow in (43, 44, 45, 46, 47, 48):
                    prelu_weight = tvm.placeholder((1, c_out, 1, 16), \
                        name='prelu_weight', dtype=out.dtype, \
                        attrs={'ori_shape': [c_out*16]})
                    tensor_list.append(prelu_weight)
                    out = prelu_compute(out, prelu_weight, None, \
                        kernel_name="prelu")
                    if data_flow == 43:
                        out = eltwise_compute([fm2, out], None)
                        tensor_list.append(fm2)
                    if data_flow in (44, 48):
                        res_quant = ascend_quant_compute(out, None, \
                            scale=0.1, offset=0.2, sqrt_mode=True)
                        if data_flow == 44:
                            out = [out, res_quant]
                        else:
                            out = res_quant
                    if data_flow in (45, 47):
                        res_add = eltwise_compute([fm2, out], None)
                        res_quant = ascend_quant_compute(res_add, None, \
                            scale=0.1, offset=0.2, sqrt_mode=True)
                        if data_flow == 45:
                            out = [out, res_quant]
                        else:
                            out = res_quant
                        tensor_list.append(fm2)
                if bias_flag:
                    tensor_list.append(bias_tensor)
                import collections.abc
                if isinstance(out, collections.abc.Sequence):
                    tensor_list.extend(out)
                else:
                    tensor_list.append(out)
                auto_sch_res = AutoScheduleOp(out)
                sch = auto_schedule(out)
            elif data_flow == 21:
                out = leaky_relu_compute(conv_res, None)
                auto_sch_res = AutoScheduleOp(out)
                sch = auto_schedule(out)
                tensor_list = [fm, filter_w, out]
            elif data_flow in (22, 23):
                fm2 = tvm.placeholder(conv_res.shape, name='fmap2', dtype="float16", attrs={'ori_format': 'NCHW'})
                out = eltwise_compute([conv_res, fm2], None)
                if data_flow == 23:
                    out = leaky_relu_compute(out, None)
                auto_sch_res = AutoScheduleOp(out)
                sch = auto_schedule(out)
                tensor_list = [fm, filter_w, fm2, out]
            elif data_flow in (24, 25):
                if data_flow == 25:
                    fm2 = tvm.placeholder(conv_res.shape, name='fmap2', dtype="float16", attrs={'ori_format': 'NCHW'})
                    conv_res = eltwise_compute([conv_res, fm2], None)
                out = leaky_relu_compute(conv_res, None, negative_slope=0.1)
                auto_sch_res = AutoScheduleOp(out)
                sch = auto_schedule(out)
                tensor_list = [fm, filter_w, out]
                if data_flow == 25:
                    tensor_list = [fm, filter_w, fm2, out]
            else:
                pass
            if bias_flag and data_flow not in (6, 30, 31, 32, 33, \
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48):
                tensor_list.append(bias_tensor)
            if bias_flag:
                fution_type = \
                data_flow_fution_type_map_bias_true.get(\
                    str(data_flow), data_flow_fution_type_map_bias_true["unknown"])
            else:
                fution_type = \
                data_flow_fution_type_map_bias_false.get(\
                    str(data_flow), \
                    data_flow_fution_type_map_bias_false["unknown"])
            print(auto_sch_res.fusion_type, fution_type, data_flow)
            assert auto_sch_res.fusion_type == fution_type
        return sch, tensor_list


    def conv_v200_fusion(fm_shape, filter, pads, strides, data_flow, \
        bias_flag, relu_flag, vector_flag, kernel_name):
        from te.platform.cce_policy import disableL2
        disableL2()
        block_size_k = 32
        block_size_n = 16
        batch, channel, height, weight = fm_shape
        C0 = 32
        if data_flow in (21, 22, 23, 24,25):
            block_size_k = 16
            C0 = 16
        C1 = (channel + C0 - 1) // C0
        shape_in = (batch, C1, height, weight, C0)

        out_channel = filter[0]
        in_channel_weight = ((filter[1] + block_size_k - 1) // block_size_k) * block_size_k
        filter_h = filter[2]
        filter_w = filter[3]

        if data_flow in (21, 22, 23, 24,25):
            c_out = (out_channel + block_size_n - 1) // block_size_n
        else:
            c_out = (out_channel + block_size_k - 1) // block_size_k * block_size_k
            c_out = (c_out + block_size_n - 1) // block_size_n
        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                c_out, block_size_n, block_size_k)

        sch, tensor_list = conv_v200_fusion_case(shape_in, shape_w, pads, \
            strides, c_out, filter, data_flow, bias_flag, relu_flag, vector_flag)

        config = {"print_ir": False,
                "need_build": True,
                "name": kernel_name,
                "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def run_testcase():
        testcases_for_all = testcases["all"]
        for key in testcases_for_all:
            conv_v200_fusion(*testcases_for_all[key], key)
            print("[passed: %s]" % key)

    def set_ddk_version(version):
        if version == "v100":
            ddk_info = "Ascend310"
        else:
            ddk_info = "Ascend310"

        cce_conf.cce_conf.te_set_version(ddk_info)

    """
    The UT for cce Test_conv2d_v200
    """

    print("---------------------------------------------------")
    set_ddk_version("v100")
    print("[ UNITTEST START conv2d v200 ddk is Ascend310]")

    run_testcase()
    set_ddk_version("v100")

print("adding Conv2D V100 Ascend310 testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_v100)
