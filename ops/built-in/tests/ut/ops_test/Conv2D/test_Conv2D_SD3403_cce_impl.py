#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_v200(test_arg):
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
    from tbe.dsl.compute.conv_compute import ConvParam
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.conv2d import _conv_layer_cce
    from impl.ascend_dequant_s16 import ascend_dequant_s16_compute
    from impl.ascend_requant_s16 import ascend_requant_s16_compute
    from impl.ascend_requant import ascend_requant_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    testcases = {
        "op_name": "conv_v200",
        "all": {
            # case name: ((fm_shape), (weight_shape), (paddings), (strides), data_flow, bias_flag, relu_flag, vector_flag)
            # data_flow
            #   10: s8s8->s32                         int8 conv
            #   20: fp16fp16->fp16                    fp16 conv
            #   30：fp16fp16->fp16                    fp16 group conv
            #   0: s32->s8                            conv + requant
            #   1: s32->s16                           conv + dequants16
            #   2: s32->s16->s8                       conv + dequants16 + requants16(relu) singleout
            #   3: s32->s16->s8/s16 double output     conv + dequants16 + requants16(relu) doubleout
            #   4: s32->fp16                          conv + dequant
            #   5: s32->s16->s8                       conv + dequants16 + requants16 singleout
            #   6: s32->s16->s8/s16 double output     conv + dequants16 + requants16 doubleout

            "conv_v200_bias_0_flow_10": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 10, 0, 0, 0),
            "conv_v200_bias_1_flow_10": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 10, 1, 0, 0),
            "conv_v200_bias_0_flow_20": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 20, 0, 0, 0),
            "conv_v200_bias_1_flow_20": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 20, 1, 0, 0),
            "conv_v200_bias_1_flow_30": ((1, 120, 14, 14), (480, 40, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 30, 1, 0, 0, 3),
            "conv_v200_relu_0_bias_0_vector_0_flow_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 0, 0, 0),
            "conv_v200_relu_0_bias_0_vector_1_flow_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 1, 0, 1),
            "conv_v200_relu_0_bias_1_vector_0_flow_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 0, 1, 0),
            "conv_v200_relu_0_bias_1_vector_1_flow_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 1, 1, 1),
            "conv_v200_relu_1_bias_0_vector_1_flow_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 0, 0, 1),
            "conv_v200_relu_1_bias_1_vector_1_flow_1": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 1, 1, 1),
            "conv_v200_relu_1_bias_1_vector_1_flow_1_cout1": ((1, 32, 7, 1), (32, 32, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1, 1, 1, 1),
            "conv_v200_relu_0_bias_1_flow_2": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 2, 0, 1, 1),
            "conv_v200_relu_1_bias_0_flow_2": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 2, 1, 0, 1),
            "conv_v200_relu_0_bias_0_flow_3": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 3, 0, 0, 1),
            "conv_v200_relu_1_bias_0_flow_3": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 3, 1, 0, 1),
            "conv_v200_relu_0_vector_1_flow_4": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 4, 0, 0, 1),
            "conv_v200_relu_1_vector_1_flow_4": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 4, 1, 0, 1),
            "conv_v200_relu_1_vector_1_flow_4_cout1": ((1, 32, 7, 1), (32, 32, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 4, 1, 0, 1),
            "conv_v200_relu_0_bias_1_flow_5": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 5, 0, 1, 1),
            "conv_v200_relu_1_bias_0_flow_5": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 5, 1, 0, 1),
            "conv_v200_relu_0_bias_1_flow_6": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 6, 0, 0, 1),
            "conv_v200_relu_1_bias_0_flow_6": ((2, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 6, 1, 0, 1),
        }
    }


    def conv_v200_fusion_case(shape_in, shape_w, pads, strides, c_out, \
        orig_shape_w, data_flow, bias_flag, relu_flag, vector_flag):
        data_flow_fution_type_map_bias_true = {
            "unknown": 100,
            "1": 46,
            "2": 47,
            "3": 48,
            "4": 9,
            "5": 47,
            "6": 48,
            }

        data_flow_fution_type_map_bias_false = {
            "unknown": 100,
            "1": 46,
            "2": 47,
            "3": 48,
            "4": 9,
            "5": 47,
            "6": 48,
            }
        with tvm.target.cce():
            # conv2d
            dilations = [1, 1, 1, 1]
            if vector_flag:
                shape_req = (1, c_out, 1, 1, 16)
            else:
                shape_req = (1, 1, 1, 1, 1)
            shape_c = (1, c_out, 1, 1, 16)
            fm = tvm.placeholder(shape_in, name='fm', dtype='int8', attrs={'ori_format': 'NCHW'})
            filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='int8',
                                    attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
            # u8/s8 -> s32
            conv_res = conv2d_compute(fm, filter_w, None, None, None, strides, pads, dilations)

            if data_flow == 0:
                # conv + requant
                vreq_reg = tvm.placeholder(shape_req, name='vreq_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                out = ascend_requant_compute(conv_res, vreq_reg, None, relu_flag=True)
                auto_sch_res = AutoScheduleOp(out)
                sch = auto_schedule(out)
                tensor_list = [fm, filter_w, vreq_reg, out]
            elif data_flow == 1:
                # conv + dequant_s16
                vdeq_reg = tvm.placeholder(shape_req, name='vdeq_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias', dtype='int16')
                else:
                    bias_tensor = None
                out = ascend_dequant_s16_compute(conv_res, vdeq_reg, bias_tensor, None, relu_flag=relu_flag)
                tiling = {'AL0_matrix':[4, 9, 16, 16], 'CL0_matrix': [1, 4, 16, 16, 1], 'CUB_matrix': [1, 4, 16, 16], 
                          'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0, 'BL0_matrix': [9, 1, 16, 16],
                          'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1, 
                          'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 1, 'CUB_pbuffer': 1, 'UBG_pbuffer': 1},
                          'n_bef_batch_flag': 0, 'AL1_shape': [], 'BL1_shape': None, 'block_dim': [1, 1, 1, 1], 'CUB_channel_wise_flag': False}
                ConvParam.tiling = tiling
                auto_sch_res = AutoScheduleOp(out)
                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq_reg, bias_tensor, out]
                else:
                    tensor_list = [fm, filter_w, vdeq_reg, out]
            elif data_flow == 2:
                # conv + dequant_s16 + requants16(relu) singleout
                vdeq16_reg = tvm.placeholder(shape_req, name='vdeqs16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias', dtype='int16')
                else:
                    bias_tensor = None
                requant_s16 = ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)
                conv16_reg = tvm.placeholder(shape_c, name='conv16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                fm2 = tvm.placeholder(conv_res.shape, name='fm2', dtype='int16')
                out = ascend_requant_s16_compute(requant_s16, conv16_reg, fm2, None, None, dual_output=False, relu_flag=True)
                auto_sch_res = AutoScheduleOp(out[0])
                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out[0]]
                else:
                    tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out[0]]
            elif data_flow == 3:
                # conv + dequant_s16 + requants16(relu) doubleout
                vdeq16_reg = tvm.placeholder(shape_req, name='vdeqs16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias', dtype='int16')
                else:
                    bias_tensor = None
                requant_s16 = ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)
                conv16_reg = tvm.placeholder(shape_c, name='conv16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                fm2 = tvm.placeholder(conv_res.shape, name='fm2', dtype='int16')
                out = ascend_requant_s16_compute(requant_s16, conv16_reg, fm2, None, None, dual_output=True, relu_flag=True)
                double_out_res = [requant_s16, out[0]]
                out_f16, out_s8 = double_out_res
                res_out_fp16 = tvm.compute(out_f16.shape, \
                    lambda i, j, k, l: out_f16(i, j, k, l), \
                    name='res_out_fp16', \
                    tag='res_out_fp16', \
                    attrs={"addr_type": 0})
                virtual_res = tvm.compute(out_s8.shape,
                                        lambda i, j, k, l:
                                        out_s8(i, j, k, l) +
                                        res_out_fp16(i, (j*32 + l) // 16, k,
                                                    (j*32 + l) % 16),
                                        name='conv_virtual_res',
                                        tag="conv_virtual_res",
                                        )
                outputs = [virtual_res, res_out_fp16, out_s8]
                auto_sch_res = AutoScheduleOp(outputs[0])
                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out[0], out[1]]
                else:
                    tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out[0], out[1]]
            elif data_flow == 4:
                # conv + dequant
                deq16_reg = tvm.placeholder(shape_req, name='deq_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                out = ascend_dequant_compute(conv_res, deq16_reg, None, sqrt_mode=False, relu_flag=True)
                tiling = {'AL0_matrix':[4, 9, 16, 16], 'CL0_matrix': [1, 4, 16, 16, 1], 'CUB_matrix': [1, 4, 16, 16], 
                          'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0, 'BL0_matrix': [9, 1, 16, 16],
                          'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 1, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1, 
                          'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 1, 'CUB_pbuffer': 1, 'UBG_pbuffer': 1},
                          'n_bef_batch_flag': 0, 'AL1_shape': [], 'BL1_shape': None, 'block_dim': [1, 1, 1, 1], 'CUB_channel_wise_flag': False}
                ConvParam.tiling = tiling
                auto_sch_res = AutoScheduleOp(out)
                sch = auto_schedule(out)
                tensor_list = [fm, filter_w, deq16_reg, out]
            elif data_flow == 5:
                # conv + dequant_s16 + requants16 singleout
                vdeq16_reg = tvm.placeholder(shape_req, name='vdeqs16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias', dtype='int16')
                else:
                    bias_tensor = None
                requant_s16 = ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)
                conv16_reg = tvm.placeholder(shape_c, name='conv16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                fm2 = tvm.placeholder(conv_res.shape, name='fm2', dtype='int16')
                out = ascend_requant_s16_compute(requant_s16, conv16_reg, fm2, None, None, dual_output=False, relu_flag=False)
                auto_sch_res = AutoScheduleOp(out[0])
                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out[0]]
                else:
                    tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out[0]]
            elif data_flow == 6:
                # conv + dequant_s16 + requants16 doubleout
                vdeq16_reg = tvm.placeholder(shape_req, name='vdeqs16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias', dtype='int16')
                else:
                    bias_tensor = None
                requant_s16 = ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)
                conv16_reg = tvm.placeholder(shape_c, name='conv16_reg', dtype='uint64',
                    attrs={'ori_shape': [c_out*16 if vector_flag else 1]})
                fm2 = tvm.placeholder(conv_res.shape, name='fm2', dtype='int16')
                out = ascend_requant_s16_compute(requant_s16, conv16_reg, fm2, None, None, dual_output=True, relu_flag=False)
                double_out_res = [requant_s16, out[0]]
                out_f16, out_s8 = double_out_res
                res_out_fp16 = tvm.compute(out_f16.shape, \
                    lambda i, j, k, l: out_f16(i, j, k, l), \
                    name='res_out_fp16', \
                    tag='res_out_fp16', \
                    attrs={"addr_type": 0})
                virtual_res = tvm.compute(out_s8.shape,
                                        lambda i, j, k, l:
                                        out_s8(i, j, k, l) +
                                        res_out_fp16(i, (j*32 + l) // 16, k,
                                                    (j*32 + l) % 16),
                                        name='conv_virtual_res',
                                        tag="conv_virtual_res",
                                        )
                outputs = [virtual_res, res_out_fp16, out_s8]
                auto_sch_res = AutoScheduleOp(outputs[0])
                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out[0], out[1]]
                else:
                    tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out[0], out[1]]
            if bias_flag:
                fution_type = \
                data_flow_fution_type_map_bias_true.get(\
                    str(data_flow), data_flow_fution_type_map_bias_true["unknown"])
            else:
                fution_type = \
                data_flow_fution_type_map_bias_false.get(\
                    str(data_flow), \
                    data_flow_fution_type_map_bias_false["unknown"])
            assert auto_sch_res.fusion_type == fution_type
        return sch, tensor_list


    def conv_v200(fm_shape, filter, pads, strides, data_flow, \
        bias_flag, relu_flag, vector_flag, kernel_name_val):
        if data_flow == 10:
            fm_type = "int8"
            weight_type = "int8"
            output_type = "int32"
        else:
            fm_type = "float16"
            weight_type = "float16"
            output_type = "float16"
        padh = pads[0]
        padw = pads[2]
        strideh = strides[2]
        stridew = strides[3]

        if bias_flag == 1:
            bias_flag = True
        else:
            bias_flag = False

        _conv_layer_cce(fm_shape, filter, fm_type, weight_type, output_type,
                    padh, padw, strideh, stridew, bias=bias_flag, kernel_name=kernel_name_val)

    def conv_v200_group(fm_shape, filter, pads, strides, data_flow, \
        bias_flag, relu_flag, vector_flag, groups_num, kernel_name_val):
        if data_flow == 10:
            fm_type = "int8"
            weight_type = "int8"
            output_type = "int32"
        else:
            fm_type = "float16"
            weight_type = "float16"
            output_type = "float16"
        padh = pads[0]
        padw = pads[2]
        strideh = strides[2]
        stridew = strides[3]

        if bias_flag == 1:
            bias_flag = True
        else:
            bias_flag = False

        _conv_layer_cce(fm_shape, filter, fm_type, weight_type, output_type,
                        padh, padw, strideh, stridew, groups=groups_num, bias=bias_flag, kernel_name=kernel_name_val)

    def conv_v200_fusion(fm_shape, filter, pads, strides, data_flow, \
        bias_flag, relu_flag, vector_flag, kernel_name):
        from te.platform.cce_policy import disableL2
        disableL2()
        block_size_k = 32
        block_size_n = 16
        batch, channel, height, weight = fm_shape
        C0 = 32
        C1 = (channel + C0 - 1) // C0
        shape_in = (batch, C1, height, weight, C0)

        out_channel = filter[0]
        in_channel_weight = ((filter[1] + block_size_k - 1) // block_size_k) * block_size_k
        filter_h = filter[2]
        filter_w = filter[3]

        if data_flow == 1:
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
            if testcases_for_all[key][-5] == 30:
                conv_v200_group(*testcases_for_all[key], key)
            elif testcases_for_all[key][-4] == 10 or testcases_for_all[key][-4] == 20:
                conv_v200(*testcases_for_all[key], key)
            else:
                conv_v200_fusion(*testcases_for_all[key], key)
            print("[passed: %s]" % key)

    """
    The UT for cce Test_conv2d_v200
    """
    print("---------------------------------------------------")
    cce_conf.te_set_version('SD3403')
    print("[ UNITTEST START conv2d SD3403]")

    run_testcase()
    cce_conf.te_set_version('Ascend310')

print("adding Conv2D v200 cce ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_v200)

if __name__ == '__main__':
    # ut_case.add_cust_test_func(test_func=test_leakyrelu_depthwise_fusion_testcase)
    ut_case.run("Ascend310")
    ut_case.run("Ascend910A")
    exit(0)
