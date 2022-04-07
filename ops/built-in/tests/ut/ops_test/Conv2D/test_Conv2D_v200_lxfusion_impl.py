#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_v200_lxfusion(test_arg):
    from functools import reduce
    from impl.conv2d import conv2d_compute
    from tbe import tvm
    from tbe.dsl import auto_schedule
    from tbe.dsl import build
    from tbe.common.platform.platform_info import set_current_compile_soc_info
    from impl.ascend_dequant_s16 import ascend_dequant_s16_compute
    from impl.ascend_requant_s16 import ascend_requant_s16_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_requant import ascend_requant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.eltwise import eltwise_compute
    from impl.leaky_relu import leaky_relu_compute
    from tbe.common.buildcfg.compatible_interface import set_L1_info
    from tbe.tvm.buffer_manager import RemappedBuffer
    from tbe.tvm.buffer_manager import get_buffer_manager
    from tbe.common.buildcfg import build_config
    # l1_fusion_type = 0
    # fm_addr_type = 0
    # out_addr_type = 0
    # out_16_addr_type = 0
    # eltwise_addr_type = 0

    # l1_fusion_type = 0
    # fm_addr_type = 1
    # out_addr_type = 1
    # out_16_addr_type = 1
    # eltwise_addr_type = 1

    l1_fusion_type = 1
    fm_addr_type = 1
    out_addr_type = 1
    out_16_addr_type = 1
    eltwise_addr_type = 1

    l1_space = 0
    # [0, 32, 96, 128, 512, 1024] for weight space.

    dict0 = {
            "l1_fusion_type": l1_fusion_type,
            "l1_space": l1_space,

            "fm_addr_type": fm_addr_type,
            "out_addr_type": out_addr_type,
            "out_16_addr_type": out_16_addr_type,
            "eltwise_addr_type": eltwise_addr_type,

            "fm_total_shape": [],

            "eltwise_total_shape": [],
            "eltwise_shape": [],

            "ws_s8_valid_shape": [],
            "ws_16_valid_shape": [],
    }

    dict1 = {
            #===============read select=====================
            "l1_fusion_type": l1_fusion_type,
            "l1_space": l1_space,

            "fm_addr_type": fm_addr_type,
            "out_addr_type": out_addr_type,
            "out_16_addr_type": out_16_addr_type,
            "eltwise_addr_type": eltwise_addr_type,

            "fm_total_shape": [2, 2, 64, 32, 32], # thread_shape: [2, 2, 32, 32, 32]

            "eltwise_total_shape": [2, 4, 64*32, 16],
            "eltwise_shape": [2, 4, 32*32, 16], # splited fm2 shape for conv2d input

            "ws_s8_valid_shape": [],
            "ws_16_valid_shape": [],
    }

    dict2 = {
            #============writeselect only in H======================
            "l1_fusion_type": l1_fusion_type,
            "l1_space": l1_space,

            "fm_addr_type": fm_addr_type,
            "out_addr_type": out_addr_type,
            "out_16_addr_type": out_16_addr_type,
            "eltwise_addr_type": eltwise_addr_type,

            "fm_total_shape": [],

            "eltwise_total_shape": [],
            "eltwise_shape": [],

            "ws_s8_valid_shape": [2, 2, 64*32, 32],
            "ws_16_valid_shape": [2, 4, 64*32, 16],
    }

    v200_l1fusion_testcase = {
        # format: [dataflow_v200,
        #          shape_in,
        #          shape_w,
        #          pads,
        #          strides,
        #          offset_d,
        #          bias_flag,
        #          relu_flag,
        #          (reqs16_relu_flag),
        #          (quant_scale),
        #          (quant_offset),
        #          l1fusion_dict]

        # [dataflow_v200]
        # 0 dequant
        # 1 dequant + add + (relu) + quant singleout
        # 2 dequant + add + (relu) + quant dualout
        # 3 requant
        # 4 dequants16
        # 5 dequants16 + requants16 singleout
        # 6 dequants16 + requants16 dualout

        # [reqs16_relu_flag]
        # optional for requants16 fusion.

        # [l1fusion_dict]
        # {"l1_fusion_type": l1_fusion_type, # -1: no L1 fusion 0: depth fusion 1: breadth fusion

        #  "fm_addr_type": fm_addr_type, # 0: DDR 1: L1
        #  "out_addr_type": out_addr_type,
        #  "out_16_addr_type": out_16_addr_type,
        #  "eltwise_addr_type": eltwise_addr_type,

        #  "fm_total_shape": [],

        #   "eltwise_shape": [],
        #   "eltwise_total_shape": [],

        #   "ws_s8_valid_shape": [],
        #   "ws_16_valid_shape": [],}

        "3796CS": {
            "st": (
            #====================L1/DDR in L1/DDR out========================================
            [0, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict0],

            [1, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, 0, dict0],

            [2, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, 0, dict0],

            [3, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict0],

            [4, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict0],

            [5, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, dict0],

            [6, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, dict0],

            #====================fm valid shape + fm2 readselect in==========================
            [0, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict1],

            [1, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, 0, dict1],

            [2, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, 0, dict1],

            [3, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict1],

            [4, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict1],

            [5, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, dict1],

            [6, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, dict1],

            #============================writeselect====================================
            [0, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict2],

            [1, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, 0, dict2],

            [2, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, 0, dict2],

            [3, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict2],

            [4, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, dict2],

            [5, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, dict2],

            [6, (2, 64, 32, 32), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), 0, 1, 1, 1, dict2],

            ),
        },
    }


    def conv_v200_fusion_case(dataflow_v200, l1fusion_dict,
                              shape_in, shape_w, pads, strides, offset_d,
                              bias_flag, relu_flag, vector_flag, reqs16_relu_flag=True,
                              quant_scale=1, quant_offset=0, kernel_name="v200_l1fusion"):
        """
        dataflow_v200:
            0 dequant
            1 dequant + add + (relu) + quant singleout
            2 dequant + add + (relu) + quant dualout
            3 requant
            4 dequants16
            5 dequants16 + requants16 singleout
            6 dequants16 + requants16 dualout

        shape_in: fmap shape [NCHW]

        shape_w: weight shape [NCHW]

        pads: [pad_up, pad_down, pad_left, pad_right]

        strides: [strideh, stridew]

        offset_d: offset_x for set padding value and mad compute

        bias_flag: 1 for with bias, 0 for no bias

        relu_flag: 1 for with relu, 0 for no relu

        vector_flag: 1 for vector, 0 for scalar
        """
        Ni, Ci, Hi, Wi = shape_in
        Co, _, Hk, Wk = shape_w

        Ci1 = (Ci + 31) // 32
        Ci0 = 32

        # May not be appropriate, for now Co is aligned by 32 when output dtype is int8.
        if dataflow_v200 in (1, 2, 3, 5, 6):
            Co = ((Co + 31) // 32)*32

        Co1 = (Co + 15) // 16
        Co0 = 16

        shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
        shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        if vector_flag:
            shape_scale = (1, Co1, 1, 1, 16)
        else:
            shape_scale = (1, 1, 1, 1, 1)

        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]

        l1_fusion_type = l1fusion_dict["l1_fusion_type"]
        l1_space = l1fusion_dict["l1_space"]

        fm_addr_type = l1fusion_dict["fm_addr_type"]
        eltwise_addr_type = l1fusion_dict["eltwise_addr_type"]
        out_addr_type = l1fusion_dict["out_addr_type"]
        out_16_addr_type = l1fusion_dict["out_16_addr_type"]

        fm_total_shape = l1fusion_dict["fm_total_shape"]
        eltwise_total_shape = l1fusion_dict["eltwise_total_shape"]
        eltwise_shape = l1fusion_dict["eltwise_shape"]

        ws_s8_valid_shape = l1fusion_dict["ws_s8_valid_shape"]
        ws_16_valid_shape = l1fusion_dict["ws_16_valid_shape"]

        set_L1_info("op_L1_space", l1_space*1024)
        set_L1_info("L1_fusion_enabled", True)

        # L1 valid size: allocate AL1 memory size by lxfusion.
        L1_valid_size = reduce((lambda x, y: x*y), shape_in)

        #==============call pass interface======================
        buffer_manager = get_buffer_manager()
        buffer_manager.set_l1_fusion_type(int(l1_fusion_type))
        scope_dict = {0: "global", 1: "local.L1_Fusion"}
        rb_fmap = RemappedBuffer(0, scope_dict[fm_addr_type], fm_total_shape, shape_in_5HD)

        with build_config(enable_L1_fusion = int(l1_fusion_type)):
            with tvm.target.cce():
                if fm_addr_type:
                    if fm_total_shape == []:
                        fm_total_shape = shape_in_5HD

                fm = tvm.placeholder(shape_in_5HD, name="fm", dtype="int8", attrs={"ori_format": "NCHW",
                                                                                "L1_valid_size": L1_valid_size,
                                                                                "L1_addr_flag": 1})
                buffer_manager.set_remapped_buffers([rb_fmap])
                buffer_manager.set_tensor_list([fm])

                filter_w = tvm.placeholder(shape_w_fracz, name="filter_w", dtype="int8",
                                        attrs={"ori_shape": shape_w, "ori_format": "NCHW"})

                if dataflow_v200 in (0, 1, 2, 3):
                    if bias_flag:
                        bias_tensor = tvm.placeholder((Co1*16, ), name="bias_tensor", dtype="int32")
                    else:
                        bias_tensor = None
                    conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=offset_d)
                    vdeq = tvm.placeholder(shape_scale, name="vreq_reg", dtype="uint64",
                                        attrs={"ori_shape": [Co1*16 if vector_flag else 1]})
                    if dataflow_v200 == 0:
                        # conv + dequant
                        out = ascend_dequant_compute(conv_res, vdeq, None, sqrt_mode=False, relu_flag=relu_flag)
                        if bias_flag:
                            tensor_list = [fm, filter_w, bias_tensor, vdeq, out]
                            out_index = 4
                        else:
                            tensor_list = [fm, filter_w, vdeq, out]
                            out_index = 3

                    elif dataflow_v200 == 1:
                        # conv + dequant + add + relu + quant singleout
                        dequant = ascend_dequant_compute(conv_res, vdeq, None, sqrt_mode=False, relu_flag=relu_flag)
                        if eltwise_addr_type:
                            if eltwise_total_shape == []:
                                eltwise_total_shape = dequant.shape

                        fm2 = tvm.placeholder(dequant.shape, name="fm2", dtype="float16")
                        add = eltwise_compute([dequant, fm2], None)
                        relu = leaky_relu_compute(add, None)
                        out = ascend_quant_compute(relu, {"addr_type": out_addr_type}, scale=quant_scale, offset=quant_offset, sqrt_mode=False)

                        if bias_flag:
                            tensor_list = [fm, filter_w, bias_tensor, vdeq, fm2, out]
                            fm2_index = 4
                            out_index = 5
                        else:
                            tensor_list = [fm, filter_w, vdeq, fm2, out]
                            fm2_index = 3
                            out_index = 4

                    elif dataflow_v200 == 2:
                        # conv + dequant + add + relu + quant dualout
                        dequant = ascend_dequant_compute(conv_res, vdeq, None, sqrt_mode=False, relu_flag=relu_flag)
                        if eltwise_addr_type:
                            if eltwise_total_shape == []:
                                eltwise_total_shape = dequant.shape

                        fm2 = tvm.placeholder(dequant.shape, name="fm2", dtype="float16")
                        add = eltwise_compute([dequant, fm2], None)
                        relu = leaky_relu_compute(add, None)
                        quant = ascend_quant_compute(relu, {"addr_type": out_addr_type}, scale=quant_scale, offset=quant_offset, sqrt_mode=False)

                        out_relu = relu
                        out_quant = quant
                        out = [out_relu, out_quant]

                        if bias_flag:
                            tensor_list = [fm, filter_w, bias_tensor, vdeq, fm2, out_relu, out_quant]
                            fm2_index = 4
                            out_relu_index = 5
                            out_quant_index = 6
                        else:
                            tensor_list = [fm, filter_w, vdeq, fm2, out_relu, out_quant]
                            fm2_index = 3
                            out_relu_index = 4
                            out_quant_index = 5

                    elif dataflow_v200 == 3:
                        # conv + requant
                        out = ascend_requant_compute(conv_res, vdeq, None, relu_flag=relu_flag)
                        if bias_flag:
                            tensor_list = [fm, filter_w, bias_tensor, vdeq, out]
                            out_index = 4
                        else:
                            tensor_list = [fm, filter_w, vdeq, out]
                            out_index = 3

                else:
                    conv_res = conv2d_compute(fm, filter_w, None, None, None, strides, pads, dilations)

                    if dataflow_v200 == 4:
                        # conv + dequants16
                        vdeq_reg = tvm.placeholder(shape_scale, name="vdeq_reg", dtype="uint64",
                                                attrs={"ori_shape": [Co1*16 if vector_flag else 1]})

                        if bias_flag:
                            bias_tensor = tvm.placeholder((1, Co1, 1, 1, 16), name="bias_s16", dtype="int16")
                        else:
                            bias_tensor = None

                        out = ascend_dequant_s16_compute(conv_res, vdeq_reg, bias_tensor, None, relu_flag=relu_flag)

                        if bias_flag:
                            tensor_list = [fm, filter_w, vdeq_reg, bias_tensor, out]
                            out_index = 4
                        else:
                            tensor_list = [fm, filter_w, vdeq_reg, out]
                            out_index = 3

                    elif dataflow_v200 == 5:
                        # conv + dequant_s16 + requant_s16(relu) singleout
                        vdeq16_reg = tvm.placeholder(shape_scale, name="vdeqs16_reg", dtype="uint64",
                                                    attrs={"ori_shape": [Co1*16 if vector_flag else 1]})

                        if bias_flag:
                            bias_tensor = tvm.placeholder((1, Co1, 1, 1, 16), name="bias_s16", dtype="int16")
                        else:
                            bias_tensor = None

                        dequant_s16 = ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)

                        # vadd(relu) + s16———>s8
                        conv16_reg = tvm.placeholder(shape_scale, name="conv16_reg", dtype="uint64",
                                                    attrs={"ori_shape": [Co1*16 if vector_flag else 1]})
                        if eltwise_addr_type:
                            if eltwise_total_shape == []:
                                eltwise_total_shape = dequant_s16.shape

                        fm2 = tvm.placeholder(dequant_s16.shape, name="fm2", dtype="int16")
                        out = ascend_requant_s16_compute(dequant_s16, conv16_reg, fm2, None, None,
                                                        dual_output=False, relu_flag=reqs16_relu_flag)
                        out = out[0]

                        if bias_flag:
                            tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out]
                            fm2_index = 5
                            out_index = 6
                        else:
                            tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out]
                            fm2_index = 4
                            out_index = 5

                    elif dataflow_v200 == 6:
                        # conv + dequant_s16 + requant_s16(relu) dualout
                        vdeq16_reg = tvm.placeholder(shape_scale, name="vdeqs16_reg", dtype="uint64",
                                                    attrs={"ori_shape": [Co1*16 if vector_flag else 1]})

                        if bias_flag:
                            bias_tensor = tvm.placeholder((1, Co1, 1, 1, 16), name="bias_s16", dtype="int16")
                        else:
                            bias_tensor = None

                        dequant_s16 = ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)

                        # vadd(relu) + s16———>s8
                        conv16_reg = tvm.placeholder(shape_scale, name="conv16_reg", dtype="uint64",
                                                    attrs={"ori_shape": [Co1*16 if vector_flag else 1]})
                        if eltwise_addr_type:
                            if eltwise_total_shape == []:
                                eltwise_total_shape = dequant_s16.shape

                        fm2 = tvm.placeholder(dequant_s16.shape, name="fm2", dtype="int16")
                        res_s8, res_s16 = ascend_requant_s16_compute(dequant_s16, conv16_reg, fm2, None, None,
                                                        dual_output=True, relu_flag=reqs16_relu_flag)
                        out_s8 = res_s8
                        out_s16 = res_s16

                        out = [out_s8, out_s16]

                        if bias_flag:
                            tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out_s8, out_s16]
                            fm2_index = 5
                            out_s8_index = 6
                            out_s16_index = 7
                        else:
                            tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out_s8, out_s16]
                            fm2_index = 4
                            out_s8_index = 5
                            out_s16_index = 6

                #====================set pass info===================================
                rb_list = []
                # each tensor in build tensor list needs to be set remap info.
                for i, _ in enumerate(tensor_list):
                    rb = RemappedBuffer(i, "global", (), ())
                    rb_list.append(rb)

                rb_list[0] = rb_fmap

                if dataflow_v200 in (1, 2, 5, 6): # with fm2
                    rb_fm2 = RemappedBuffer(fm2_index, scope_dict[eltwise_addr_type], eltwise_total_shape, eltwise_shape)
                    rb_list[fm2_index] = rb_fm2

                if dataflow_v200 in (0, 4): # singleout fp16/s16
                    rb_out = RemappedBuffer(out_index, scope_dict[out_addr_type], ws_16_valid_shape, list(i.value for i in out.shape))
                    rb_list[out_index] = rb_out
                elif dataflow_v200 in (1, 3, 5): # singleout s8
                    rb_out = RemappedBuffer(out_index, scope_dict[out_addr_type], ws_s8_valid_shape, list(i.value for i in out.shape))
                    rb_list[out_index] = rb_out
                elif dataflow_v200 == 2:
                    rb_out_relu = RemappedBuffer(out_relu_index, scope_dict[out_16_addr_type], ws_16_valid_shape, list(i.value for i in out_relu.shape))
                    rb_out_quant = RemappedBuffer(out_quant_index, scope_dict[out_addr_type], ws_s8_valid_shape, list(i.value for i in out_quant.shape))
                    rb_list[out_relu_index] = rb_out_relu
                    rb_list[out_quant_index] = rb_out_quant
                else:
                    rb_out_s16 = RemappedBuffer(out_s16_index, scope_dict[out_16_addr_type], ws_16_valid_shape, list(i.value for i in out_s16.shape))
                    rb_out_s8 = RemappedBuffer(out_s8_index, scope_dict[out_addr_type], ws_s8_valid_shape, list(i.value for i in out_s8.shape))
                    rb_list[out_s16_index] = rb_out_s16
                    rb_list[out_s8_index] = rb_out_s8

                buffer_manager.set_remapped_buffers(rb_list)
                buffer_manager.set_tensor_list(tensor_list)

                sch = auto_schedule(out)

            config = {
                "print_ir": False,
                "need_build": True,
                "name": kernel_name,
                "tensor_list": tensor_list}
            build(sch, config)


    def run_testcase(config_dict):
        # dataflow_v200, l1fusion_dict, shape_in, shape_w, pads, strides, bias_flag, relu_flag
        for i in config_dict:
            print("="*150)
            print("case {}".format(i))
            if i[0] in (1, 2):
                # conv + dequant + relu + quant singleout/doubleout
                dataflow_v200, shape_in, shape_w, pads, strides, offset_d, bias_flag, relu_flag, quant_scale, quant_offset, l1fusion_dict = i
                reqs16_relu_flag = 0
            elif i[0] in (5, 6):
                # conv + dequants16 + requants16
                dataflow_v200, shape_in, shape_w, pads, strides, offset_d, bias_flag, relu_flag, reqs16_relu_flag, l1fusion_dict = i
            else:
                dataflow_v200, shape_in, shape_w, pads, strides, offset_d, bias_flag, relu_flag, l1fusion_dict = i
                reqs16_relu_flag = 0

            if i[0] in (1, 2):
                conv_v200_fusion_case(dataflow_v200, l1fusion_dict,
                                      shape_in, shape_w, pads, strides, offset_d,
                                      bias_flag, relu_flag, True, reqs16_relu_flag,
                                      quant_scale=quant_scale, quant_offset=quant_offset)
            else:
                conv_v200_fusion_case(dataflow_v200, l1fusion_dict,
                                      shape_in, shape_w, pads, strides, offset_d,
                                      bias_flag, relu_flag, True, reqs16_relu_flag)


    set_current_compile_soc_info("Hi3796CV300CS")
    run_testcase(v200_l1fusion_testcase["3796CS"]["st"])

print("adding Conv2D v200 l1fusion ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_v200_lxfusion)