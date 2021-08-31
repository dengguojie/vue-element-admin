#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_vector(test_arg):
    import te
    from te import tvm
    import te.lang.cce
    from te.lang.cce import cce_build_code, ConvParam
    from tbe.dsl import auto_schedule
    from impl.strided_write import strided_write_compute
    from impl.write_select import write_select_compute
    from te.platform.cce_policy import set_L1_info


    def conv_vector_fused_fp16_0():
        with tvm.target.cce():
            shape_in = (1, 2, 16, 8, 16)
            shape_w = (2, 2, 16, 16)
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")

            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            sch = auto_schedule(conv_res)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, conv_res]
        }
        te.lang.cce.cce_build_code(sch, config)


    def conv_vector_fused_fp16_1():
        shape_in = (1, 2, 16, 8, 16)
        shape_w = (2, 2, 16, 16)
        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            res0 = te.lang.cce.vlog(conv_res)
            sch = auto_schedule(res0)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            res0 = te.lang.cce.vexp(conv_res)
            sch = auto_schedule(res0)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            res0 = te.lang.cce.vrelu(conv_res)
            sch = auto_schedule(res0)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            res0 = te.lang.cce.vmuls(conv_res, 3)
            sch = auto_schedule(res0)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            res0 = te.lang.cce.vadds(conv_res, 7)
            sch = auto_schedule(res0)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            res0 = te.lang.cce.vabs(conv_res)
            sch = auto_schedule(res0)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, res0]}
        te.lang.cce.cce_build_code(sch, config)


    def conv_vector_fused_fp16_2_1():
        shape_in = (1, 2, 16, 8, 16)
        shape_w = (2, 2, 16, 16)
        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            conv_shape = conv_res.shape
            eee = tvm.placeholder(conv_shape, name='eee', dtype="float16")
            res0 = te.lang.cce.vadd(conv_res, eee)
            sch = auto_schedule(res0)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, eee, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            conv_shape = conv_res.shape
            eee = tvm.placeholder(conv_shape, name='eee', dtype="float16")
            res0 = te.lang.cce.vsub(conv_res, eee)
            sch = auto_schedule(res0)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, eee, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            conv_shape = conv_res.shape
            eee = tvm.placeholder(conv_shape, name='eee', dtype="float16")
            res0 = te.lang.cce.vmul(conv_res, eee)
            sch = auto_schedule(res0)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, eee, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            conv_shape = conv_res.shape
            eee = tvm.placeholder(conv_shape, name='eee', dtype="float16")
            res0 = te.lang.cce.vmax(conv_res, eee)
            sch = auto_schedule(res0)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, eee, res0]}
        te.lang.cce.cce_build_code(sch, config)

        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            conv_shape = conv_res.shape
            eee = tvm.placeholder(conv_shape, name='eee', dtype="float16")
            res0 = te.lang.cce.vmin(conv_res, eee)
            sch = auto_schedule(res0)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, eee, res0]}
        te.lang.cce.cce_build_code(sch, config)


    def conv_vector_fused_fp16_6():
        shape_in = (1, 2, 16, 8, 16)
        shape_w = (2, 2, 16, 16)
        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            res0 = te.lang.cce.vlog(conv_res)
            res1 = te.lang.cce.vexp(res0)
            res2 = te.lang.cce.vrelu(res1)
            res3 = te.lang.cce.vmuls(res2, 3)
            res4 = te.lang.cce.vadds(res3, 33)
            res5 = te.lang.cce.vabs(res4)
            sch = auto_schedule(res5)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, res5]}
        te.lang.cce.cce_build_code(sch, config)


    def conv_vector_fused_fp16_2_7():
        shape_in = (1, 2, 16, 8, 16)
        shape_w = (2, 2, 16, 16)
        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='FmapW', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='FilterW', dtype="float16")
            bias_tensor = tvm.placeholder(
                (shape_w[1] * shape_w[2], ), name='Bias', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            conv_shape = conv_res.shape
            eee = tvm.placeholder(conv_shape, name='eee', dtype="float16")
            ddd = tvm.placeholder(conv_shape, name='ddd', dtype="float16")
            res0 = te.lang.cce.vadd(conv_res, eee)
            res1 = te.lang.cce.vsub(res0, ddd)
            res2 = te.lang.cce.vmul(res0, res1)
            res3 = te.lang.cce.vmin(res2, ddd)
            res4 = te.lang.cce.vmax(res3, conv_res)
            sch = auto_schedule(res4)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, bias_tensor, eee, ddd, res4]}
        te.lang.cce.cce_build_code(sch, config)


    def conv_vector_fused_fp16_buffer_reuse():
        shape_in = (1, 2, 16, 8, 16)
        shape_w = (2, 2, 16, 16)
        with tvm.target.cce():
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='FmapW', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='FilterW', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            conv_shape = conv_res.shape
            eee = tvm.placeholder(conv_shape, name='eee', dtype="float16")
            ddd = tvm.placeholder(conv_shape, name='ddd', dtype="float16")
            res0 = te.lang.cce.vadd(conv_res, eee)
            res1 = te.lang.cce.vsub(res0, ddd)
            res2 = te.lang.cce.vmul(res0, res1)
            res3 = te.lang.cce.vmin(res2, res1)
            res4 = te.lang.cce.vmax(res3, conv_res)
            res5 = te.lang.cce.vadd(res4, res3)
            res6 = te.lang.cce.vsub(res5, res2)
            sch = auto_schedule(res6)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, eee, ddd, res6]}
        te.lang.cce.cce_build_code(sch, config)


    def conv_vector_fused_fp16_tiling():
        with tvm.target.cce():
            shape_in = (4, 2, 16, 8, 16)
            shape_w = (2, 1, 16, 16)
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='FmapW', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='FilterW', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})
            m = 8
            k = 2
            n = 2
            tt = {'AL0_matrix': [m, k, 16, 16], 'BL0_matrix': [k, n, 16, 16], 'CL0_matrix': [n, m, 16, 16], 'AUB_shape': None,
                  'CUB_matrix': [n, m, 16, 16], 'BL1_shape': None, 'AL1_shape': [], 'cout_bef_batch_flag': 1, "block_dim": [1,1,1],
                  'A_overhead_opt_flag':0, 'B_overhead_opt_flag':0}
            ConvParam.tiling = tt
            sch = auto_schedule(conv_res)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, conv_res]}
        te.lang.cce.cce_build_code(sch, config)

    def conv_vector_fused_fp16_default_tiling():
        with tvm.target.cce():
            shape_in = (4, 2, 16, 8, 16)
            shape_w = (2, 1, 16, 16)
            pad_h = 0
            pad_w = 0
            stride_h = 1
            stride_w = 1
            filter_h = 1
            filter_w = 1
            Data = tvm.placeholder(shape_in, name='FmapW', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='FilterW', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})

            ConvParam.tiling_query_param["default_tiling"] = True

            sch = auto_schedule(conv_res)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, conv_res]}
        te.lang.cce.cce_build_code(sch, config)

    def conv2d_inner_batch():
        with tvm.target.cce():
            shape_in = (304, 32, 7, 7, 16)
            shape_w = (288, 32, 16, 16)
            pad_h = 1
            pad_w = 1
            stride_h = 1
            stride_w = 1
            filter_h = 3
            filter_w = 3
            Data = tvm.placeholder(shape_in, name='FmapW', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='FilterW', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"kernel_name": "test_ut",
                               "pad_h": pad_h, "pad_w": pad_w,
                               "stride_h": stride_h, "stride_w": stride_w,
                               "filter_h": filter_h, "filter_w": filter_w,
                               'offset_x': 0, "dilate_w": 1, "dilate_h": 1})

            ConvParam.tiling = {
                'AL0_matrix': [4, 12, 16, 16], 'CL0_matrix': [2, 4, 16, 16, 2],
                'CUB_matrix': [2, 4, 16, 16], 'A_overhead_opt_flag': True, 'B_overhead_opt_flag': False,
                'BL0_matrix': [12, 2, 16, 16],
                'manual_pingpong_buffer': {'AUB_pbuffer': 1, 'BUB_pbuffer': 1, 'AL1_pbuffer': 1, 'BL1_pbuffer': 1, 'AL0_pbuffer': 1, 'BL0_pbuffer': 1, 'CL0_pbuffer': 1, 'CUB_pbuffer': 1, 'UBG_pbuffer': 1}, 'n_bef_batch_flag': False, 
                'AL1_shape': [32, 1, 1, 1], 'BL1_shape': None, 'block_dim': [4, 2, 1, 1], 'CUB_channel_wise_flag': False}

            sch = auto_schedule(conv_res)

        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, conv_res]}
        te.lang.cce.cce_build_code(sch, config)



    def test_conv_vector_fused_fp16_0():
        print("[ conv_vector_fused_fp16_0 ]")
        conv_vector_fused_fp16_0()

    def test_conv_vector_fused_fp16_1():
        print("[ conv_vector_fused_fp16_1 ]")
        conv_vector_fused_fp16_1()

    def test_conv_vector_fused_fp16_2_1():
        print("[ conv_vector_fused_fp16_2_1 ]")
        conv_vector_fused_fp16_2_1()

    def test_conv_vector_fused_fp16_6():
        print("[ conv_vector_fused_fp16_6 ]")
        conv_vector_fused_fp16_6()

    def test_conv_vector_fused_fp16_2_7():
        print("[ conv_vector_fused_fp16_2_7 ]")
        conv_vector_fused_fp16_2_7()

    def test_conv_vector_fused_fp16_buffer_reuse():
        print("[ conv_vector_fused_fp16_buffer_reuse ]")
        conv_vector_fused_fp16_buffer_reuse()

    def test_conv_vector_fused_fp16_tiling():
        print("[ conv_vector_fused_fp16_tiling ]")
        conv_vector_fused_fp16_tiling()

    def test_conv_vector_fused_fp16_default_tiling():
        print("[ conv_vector_fused_fp16_default_tiling ]")
        conv_vector_fused_fp16_default_tiling()

    def test_conv2d_inner_batch():
        print("[ test_conv2d_inner_batch ]")
        conv2d_inner_batch()

    def test_conv2d_write_select__stride_write():
        print("[ test_conv2d_write_select__stride_write ]")
        set_L1_info("L1_fusion_enabled", True)
        set_L1_info("L2_fusion_enabled", False)
        set_L1_info("op_L1_space", 512*1024)
        shape_in = (2, 8, 14, 14, 16)
        shape_w = (8, 32, 16, 16)
        write_valid_shape = (2, 32, 28, 14, 16)
        HWC0 = write_valid_shape[2] * write_valid_shape[3] * write_valid_shape[4]
        with tvm.target.cce():
            Data = tvm.placeholder(shape_in, name='Fmap', dtype="float16")
            Weight = tvm.placeholder(shape_w, name='Filter', dtype="float16")
            conv_res = te.lang.cce.conv(
                Data, Weight, {"pad_h": 0, "pad_w": 0, "stride_h": 1, "stride_w": 1,
                "filter_h": 1, "filter_w": 1,
                "fusion_para": {"input_memory_type": 0, "output_memory_type": 0,
                "l1_fusion_type": 0}, })
            res1 = te.lang.cce.vrelu(conv_res)
            output = {"valid_shape":write_valid_shape}
            res =write_select_compute(res1, output)
            stride = write_valid_shape[1]*write_valid_shape[2]*write_valid_shape[3]*write_valid_shape[4]
            conv_swrite_res = strided_write_compute(res, None, 1, stride, "strided_write")
            sch = auto_schedule(conv_swrite_res)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "aaaa",
            "tensor_list": [Data, Weight, conv_swrite_res]}
        te.lang.cce.cce_build_code(sch, config)
        set_L1_info("L1_fusion_enabled", False)

    test_conv_vector_fused_fp16_0()
    test_conv_vector_fused_fp16_1()
    test_conv_vector_fused_fp16_2_1()
    test_conv_vector_fused_fp16_6()
    test_conv_vector_fused_fp16_2_7()
    test_conv_vector_fused_fp16_buffer_reuse()
    test_conv_vector_fused_fp16_tiling()
    test_conv_vector_fused_fp16_default_tiling()
    test_conv2d_write_select__stride_write()
    test_conv2d_inner_batch()

# ut_case.add_cust_test_func(test_func=test_conv2d_vector)

if __name__ == '__main__':
    ut_case.add_cust_test_func(test_func=test_conv2d_vector)
    exit(0)
