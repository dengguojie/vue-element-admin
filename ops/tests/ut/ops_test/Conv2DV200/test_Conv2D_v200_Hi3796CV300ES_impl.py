#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def TestConv2dV200(test_arg):
    """
    v200 ut case.
    """
    import te.lang.cce
    from te import tvm
    from topi import generic
    from topi.cce import util
    from te import platform as cce_conf
    from te.lang.cce import ConvParam
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.prelu import prelu_compute
    from impl.eltwise import eltwise_compute
    from te.lang.cce import AutoScheduleOp

    testcases = {
        "op_name": "conv_v200",
        "all": {
            # case name: ((fm_shape), (weight_shape), (paddings), (strides),
            # data_flow, bias_flag, relu_flag, vector_flag)
            # data_flow
            #FP16 UB FUSION
            #   21:conv2d+relu
            #   22:Conv2d+Eltwise(Add)
            #   23:Conv2d+Eltwise(Add)+ReLU
            #   24:conv2d+LeakyRelu
            #   25:Conv2d+LeakyReLU+Eltwise(Add)
            #   quant fusion
            #   49:conv2d+Prelu
            #   50:Conv2d+Eltwise(Add)+Prelu
            #   51:Conv2d+Prelu+Eltwise(Add)

            "conv_v200_bias_1_flow_21": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 21, 1, 0, 0),
            "conv_v200_bias_1_flow_22": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 22, 1, 0, 0),
            "conv_v200_bias_1_flow_23": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 23, 1, 0, 0),
            "conv_v200_bias_1_flow_24": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 24, 1, 0, 0),
            "conv_v200_bias_1_flow_25": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 25, 1, 0, 0),

            # prelu
            "conv_v200_bias_1_flow_49": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 49, 1, 0, 0),
            "conv_v200_bias_1_flow_50": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 50, 1, 0, 0),
            "conv_v200_bias_1_flow_51": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 51, 1, 0, 0),

            "conv_v200_bias_0_flow_21": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 21, 0, 0, 0),
            "conv_v200_bias_0_flow_22": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 22, 0, 0, 0),
            "conv_v200_bias_0_flow_23": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 23, 0, 0, 0),
            "conv_v200_bias_0_flow_24": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 24, 0, 0, 0),
            "conv_v200_bias_0_flow_25": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 25, 0, 0, 0),

            # prelu
            "conv_v200_bias_0_flow_49": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 49, 0, 0, 0),
            "conv_v200_bias_0_flow_50": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 50, 0, 0, 0),
            "conv_v200_bias_0_flow_51": ((2, 32, 7, 7), (32, 32, 2, 2), \
                [0, 0, 0, 0], [1, 1, 1, 1], 51, 0, 0, 0),
        }
    }


    def conv_v200_fusion_case(shape_in, shape_w, pads, strides, c_out, \
        orig_shape_w, data_flow, bias_flag, relu_flag, vector_flag):
        data_flow_fution_type_map_bias_true = {
            "unknown": 100,
            "21": 4,
            "22": 23,
            "23": 8,
            "24": 18,
            "25": 25,
            # 10 0000 1000 0010 0000 0000 0000 pattern_value:2,oplist_num:2-4,2-3
            "49": 34086912,
            # 10 0000 1100 0010 0000 0000 0000 pattern_value:2,oplist_num:3-4,2-3
            "50": 34349056,
            # 10 0000 1100 0010 0000 0000 0000 pattern_value:2,oplist_num:3-4,2-3
            "51": 34349056
            }

        data_flow_fution_type_map_bias_false = {
            "unknown": 100,
            "21": 3,
            "22": 22,
            "23": 7,
            "24": 5,
            "25": 24,
            # 1 0000 1000 0010 0000 0000 0000 pattern_value:1,oplist_num:2-4,2-3
            "49": 17309696,
            # 1 0000 1100 0010 0000 0000 0000 pattern_value:1,oplist_num:3-4,2-3
            "50": 17571840,
            # 1 0000 1100 0010 0000 0000 0000 pattern_value:1,oplist_num:3-4,2-3
            "51": 17571840
            }
        with tvm.target.cce():
            # conv2d
            dilations = [1, 1, 1, 1]

            fm = tvm.placeholder(shape_in, name='fm', dtype='float16', \
                attrs={'ori_format': 'NCHW'})
            filter_w = tvm.placeholder(shape_w, name='filter_w', \
                dtype='float16', \
                attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
            if bias_flag:
                bias_tensor = tvm.placeholder((c_out*16,), name='bias', \
                    dtype='float16')
            else:
                bias_tensor = None
            conv_res = conv2d_compute(fm, filter_w, bias_tensor, \
                None, None, strides, pads, dilations)
            if data_flow in (21, 22, 23, 24, 25):
                if data_flow == 21:
                    out = leaky_relu_compute(conv_res, None)
                    auto_sch_res = AutoScheduleOp(out)
                    sch = generic.auto_schedule(out)
                    tensor_list = [fm, filter_w, out]
                elif data_flow in (22, 23):
                    fm2 = tvm.placeholder(conv_res.shape, name='fmap2', \
                        dtype="float16", attrs={'ori_format': 'NCHW'})
                    out = eltwise_compute([conv_res, fm2], None)
                    if data_flow == 23:
                        out = leaky_relu_compute(out, None)
                    auto_sch_res = AutoScheduleOp(out)
                    sch = generic.auto_schedule(out)
                    tensor_list = [fm, filter_w, fm2, out]
                elif data_flow in (24, 25):
                    if data_flow == 25:
                        fm2 = tvm.placeholder(conv_res.shape, name='fmap2', \
                            dtype="float16", attrs={'ori_format': 'NCHW'})
                        conv_res = eltwise_compute([conv_res, fm2], None)
                    out = leaky_relu_compute(conv_res, None, negative_slope=0.1)
                    auto_sch_res = AutoScheduleOp(out)
                    sch = generic.auto_schedule(out)
                    tensor_list = [fm, filter_w, out]
                    if data_flow == 25:
                        tensor_list = [fm, filter_w, fm2, out]
                else:
                    pass
            if data_flow in (49, 50, 51):
                tensor_list = [fm, filter_w]
                prelu_weight = tvm.placeholder((1, c_out, 1, 16), \
                    name='prelu_weight', dtype=conv_res.dtype, \
                    attrs={'ori_shape': [c_out*16]})
                tensor_list.append(prelu_weight)
                fm2 = tvm.placeholder(conv_res.shape, name='fmap2', \
                        dtype="float16", attrs={'ori_format': 'NCHW'})
                if data_flow in (49, 51):
                    out = prelu_compute(conv_res, prelu_weight, None, \
                        kernel_name="prelu")
                    if data_flow == 51:
                        out = eltwise_compute([fm2, out], None)
                        tensor_list.append(fm2)
                else:
                    out = eltwise_compute([fm2, conv_res], None)
                    out = prelu_compute(out, prelu_weight, None, \
                        kernel_name="prelu")
                    tensor_list.append(fm2)
                tensor_list.append(out)
                auto_sch_res = AutoScheduleOp(out)
                sch = generic.auto_schedule(out)
            if bias_flag:
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
            assert auto_sch_res.fusion_type == fution_type
        return sch, tensor_list


    def conv_v200_fusion(fm_shape, filter, pads, strides, \
        data_flow, bias_flag, relu_flag, vector_flag, kernel_name):
        from te.platform.cce_policy import disableL2
        disableL2()
        block_size_k = 32
        block_size_n = 16
        batch, channel, height, weight = fm_shape
        c0 = 32
        if data_flow in (21, 22, 23, 24, 25, 49, 50, 51):
            block_size_k = 16
            c0 = 16
        c1 = (channel + c0 - 1) // c0
        shape_in = (batch, c1, height, weight, c0)

        out_channel = filter[0]
        in_channel_weight = ((filter[1] + block_size_k - 1) \
            // block_size_k) * block_size_k
        filter_h = filter[2]
        filter_w = filter[3]

        if data_flow in (21, 22, 23, 24, 25, 49, 50, 51):
            c_out = (out_channel + block_size_n - 1) // block_size_n
        else:
            c_out = (out_channel + block_size_k - 1) \
            // block_size_k * block_size_k
            c_out = (c_out + block_size_n - 1) // block_size_n
        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) \
            // block_size_k,
                c_out, block_size_n, block_size_k)

        sch, tensor_list = conv_v200_fusion_case(\
        shape_in, shape_w, pads, strides, c_out, filter, \
        data_flow, bias_flag, relu_flag, vector_flag)

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
            ddk_info = "Hi3796CV300ES"

        cce_conf.cce_conf.te_set_version(ddk_info)

    """
    The UT for cce Test_conv2d_v200
    """

    print("---------------------------------------------------")
    set_ddk_version("v200")
    print("[ UNITTEST START conv2d v200 ddk is Hi3796CV300ES]")

    run_testcase()
    set_ddk_version("v100")

print("adding Conv2D v200 Hi3796CV300ES ut testcases")
ut_case.add_cust_test_func(test_func=TestConv2dV200)