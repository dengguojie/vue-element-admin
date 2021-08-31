#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")


def test_conv2d_l1_fp16(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from tbe.dsl import auto_schedule
    from te import platform as cce_conf
    from impl.conv2d import conv2d_compute
    from impl.conv2d import _conv_layer_cce
    from impl.leaky_relu import leaky_relu_compute
    from impl.eltwise import eltwise_compute
    from impl.read_select import read_select_compute
    from impl.write_select import write_select_compute
    from te.platform.cce_policy import disableL2
    from te.platform.cce_policy import set_L1_info

    sys.path.append("./llt/ops/ut/testcase_python/")

    set_L1_info("L1_fusion_enabled", True)

    testcases = {
        "op_name": "conv_DDB",
        "all": {

            # case name: fm_shape, weight_shape, padding, stride, bias_flag,
            #            data_flow,
            #            l1_fusion_type, l1_space, fm_addr_type, out_addr_type,
            #            fm_valid_shape, fm_offset,
            #            eltwise_shape, eltwise_valid_shape, eltwise_addr_type, eltwise_offset, eltwise_rs_stride, ws_valid_shape
            #

            # l1_fusion_type
            #  -1: no l1 fusion
            #   0: deep fusion
            #   1: breadth fusion
            #
            # l1_space: for tiling, space of weight(Bytes)
            #
            # *_addr_type
            #   0: from/to DDR
            #   1: from/to L1
            #
            # data_flow : this py file only suport -2 and 4
            #   0: l1 fusion error case check
            #  -2: f16f16->f16 no_UB_fusion
            #   4: fp16_l1_fusion no_quant

            # DDR in L1 Fusion case
            "conv_l1_fusion_dido_rs_delt": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 0, 0,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [], [], 0, [], 0, []),
            "conv_l1_fusion_dil1o_rs_l1elt": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 0, 1,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [], [], 1, [], 0, []),
            "conv_l1_fusion_dido_rs_delt_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 0, 0,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 0, [0, 0, 1, 0, 0], 1,
                []),
            "conv_l1_fusion_dido_rs_l1elt_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 0, 1,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 1, [0, 0, 1, 0, 0], 1,
                []),
            "conv_l1_fusion_dido_rs_delt_rs_ws": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 0, 0,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 0, [0, 0, 1, 0, 0], 1,
                [1, 4, 56, 56, 16]),
            "conv_l1_fusion_dido_rs_l1elt_rs_ws": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 0, 1,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 1, [0, 0, 1, 0, 0], 1,
                [1, 4, 56, 56, 16]),

            "conv_l1_fusion_l1ido_rs_delt": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 1, 0,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [], [], 0, [], 0, []),
            "conv_l1_fusion_l1il1o_rs_l1elt": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 1, 1,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [], [], 1, [], 0, []),
            "conv_l1_fusion_l1ido_rs_delt_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 1, 0,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 0, [0, 0, 1, 0, 0], 1,
                []),
            "conv_l1_fusion_l1ido_rs_l1elt_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 1, 1,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 1, [0, 0, 1, 0, 0], 1,
                []),
            "conv_l1_fusion_l1ido_rs_delt_rs_ws": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 1, 0,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 0, [0, 0, 1, 0, 0], 1,
                [1, 4, 56, 56, 16]),
            "conv_l1_fusion_l1ido_rs_l1elt_rs_ws": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                4,
                0, 0, 1, 1,
                [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [1, 4, 9, 56, 16], [1, 4, 8, 56, 16], 1, [0, 0, 1, 0, 0], 1,
                [1, 4, 56, 56, 16]),

            # DDR in L1 Fusion case
            "conv_l1_fusion_dido": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                1, 0, 0, 0, [], [],
                [], [], None, [], 0, []),
            "conv_l1_fusion_dil1o": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                1, 0, 0, 1, [], [],
                [], [], None, [], 0, []),
            "conv_l1_fusion_dido_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                1, 0, 0, 0, [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [], [], None, [], 0, []),
            "conv_l1_fusion_dil1o_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                1, 0, 0, 1, [1, 4, 8, 56, 16], [0, 0, 9, 0, 0],
                [], [], None, [], 0, []),

            # # L1 in L1 Fusion case
            "conv_l1_fusion_l1ido": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                0, 0, 1, 0, [], [],
                [], [], None, [], 0, []),

            "conv_l1_fusion_l1il1o": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                0, 0, 1, 1, [], [],
                [], [], None, [], 0, []),

            "conv_l1_fusion_l1ido_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                0, 0, 1, 0, [1, 2, 8, 56, 32], [0, 0, 9, 0, 0],
                [], [], None, [], 0, []),
            "conv_l1_fusion_l1il1o_rs": (
                (1, 64, 56, 56), (64, 64, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], 1,
                -2,
                0, 0, 1, 1, [1, 2, 8, 56, 32], [0, 0, 9, 0, 0],
                [], [], None, [], 0, [])
        }
    }

    def conv_l1_fusion_case(shape_in, shape_w, pads, strides, bias_flag,
                            orig_shape_w,
                            l1_fusion_type, fm_addr_type, out_addr_type,
                            fm_valid_shape, fm_offset,
                            eltwise_shape, eltwise_valid_shape,
                            eltwise_addr_type, eltwise_offset,
                            eltwise_rs_stride, ws_valid_shape):
        with tvm.target.cce():
            dilations = [1, 1, 1, 1]
            c_out = shape_w[1] * 16
            L1_valid_size = 1
            if fm_valid_shape:
                valid_shape = fm_valid_shape
            else:
                valid_shape = shape_in
            for i in valid_shape:
                L1_valid_size = L1_valid_size * i
            fm = tvm.placeholder(shape_in, name='fm', dtype='float16',
                                 attrs={'ori_format': 'NCHW',
                                        'addr_type': fm_addr_type,
                                        'valid_shape': fm_valid_shape,
                                        'slice_offset': fm_offset,
                                        'L1_fusion_type': l1_fusion_type,
                                        "fmap_l1_valid_size": L1_valid_size,
                                        "L1_addr_flag": 1})
            filter_w = tvm.placeholder(shape_w, name='filter_w',
                                       dtype='float16',
                                       attrs={'ori_shape': orig_shape_w,
                                              'ori_format': 'NCHW'})

            bias = None
            if bias_flag:
                bias = tvm.placeholder((c_out,), name='bias', dtype='float16')

            outputs = {"addr_type": out_addr_type}

            conv_res = conv2d_compute(fm, filter_w, bias, None, outputs,
                                      strides, pads, dilations, offset_x=0)
            print(eltwise_shape)
            if eltwise_shape:
                fm2 = tvm.placeholder(eltwise_shape, name='fmap2',
                                      dtype="float16",
                                      attrs={'ori_format': 'NCHW',
                                             'addr_type': eltwise_addr_type,
                                             'valid_shape': eltwise_valid_shape,
                                             'slice_offset': eltwise_offset})
                data_other = read_select_compute(fm2, {},
                                                 [0, 0, eltwise_rs_stride,
                                                  eltwise_rs_stride, 0])
            else:
                fm2 = tvm.placeholder(conv_res.shape, name='fmap2',
                                      dtype="float16",
                                      attrs={'ori_format': 'NCHW',
                                             'addr_type': eltwise_addr_type})
                data_other = fm2
            tensor_list = [fm, filter_w]
            if bias_flag:
                tensor_list.append(bias)
            tensor_list.append(fm2)
            print(data_other, conv_res)
            res_add = eltwise_compute([data_other, conv_res], None)
            res_relu = leaky_relu_compute(res_add, None)
            if ws_valid_shape:
                res = write_select_compute(res_relu,
                                           {"valid_shape": ws_valid_shape})
            else:
                res = res_relu

            res.op.attrs["addr_type"] = out_addr_type
            tensor_list.append(res)
            sch = auto_schedule(res)

        return sch, tensor_list

    def conv_layer(fm_shape, filter, pads, strides, bias_flag, data_flow,
                   l1_fusion_type, l1_space, fm_addr_type, out_addr_type,
                   fm_valid_shape, fm_offset,
                   eltwise_shape, eltwise_valid_shape, eltwise_addr_type,
                   eltwise_offset, eltwise_rs_stride, ws_valid_shape,
                   kernel_name_val="convolutin_cce"):

        disableL2()
        set_L1_info("op_L1_space", l1_space)
        if data_flow == -2:
            fm_type = "float16"
            weight_type = "float16"
            output_type = "float16"
        else:
            fm_type = "int8"
            weight_type = "int8"
            output_type = "int32"

        padh = pads[0]
        padw = pads[2]
        strideh = strides[2]
        stridew = strides[3]
        bias_tensor = False
        if bias_flag == 1:
            bias_tensor = True
        input_memory_type = True if fm_addr_type else False
        output_memory_type = True if out_addr_type else False
        L1_valid_size = 1
        if fm_valid_shape:
            valid_shape = fm_valid_shape
        else:
            valid_shape = fm_shape
        for i in valid_shape:
            L1_valid_size = L1_valid_size * i
        fusion_para = {"input_memory_type": input_memory_type,
                       "output_memory_type": output_memory_type,
                       "valid_shape": fm_valid_shape,
                       "slice_offset": fm_offset,
                       "l1_fusion_type": l1_fusion_type,
                       "fmap_l1_valid_size": L1_valid_size,
                       "fmap_l1_addr_flag": 1}
        _conv_layer_cce(fm_shape, filter, fm_type, weight_type, output_type,
                        padh, padw, strideh, stridew, bias=bias_tensor,
                        fusion_para=fusion_para,
                        kernel_name=kernel_name_val, need_build=True,
                        need_print=False)

    def conv_l1_fusion(fm_shape, filter, pads, strides, bias_flag, data_flow,
                       l1_fusion_type, l1_space, fm_addr_type, out_addr_type,
                       fm_valid_shape, fm_offset,
                       eltwise_shape, eltwise_valid_shape, eltwise_addr_type,
                       eltwise_offset, eltwise_rs_stride, ws_valid_shape,
                       kernel_name_val="convolution_fusion_cce"):
        disableL2()
        set_L1_info("op_L1_space", l1_space)

        block_size_k = 16
        C0 = 16

        block_size_n = 16
        batch, channel, height, weight = fm_shape
        C1 = (channel + C0 - 1) // C0
        shape_in = (batch, C1, height, weight, C0)

        out_channel = filter[0]
        in_channel_weight = ((filter[
                                  1] + block_size_k - 1) // block_size_k) * block_size_k
        filter_h = filter[2]
        filter_w = filter[3]

        c_out = (out_channel + block_size_k - 1) // block_size_k * block_size_k
        c_out = (c_out + block_size_n - 1) // block_size_n
        shape_w = ((
                           in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                   c_out, block_size_n, block_size_k)

        sch, tensor_list = conv_l1_fusion_case(shape_in, shape_w, pads,
                                               strides, bias_flag, filter,
                                               l1_fusion_type, fm_addr_type,
                                               out_addr_type, fm_valid_shape,
                                               fm_offset,
                                               eltwise_shape,
                                               eltwise_valid_shape,
                                               eltwise_addr_type,
                                               eltwise_offset,
                                               eltwise_rs_stride,
                                               ws_valid_shape)

        config = {"print_ir": False,
                  "need_build": True,
                  "name": kernel_name_val,
                  "tensor_list": tensor_list,
                  "l1_fusion_option": 1}

        te.lang.cce.cce_build_code(sch, config)

    def conv_l1_fusion_check_case(fm_shape, filter, pads, strides, bias_flag,
                                  data_flow,
                                  l1_fusion_type, l1_space, fm_addr_type,
                                  out_addr_type, fm_valid_shape, fm_offset,
                                  eltwise_shape, eltwise_valid_shape,
                                  eltwise_addr_type, eltwise_offset,
                                  eltwise_rs_stride, ws_valid_shape,
                                  kernel_name_val="convolutin_cce"):

        disableL2()
        fm_type = "float16"
        weight_type = "float16"
        output_type = "float16"

        padh = pads[0]
        padw = pads[2]
        strideh = strides[2]
        stridew = strides[3]
        bias_tensor = False
        if bias_flag == 1:
            bias_tensor = True
        input_memory_type = True if fm_addr_type else False
        output_memory_type = True if out_addr_type else False
        L1_valid_size = 1
        if fm_valid_shape:
            valid_shape = fm_valid_shape
        else:
            valid_shape = fm_shape
        for i in valid_shape:
            L1_valid_size = L1_valid_size * i
        fusion_para = {"input_memory_type": input_memory_type,
                       "output_memory_type": output_memory_type,
                       "valid_shape": fm_valid_shape,
                       "slice_offset": fm_offset,
                       "l1_fusion_type": l1_fusion_type,
                       "fmap_l1_valid_size": L1_valid_size,
                       "fmap_l1_addr_flag": 1}
        if cce_conf.get_soc_spec("SOC_VERSION") == "Hi3796CV300ES":
            if data_flow == 0:
                try:
                    _conv_layer_cce(fm_shape, filter, fm_type, weight_type,
                                    output_type, padh, padw, strideh, stridew,
                                    bias=bias_tensor, fusion_para=fusion_para,
                                    kernel_name=kernel_name_val)
                except Exception as e:
                    print(e)
            else:
                _conv_layer_cce(fm_shape, filter, fm_type, weight_type,
                                output_type, padh, padw, strideh, stridew,
                                bias=bias_tensor, fusion_para=fusion_para,
                                kernel_name=kernel_name_val)
        else:
            if data_flow == 0:
                try:
                    _conv_layer_cce(fm_shape, filter, fm_type, weight_type,
                                    output_type,
                                    padh, padw, strideh, stridew,
                                    bias=bias_tensor, fusion_para=fusion_para,
                                    kernel_name=kernel_name_val,
                                    need_build=True, need_print=False)
                except Exception as e:
                    print(e)
            else:
                _conv_layer_cce(fm_shape, filter, fm_type, weight_type,
                                output_type,
                                padh, padw, strideh, stridew, bias=bias_tensor,
                                fusion_para=fusion_para,
                                kernel_name=kernel_name_val, need_build=True,
                                need_print=False)

    def run_testcase():
        testcases_for_all = testcases["all"]
        for key in testcases_for_all:
            print("run the case:", key)
            if testcases_for_all[key][5] == -2:
                conv_layer(*testcases_for_all[key], key)
            if testcases_for_all[key][5] == 4:
                conv_l1_fusion(*testcases_for_all[key], key)
            if testcases_for_all[key][5] == 0 or testcases_for_all[key][
                5] == 1:
                conv_l1_fusion_check_case(*testcases_for_all[key], key)
            print("[success: %s]" % key)

    def set_ddk_version(version):
        if version == "v100":
            ddk_info = "Ascend310"
        else:
            ddk_info = "Hi3796CV300ES"
        cce_conf.cce_conf.te_set_version(ddk_info)

    def test_conv_v100():
        set_ddk_version("v100")
        print("[ UNITTEST START conv2d v100]")
        run_testcase()

    # test_conv_v100()


# ut_case.add_cust_test_func(test_func=test_conv2d_l1_fp16)

if __name__ == '__main__':
    ut_case.add_cust_test_func(test_func=test_conv2d_l1_fp16)
    exit(0)
