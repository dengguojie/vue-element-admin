#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_compress(test_arg):
    import os
    import shutil
    import sys
    from impl.conv2d_compress import conv2dcompress_compute
    from te import tvm
    from tbe.dsl import auto_schedule
    import te.lang.cce
    from te import platform as cceconf
    import math
    import numpy as np

    sys.path.append("./llt/ops/st_all/cce_all/testcase_python")
    sys.path.append("./llt/ops/common/op_cfg/")
    sys.path.append("./llt/ops/ut/testcase_python/tvm_tbe_utest_phase_conv_compress/conv_compress/")
    import conv_compress_op_testcase as tc
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.eltwise import eltwise_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.conv2d_compress import _conv_layer_compress_cce

    from run_testcase import run_testcase, get_path_val, print_func_name
    bin_path_val = ""


    def _conv_v200_case(fm_type, weight_type, output_type,
                        fm_shape, w_shape, padding, strides,
                        data_flow_type, bias_flag, relu_flag, kernel_name_val):
            # data_flow
            #   10: s8s8->s32
            #   20: fp16fp16->fp16
            #   0: s32->s8
            #   1: s32->s16
            #   2: s32->s16->s8
            #   3: s32->s16->s8/s16 double output
            #   6: s32->fp16

        padh = padding[0]
        padw = padding[1]
        strideh = strides[0]
        stridew = strides[1]
        if fm_shape[1] <= 4 and not (w_shape[2] == 1 and w_shape[3] == 1) \
                and fm_type != "int8":
            optim_dict = {"c0_optim_flg": True}
            _conv_layer_compress_cce(fm_shape, w_shape, w_shape, fm_type,
                                    weight_type, weight_type, output_type,
                                    padh, padw, strideh, stridew,
                                    bias=True if bias_flag else False,
                                    optim_dict=optim_dict, kernel_name=kernel_name_val)
        else:
            _conv_layer_compress_cce(fm_shape, w_shape, w_shape, fm_type,
                                    weight_type, weight_type, output_type,
                                    padh, padw, strideh, stridew,
                                    bias=True if bias_flag else False,
                                    kernel_name=kernel_name_val)


    def _conv_v200_fusion_case(fm_type, weight_type, output_type, shape_in,
                               shape_w, pads, strides, c_out, orig_shape_w,
                               data_flow, bias_flag, relu_flag, vector_flag):
        with tvm.target.cce():
            # conv2d
            dilations = [1, 1, 1, 1]
            if vector_flag:
                shape_req = (1, c_out, 1, 1, 16)
            else:
                shape_req = (1, 1, 1, 1, 1)
            shape_c = (1, c_out, 1, 1, 16)
            fm = tvm.placeholder(shape_in, name='fm', dtype=fm_type, attrs={'ori_format': 'NCHW'})
            filter_w = tvm.placeholder(shape_w, name='Filter', dtype=weight_type,
                                       attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
            # u8/s8 -> s32
            conv_res = conv2d_compute(fm, filter_w, None, None, None, strides, pads, dilations)
            print("conv_res",conv_res)
            if data_flow == 0:
                #requant
                vreq_reg = tvm.placeholder(shape_req, name='vreq_reg', dtype='uint64')
                out = _ascend_requant_compute(conv_res, vreq_reg, None, relu_flag=relu_flag)
                sch = auto_schedule(out)
                tensor_list = [fm, filter_w, vreq_reg, out]
            elif data_flow == 1:
                # dequant_s16
                # s32->s16
                vdeq_reg = tvm.placeholder(shape_req, name='vdeq_reg', \
                    dtype='uint64', attrs={'ori_shape': [shape_w[1] * 16]})
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias_s16', dtype='int16')
                else:
                    bias_tensor = None
                out = _ascend_dequant_s16_compute(conv_res, vdeq_reg, bias_tensor, None, relu_flag=relu_flag)

                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq_reg, bias_tensor, out]
                else:
                    tensor_list = [fm, filter_w, vdeq_reg, out]
            elif data_flow == 2:
                # dequant_s16
                # s32->s16
                vdeq16_reg = tvm.placeholder(shape_req, name='vdeqs16_reg', dtype='uint64')
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias_s16', dtype='int16')
                else:
                    bias_tensor = None
                requant_s16 = _ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)
                # requant_s16
                # vadd_relu + requant_s16
                # relu_s16 = relu(fm2 + requant_s16)
                # relu_s16 -> relu_s8
                conv16_reg = tvm.placeholder(shape_c, name='conv16_reg', dtype='uint64')
                fm2 = tvm.placeholder(conv_res.shape, name='fm2', dtype='int16')
                out = _ascend_requant_s16_compute(requant_s16, conv16_reg, fm2, None, None, dual_output=False,
                                                  relu_flag=True)
                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out[0]]
                else:
                    tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out[0]]
            elif data_flow == 3:
                # dequant_s16
                # s32->s16
                vdeq16_reg = tvm.placeholder(shape_req, name='vdeqs16_reg', dtype='uint64')
                if bias_flag:
                    bias_tensor = tvm.placeholder(shape_c, name='bias_s16', dtype='int16')
                else:
                    bias_tensor = None
                requant_s16 = _ascend_dequant_s16_compute(conv_res, vdeq16_reg, bias_tensor, None, relu_flag=relu_flag)

                # requant_s16
                # vadd_relu + requant_s16
                # relu_s16 = relu(fm2 + requant_s16)
                # relu_s16 -> relu_s8
                conv16_reg = tvm.placeholder(shape_c, name='conv16_reg', dtype='uint64')
                fm2 = tvm.placeholder(conv_res.shape, name='fm2', dtype='int16')
                out = _ascend_requant_s16_compute(requant_s16, conv16_reg, fm2, None, None, dual_output=True,
                                                  relu_flag=True)
                sch = auto_schedule(out)
                if bias_flag:
                    tensor_list = [fm, filter_w, vdeq16_reg, bias_tensor, conv16_reg, fm2, out[0], out[1]]
                else:
                    tensor_list = [fm, filter_w, vdeq16_reg, conv16_reg, fm2, out[0], out[1]]
            elif data_flow == 4:
                deq16_reg = tvm.placeholder(shape_req, name='deq_reg', \
                    dtype='uint64', attrs={'ori_shape': [shape_w[1] * 16]})
                out = ascend_dequant_compute(conv_res, deq16_reg, None, \
                    sqrt_mode=False, relu_flag=relu_flag)
                sch = auto_schedule(out)
                tensor_list = [fm, filter_w, deq16_reg, out]
            else:
                pass

        return sch, tensor_list


    def _conv_v200_fusion(fm_type, weight_type, output_type, fm_shape, w_shape,
                      padding, strides, data_flow_type, bias_flag, relu_flag, vector_flag, kernel_name):

        from te.platform.cce_policy import disableL2
        disableL2()
        block_size_k = 32
        block_size_n = 16
        batch, channel, height, weight = fm_shape
        C0 = 32
        C1 = (channel + C0 - 1) // C0
        shape_in = (batch, C1, height, weight, C0)

        out_channel = w_shape[0]
        in_channel_weight = ((w_shape[1] + block_size_k - 1) // block_size_k) * block_size_k
        filter_h = w_shape[2]
        filter_w = w_shape[3]

        if data_flow_type == 1 or data_flow_type == 4:
            c_out = (out_channel + block_size_n - 1) // block_size_n
        else:
            c_out = (out_channel + block_size_k - 1) // block_size_k * block_size_k
            c_out = (c_out + block_size_n - 1) // block_size_n
        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                   c_out, block_size_n, block_size_k)

        padding_4d = [padding[0], padding[0], padding[1], padding[1]]
        strides_4d = [0, 0, strides[0], strides[1]]     # NCHW
        sch, tensor_list = _conv_v200_fusion_case(fm_type, weight_type, output_type,
                                                  shape_in, shape_w, padding_4d, strides_4d,
                                                  c_out, w_shape, data_flow_type,
                                                  bias_flag, relu_flag, vector_flag)

        config = {"print_ir": False,
                  "need_build": True,
                  "name": kernel_name,
                  "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def _test_conv200(fm_type, weight_type, output_type, fm_shape, w_shape,
                      padding, strides, data_flow_type, bias_flag, relu_flag, vector_flag, kernel_name_val):
        if data_flow_type == -1:
            _conv_v200_case(fm_type, weight_type, output_type, fm_shape, w_shape,
                      padding, strides, data_flow_type, bias_flag, relu_flag, kernel_name_val)
        else:
            _conv_v200_fusion(fm_type, weight_type, output_type, fm_shape, w_shape,
                      padding, strides, data_flow_type, bias_flag, relu_flag, vector_flag, kernel_name_val)

        kernel_meta_path = "./kernel_meta/"
        lib_kernel_name = "lib" + kernel_name_val + ".so"
        if (os.path.isfile(kernel_meta_path + lib_kernel_name)):
            shutil.move(kernel_meta_path + lib_kernel_name, bin_path_val + "/" + lib_kernel_name)
        else:
            shutil.move(kernel_meta_path + kernel_name_val + ".o", bin_path_val + "/" + kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json", bin_path_val + "/" + kernel_name_val + ".json")
        print('gen .o and .json of %s succ' % kernel_name_val)


    def _conv_lhisi_fusion_case(shape_in, shape_w, pads, strides, c_out, \
        orig_shape_w, data_flow, bias_flag, deq_sqrt, deq_relu, q_sqrt, \
        q_scaler, q_offset):
        q_scaler = np.float16(q_scaler)
        q_offset = np.float16(q_offset)
        with tvm.target.cce():
            dilations = [1, 1, 1, 1]
            shape_c = (1, c_out, 1, 1, 16)
            fm = tvm.placeholder(shape_in, name='fm', dtype='int8', attrs={'ori_format': 'NCHW'})
            filter_w = tvm.placeholder(shape_w, name='Filter', dtype='int8',
                                       attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
            if bias_flag:
                bias_tensor = tvm.placeholder((c_out * 16,), name='bias', dtype='int32')
            else:
                bias_tensor = None
            compress_index_shape = tvm.var("compress_index_shape", dtype="int32")
            compress_index = tvm.placeholder((compress_index_shape,),
                                             name='compress_index', dtype="int8")
            conv_res = conv2dcompress_compute(fm, filter_w, compress_index, \
                bias_tensor, None, None, strides, pads, dilations, 1, 'NCHW', \
                q_offset)
            deq_reg = tvm.placeholder(shape_c, name='deq_reg', dtype='float16', \
                attrs={'ori_shape': [shape_w[1] * 16]})
            out = ascend_dequant_compute(conv_res, deq_reg, None, deq_sqrt, \
                deq_relu)
            fm2 = tvm.placeholder(out.shape, name='fmap2', dtype="float16", \
                attrs={'ori_format': 'NCHW'})
            tensor_list = [fm, filter_w, compress_index, deq_reg]
            if data_flow == 1:
                out = ascend_quant_compute(out, None, q_scaler, q_offset, q_sqrt)
            if data_flow in (2, 3):
                res_add = eltwise_compute([fm2, out], None)
                res_relu = leaky_relu_compute(res_add, None)
                res_quant = ascend_quant_compute(res_relu, None, q_scaler, \
                    q_offset, q_sqrt)
                out = res_quant
                tensor_list.append(fm2)
                if data_flow == 3:
                    out = [res_relu, res_quant]

            if bias_flag:
                tensor_list.append(bias_tensor)
            import collections.abc
            if isinstance(out, collections.abc.Sequence):
                tensor_list.extend(out)
            else:
                tensor_list.append(out)
            sch = auto_schedule(out)
        return sch, tensor_list


    def _conv_lhisi_fusion(fm_type, weight_type, output_type, fm_shape, w_shape,
                      padding, strides, data_flow_type, bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset,kernel_name):
        from te.platform.cce_policy import disableL2
        disableL2()
        block_size_k = 32
        block_size_n = 16
        batch, channel, height, weight = fm_shape
        C0 = 32
        C1 = (channel + C0 - 1) // C0
        shape_in = (batch, C1, height, weight, C0)
        out_channel = w_shape[0]
        in_channel_weight = ((w_shape[1] + block_size_k - 1) // block_size_k) * block_size_k
        filter_h = w_shape[2]
        filter_w = w_shape[3]
        if output_type == 'int8':
            out_channel = math.ceil(out_channel / 32) * 32
        elif output_type == 'float16':
            out_channel = math.ceil(out_channel / 16) * 16
        else:
            raise AttributeError('not support output_type %s' % output_type)
        c_out = (out_channel + block_size_n - 1) // block_size_n
        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                   c_out, block_size_n, block_size_k)
        padding_4d = [padding[0], padding[0], padding[1], padding[1]] if len(padding) == 2 else padding
        strides_4d = [0, 0, strides[0], strides[1]]     # NCHW
        sch, tensor_list = _conv_lhisi_fusion_case(shape_in, shape_w, padding_4d, strides_4d, c_out, w_shape, data_flow_type,
                                                        bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset)
        config = {"print_ir": False,
                  "need_build": True,
                  "name": kernel_name,
                  "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)


    def _test_conv_lhisi(fm_type, weight_type, output_type, fm_shape, w_shape,
                         padding, strides, data_flow_type, bias_flag, deq_sqrt,
                         deq_relu, q_sqrt, q_scaler, q_offset, kernel_name_val):
        if data_flow_type == -1:
            if fm_shape[1] <= 4 and not (w_shape[2] == 1 and w_shape[3] == 1) \
                    and fm_type != "int8":
                print("okkkk000")
                optim_dict = {"c0_optim_flg": True}
                _conv_layer_compress_cce(fm_shape, w_shape, w_shape, fm_type, weight_type, weight_type,
                                    output_type, [padding[0], padding[1]], [padding[2], padding[3]],
                                    strides[0], strides[1], offset_x=q_offset,
                                    bias=bool(bias_flag), optim_dict=optim_dict,
                                    kernel_name=kernel_name_val,
                                    need_build=True, need_print=False)
            else:
                print("okkkk00000")
                _conv_layer_compress_cce(fm_shape, w_shape, w_shape, fm_type, weight_type, weight_type,
                                    output_type, [padding[0], padding[1]], [padding[2], padding[3]],
                                    strides[0], strides[1], offset_x=q_offset,
                                    bias=bool(bias_flag),
                                    kernel_name=kernel_name_val, need_build=True,
                                    need_print=False)
        else:
            _conv_lhisi_fusion(fm_type, weight_type, output_type, fm_shape,
                               w_shape, padding, strides, data_flow_type,
                               bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler,
                               q_offset, kernel_name_val)



    def test_cce_conv200():
        global bin_path_val
        cceconf.cce_conf.te_set_version("Hi3796CV300ES")
        testcases = tc.gen_dict_4_op("conv_compress", platform="Hi3796CV300ES")
        print("Hi3796CV300ES:", testcases)
        bin_path_val = "./llt/ops/common/kernel_bin/conv_compress/"
        print("bin_path_val:", bin_path_val)
        run_testcase(testcases, _test_conv_lhisi)
        print("end")
    test_cce_conv200()

# ut_case.add_cust_test_func(test_func=test_conv2d_compress)


def run_v300_conv2d_compress(test_arg):
    Ci0_dict = {
        "float32": 8,
        "float16": 16,
        "int8": 32,
        "bfloat16": 16
    }
    bias_dtype_dict = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "float32",
        "int8": "int32"
    }
    res_dtype_dict = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "int8": "int32",
        "hf32": "float32"
    }

    def get_input_dict(conv_type, shape_in, shape_w, bias_flag):
        Ni, Ci, Hi, Wi = shape_in
        Cout, Ci, Hk, Wk = shape_w  # [in_channels, channel_multiplier, filter_height, filter_width]

        Ci0 = Ci0_dict[conv_type]
        Ci1 = (Ci + Ci0 - 1) // Ci0
        shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
        shape_w_fracz = [Ci1*Hk*Wk, (Cout + 15) // 16, 16, 32]

        inputs = {
            "ori_shape": shape_in,
            "ori_format": "NCHW",
            "shape": shape_in_5HD,
            "format": "NC1HWC0",
            "dtype": conv_type,
            "is_first_layer": False
        }
        weights = {
            "ori_shape": shape_w,
            "ori_format": "NCHW",
            "shape": shape_w_fracz,
            "format": "FRACTAL_Z",
            "dtype": conv_type,
        }
        bias = {
            "ori_shape": (Cout,),
            "dtype": bias_dtype_dict[conv_type],
            "shape": (Cout,),
            "format": "ND",
            "ori_format": "ND",
        } if bias_flag else None
        outputs = {
            "ori_shape": [],
            "ori_format": "NCHW",
            "dtype": res_dtype_dict[conv_type],
            "shape": [],
            "format": "NC1HWC0",
        }

        return inputs, weights, bias, outputs

    def v300_conv2d_compress_single_op(casename, conv_type, shape_in, shape_w, pads, strides,
                                       groups, bias_flag):
        from impl.conv2d_compress import conv2dcompress
        from tbe.common.utils.op_util.op_util_conv2d import WEIGHT_SPARSE_4_2
        inputs, weights, bias, outputs = get_input_dict(conv_type, shape_in, shape_w, bias_flag)
        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]
        compress_index_shape = weights.get("shape")[:-1] + [weights.get("shape")[:-1] // 4]
        compress_index = {
            "ori_shape": shape_w,
            "ori_format": "NCHW",
            "shape": compress_index_shape,
            "format": "FRACTAL_Z",
            "dtype": "int8",
        }
        conv2dcompress(inputs, weights, compress_index, bias, None, outputs, strides, pads,
                       dilations, groups=groups, data_format='NCHW', alg=WEIGHT_SPARSE_4_2,
                       kernel_name=casename)

    def v300_conv2d_compress_compute_op(casename, conv_type, shape_in, shape_w, pads, strides,
                                        groups, bias_flag):
        import tvm
        from tbe.dsl import auto_schedule
        from impl.conv2d_compress import conv2dcompress_compute
        from tbe.common.utils.op_util.op_util_conv2d import WEIGHT_SPARSE_4_2
        inputs, weights, bias, outputs = get_input_dict(conv_type, shape_in, shape_w, bias_flag)
        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]
        with tvm.target.cce():
            fmap = tvm.placeholder(inputs.get("shape"), name='fmap', dtype=conv_type)
            weight = tvm.placeholder(weights.get("shape"), name='weight', dtype=conv_type,
                                     attrs={'ori_shape': shape_w, 'ori_format': "NCHW"})
            compress_index_shape = weights.get("shape")[:-1] + [weights.get("shape")[-1] // 4]

            compress_index = tvm.placeholder(compress_index_shape,
                                             name='weight_index',
                                             dtype=conv_type,
                                             attrs={
                                                 'ori_shape': shape_w,
                                                 'ori_format': "NCHW"
                                             })
            bias = tvm.placeholder(bias.get("shape"), name='bias',
                                   dtype=bias.get("dtype")) if bias_flag else None
            out = conv2dcompress_compute(fmap, weight, compress_index, bias, None, outputs,
                                         strides, pads, dilations, groups=groups,
                                         data_format='NCHW',
                                         alg=WEIGHT_SPARSE_4_2, kernel_name="conv2dcompress",
                                         options=None)
            sch = auto_schedule(out)

    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.context import op_context
    v300_cases = [
        ("conv2d", "conv2d", "int8", (1, 3, 17, 17), (64, 3, 7, 7), (3, 3, 3, 3), (1, 1), 1, 1,
         False, 0, 0, 0)
    ]
    set_current_compile_soc_info("Ascend320")
    with op_context.OpContext():
        for case in v300_cases:
            print("=" * 150)
            print("case {}".format(case))
            casename, dataflow, conv_type, shape_in, shape_w, pads, strides, \
            dilation, groups, bias_flag, quant_scale, quant_offset, relu_param = case
            # v300_conv2d_compress_single_op(casename, conv_type, shape_in, shape_w, pads, strides,
            #                                groups, bias_flag)
            v300_conv2d_compress_compute_op(casename, conv_type, shape_in, shape_w, pads, strides,
                                            groups, bias_flag)


ut_case.add_cust_test_func(test_func=run_v300_conv2d_compress)


if __name__ == "__main__":
    ut_case.add_cust_test_func(test_func=test_conv2d_compress)
    run_v300_conv2d_compress("")
    exit(0)
