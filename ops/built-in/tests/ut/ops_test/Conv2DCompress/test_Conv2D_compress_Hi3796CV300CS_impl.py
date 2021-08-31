#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")
def test_conv2d_compress_hi3796cv300cs(test_arg):
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

    def _conv_lhisi_fusion_case(shape_in, shape_w, pads, strides, c_out, orig_shape_w, data_flow, bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset):
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
            conv_res = conv2dcompress_compute(fm, filter_w, compress_index, bias_tensor, None, None, strides, pads, dilations, 1, 'NCHW', q_offset)
            deq_reg = tvm.placeholder(shape_c,
                          name='deq',
                          dtype='uint64',
                          attrs={'ori_shape': [shape_w[1] * 16]})
            out = ascend_dequant_compute(conv_res, deq_reg, None, deq_sqrt, deq_relu)

            fm2 = tvm.placeholder(out.shape, name='fmap2', dtype="float16", attrs={'ori_format': 'NCHW'})
            tensor_list = [fm, filter_w, compress_index, deq_reg]
            if data_flow == 1:
                out = ascend_quant_compute(out, None, q_scaler, q_offset, q_sqrt)
            if data_flow in (2, 3):
                res_add = eltwise_compute([fm2, out], None)
                res_relu = leaky_relu_compute(res_add, None)
                res_quant = ascend_quant_compute(res_relu, None, q_scaler, q_offset, q_sqrt)
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
            optim_dict = None
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
        cceconf.cce_conf.te_set_version("Hi3796CV300CS")
        testcases = tc.gen_dict_4_op("conv_compress", platform="Hi3796CV300CS")
        print("Hi3796CV300CS:", testcases)
        bin_path_val = "./llt/ops/common/kernel_bin/conv_compress/"
        print("bin_path_val:", bin_path_val)
        run_testcase(testcases, _test_conv_lhisi)
        print("end")

    test_cce_conv200()

# ut_case.add_cust_test_func(test_func=test_conv2d_compress_hi3796cv300cs)

if __name__ == "__main__":
    # ut_case.add_cust_test_func(test_func=test_conv2d_compress_hi3796cv300cs)
    exit(0)
