#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_lx(test_arg):
    import sys
    import numpy as np
    import te.lang.cce
    from te import tvm
    from tbe.dsl import auto_schedule
    from te import platform as cce_conf
    from impl.conv2d import conv2d_compute
    from impl.conv2d import _conv_layer_cce
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.eltwise import eltwise_compute
    from impl.read_select import read_select_compute
    from impl.write_select import write_select_compute
    from te.platform.cce_policy import set_L1_info

    sys.path.append("./llt/ops/ut/testcase_python/")
    sys.path.append("./llt/ops/common/op_cfg/")

    set_L1_info("L1_fusion_enabled", True)

    import conv_lx_fusion_testcase as tc


    def conv_lx_fusion_quant_case(shape_in, shape_w, pads, strides, data_flow, bias_flag, dilations,
                                  relu_flag, deq_vector, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset, orig_shape_w,
                                  l1_fusion_type, fm_addr_type, out_addr_type, fm_valid_shape, fm_offset,
                                  eltwise_shape, eltwise_valid_shape, eltwise_addr_type, eltwise_offset, eltwise_rs_stride,
                                  ws_valid_shape, kernel_name_val):
        q_scaler = np.float16(q_scaler)
        q_offset = np.float16(q_offset)
        with tvm.target.cce():
            # conv2d
            c_out = shape_w[1] * 16
            shape_c = (1, shape_w[1], 1, 1, 16) if deq_vector else (1, 1, 1, 1, 1)
            L1_valid_size = 1
            if fm_valid_shape:
                valid_shape = fm_valid_shape
            else:
                valid_shape = shape_in
            for i in valid_shape:
                L1_valid_size = L1_valid_size*i
            fm = tvm.placeholder(shape_in, name='fm', dtype='int8', attrs={'ori_format': 'NCHW',
                                                                           'addr_type': fm_addr_type,
                                                                           'valid_shape': fm_valid_shape,
                                                                           'slice_offset': fm_offset,
                                                                           'L1_fusion_type': l1_fusion_type,
                                                                            "L1_valid_size": L1_valid_size,
                                                                              "L1_addr_flag": 1})
            filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='int8',
                                       attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
            tensor_list = [fm, filter_w]
            bias = None
            if bias_flag:
                bias = tvm.placeholder((c_out,), name='bias', dtype='int32')
                tensor_list.append(bias)
            deq = tvm.placeholder(shape_c, name='deq', dtype='float16', attrs={'ori_shape': [c_out if deq_vector else 1]})
            tensor_list.append(deq)

            # s8 -> s32
            conv_res = conv2d_compute(fm, filter_w, bias, None, {}, strides, pads, dilations, offset_x=0)
            out = ascend_dequant_compute(conv_res, deq, None, sqrt_mode=deq_sqrt, relu_flag=deq_relu)

            if data_flow == 0 and ws_valid_shape:
                out = write_select_compute(out, {"valid_shape": ws_valid_shape})

            if data_flow == 1:
                out = ascend_quant_compute(out, None, q_scaler, q_offset, q_sqrt)
                if ws_valid_shape:
                    out = write_select_compute(out, {"valid_shape": ws_valid_shape})

            if data_flow in (2, 3, 4):
                if eltwise_shape and eltwise_valid_shape:
                    fm2 = tvm.placeholder(eltwise_shape, name='fmap2', dtype="float16", attrs={'ori_format': 'NCHW',
                                                                                               'addr_type': eltwise_addr_type,
                                                                                               'valid_shape': eltwise_valid_shape,
                                                                                               'slice_offset': eltwise_offset})
                    data_other = read_select_compute(fm2, {}, [0, 0, eltwise_rs_stride[0], eltwise_rs_stride[1], 0])
                else:
                    fm2 = tvm.placeholder(out.shape, name='fmap2', dtype="float16",
                                          attrs={'ori_format': 'NCHW', 'addr_type': eltwise_addr_type})
                    data_other = fm2
                tensor_list.append(fm2)

            if data_flow == 4:
                res = eltwise_compute([out, data_other], None)
                if relu_flag:
                    res = leaky_relu_compute(res, None)
                if ws_valid_shape:
                    out = write_select_compute(res, {"valid_shape": ws_valid_shape})
                else:
                    out = res

            if data_flow == 2:
                res = eltwise_compute([out, data_other], None)
                if relu_flag:
                    res = leaky_relu_compute(res, None)
                out = ascend_quant_compute(res, None, q_scaler, q_offset, q_sqrt)
                if ws_valid_shape:
                    out = write_select_compute(out, {"valid_shape": ws_valid_shape})

            if data_flow == 3:
                res = eltwise_compute([out, data_other], None)
                print("res_shape ", res.shape)
                if relu_flag:
                    res_relu = leaky_relu_compute(res, None)
                else:
                    res_relu = res
                if ws_valid_shape and ws_valid_shape[0]:
                    res_relu_ws = write_select_compute(res_relu, {"valid_shape": ws_valid_shape[0]})
                res = ascend_quant_compute(res_relu, None, q_scaler, q_offset, q_sqrt)
                if ws_valid_shape and ws_valid_shape[1]:
                    out = write_select_compute(res, {"valid_shape": ws_valid_shape[1]})
                else:
                    out = res

                out.op.attrs["addr_type"] = out_addr_type[1]
                if ws_valid_shape and ws_valid_shape[0]:
                    res_relu_ws.op.attrs["addr_type"] = out_addr_type[0]
                    out = [res_relu_ws, out]
                else:
                    res_relu.op.attrs["addr_type"] = out_addr_type[0]
                    out = [res_relu, out]

            import collections.abc
            if isinstance(out, collections.abc.Sequence):
                tensor_list.extend(out)
            else:
                out.op.attrs["addr_type"] = out_addr_type[0]
                tensor_list.append(out)

            sch = auto_schedule(out)

        return sch, tensor_list


    def conv_lx_fusion_case(shape_in, shape_w, pads, strides, bias_flag, orig_shape_w, dilations,
                            vadd_flag, relu_flag, l1_fusion_type, fm_addr_type, out_addr_type, fm_valid_shape, fm_offset,
                            eltwise_shape, eltwise_valid_shape, eltwise_addr_type, eltwise_offset, eltwise_rs_stride,
                            ws_valid_shape, kernel_name_val):
        with tvm.target.cce():
            # conv2d
            c_out = shape_w[1] * 16
            L1_valid_size = 1
            if fm_valid_shape:
                valid_shape = fm_valid_shape
            else:
                valid_shape = shape_in
            for i in valid_shape:
                L1_valid_size = L1_valid_size*i
            fm = tvm.placeholder(shape_in, name='fm', dtype='float16', attrs={'ori_format': 'NCHW',
                                                                              'addr_type': fm_addr_type,
                                                                              'valid_shape': fm_valid_shape,
                                                                              'slice_offset': fm_offset,
                                                                              'L1_fusion_type': l1_fusion_type,
                                                                               "L1_valid_size": L1_valid_size,
                                                                              "L1_addr_flag": 1})
            filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='float16',
                                       attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})

            bias = None
            if bias_flag:
                bias = tvm.placeholder((c_out,), name='bias', dtype='float16')

            conv_res = conv2d_compute(fm, filter_w, bias, None, {}, strides, pads, dilations, offset_x=0)
            if vadd_flag:
                if eltwise_shape and eltwise_valid_shape:
                    fm2 = tvm.placeholder(eltwise_shape, name='fmap2', dtype="float16", attrs={'ori_format': 'NCHW',
                                                                                               'addr_type': eltwise_addr_type,
                                                                                               'valid_shape': eltwise_valid_shape,
                                                                                               'slice_offset': eltwise_offset})
                    data_other = read_select_compute(fm2, {}, [0, 0, eltwise_rs_stride[0], eltwise_rs_stride[1], 0])
                else:
                    fm2 = tvm.placeholder(conv_res.shape, name='fmap2', dtype="float16",
                                          attrs={'ori_format': 'NCHW', 'addr_type': eltwise_addr_type})
                    data_other = fm2
            tensor_list = [fm, filter_w]
            if bias_flag:
                tensor_list.append(bias)
            if vadd_flag:
                tensor_list.append(fm2)

            if vadd_flag:
                conv_res = eltwise_compute([conv_res, data_other], None)
            if relu_flag:
                print("~~~~~~~~~~~~~~~~~~~~~~ relu_flag", relu_flag)

                conv_res = leaky_relu_compute(conv_res, None)
            if ws_valid_shape:
                res = write_select_compute(conv_res, {"valid_shape": ws_valid_shape})
            else:
                res = conv_res

            res.op.attrs["addr_type"] = out_addr_type[0]
            tensor_list.append(res)
            sch =  auto_schedule(res)

        return sch, tensor_list


    def conv_lx_fusion(fm_shape, filter, pads, strides, bias_flag, dilations, data_flow,
                       vadd_flag, relu_flag, deq_vector, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset,
                       l1_fusion_type, fm_addr_type, out_addr_type, fm_valid_shape, fm_offset,
                       eltwise_shape, eltwise_valid_shape, eltwise_addr_type, eltwise_offset, eltwise_rs_stride,
                       ws_valid_shape, kernel_name_val):
        if data_flow == 5:
            block_size_k = 16
            C0 = 16
        else:
            block_size_k = 32
            C0 = 32
        block_size_n = 16
        batch, channel, height, weight = fm_shape
        C1 = (channel + C0 - 1) // C0
        shape_in = (batch, C1, height, weight, C0)

        out_channel = filter[0]
        in_channel_weight = ((filter[1] + block_size_k - 1) // block_size_k) * block_size_k
        filter_h = filter[2]
        filter_w = filter[3]

        if data_flow == 5:
            c_out = (out_channel + block_size_n - 1) // block_size_n
        else:
            c_out = (out_channel + block_size_k - 1) // block_size_k * block_size_k
            c_out = (c_out + block_size_n - 1) // block_size_n
        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                   c_out, block_size_n, block_size_k)

        if data_flow == 5:
            if not vadd_flag and not relu_flag:
                raise RuntimeError("It is impossible that vadd and relu do not exist at the same time for conv_fusion!")
            sch, tensor_list = conv_lx_fusion_case(shape_in, shape_w, pads, strides, bias_flag, filter, dilations,
                                                   vadd_flag, relu_flag, l1_fusion_type, fm_addr_type, out_addr_type,
                                                   fm_valid_shape, fm_offset,
                                                   eltwise_shape, eltwise_valid_shape, eltwise_addr_type, eltwise_offset,
                                                   eltwise_rs_stride, ws_valid_shape, kernel_name_val)
        else:
            sch, tensor_list = conv_lx_fusion_quant_case(shape_in, shape_w, pads, strides, data_flow, bias_flag, dilations,
                                                         relu_flag, deq_vector, deq_sqrt, deq_relu, q_sqrt, q_scaler,
                                                         q_offset, filter,
                                                         l1_fusion_type, fm_addr_type, out_addr_type, fm_valid_shape,
                                                         fm_offset,
                                                         eltwise_shape, eltwise_valid_shape, eltwise_addr_type,
                                                         eltwise_offset, eltwise_rs_stride, ws_valid_shape, kernel_name_val)

        config = {"print_ir": False,
                  "need_build": True,
                  "name": kernel_name_val,
                  "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def conv_layer(fm_shape, filter, pads, strides, bias_flag, data_flow, dilations,
                   l1_fusion_type, fm_addr_type, out_addr_type, fm_valid_shape, fm_offset, kernel_name_val):
        if data_flow == -2:
            fm_type = "float16"
            weight_type = "float16"
            output_type = "float16"
        else:
            fm_type = "int8"
            weight_type = "int8"
            output_type = "int32"

        padh = pads[0:2]
        padw = pads[2:]
        strideh = strides[2]
        stridew = strides[3]
        dilateh = dilations[2]
        dilatew = dilations[3]
        bias_tensor = False
        if bias_flag == 1:
            bias_tensor = True

        fusion_para = {"input_memory_type": fm_addr_type, "output_memory_type": out_addr_type[0], \
                       "valid_shape": fm_valid_shape, "slice_offset": fm_offset, \
                       "l1_fusion_type": -1}
        _conv_layer_cce(shape_in=fm_shape, shape_w=filter, in_dtype=fm_type,
          w_dtype=weight_type, res_dtype=output_type, padh=padh, padw=padw,
          strideh=strideh, stridew=stridew, dilateh=dilateh, dilatew=dilatew, bias=bias_tensor,
          fusion_para=fusion_para, kernel_name=kernel_name_val, need_build=True, need_print=False)


    def test_conv(fm_shape, filter, pads, strides, bias_flag, dilations, data_flow,
                  vadd_flag, relu_flag, deq_vector, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset,
                  l1_fusion_type, l1_space, fm_addr_type, out_addr_type, fm_valid_shape, fm_offset,
                  eltwise_shape, eltwise_valid_shape, eltwise_addr_type, eltwise_offset, eltwise_rs_stride, ws_valid_shape,
                  kernel_name_val):

        if isinstance(out_addr_type, int):
            out_addr_type = [out_addr_type]

        if data_flow in (-1, -2):
            conv_layer(fm_shape, filter, pads, strides, bias_flag, data_flow, dilations,
                       l1_fusion_type, fm_addr_type, out_addr_type, fm_valid_shape, fm_offset, kernel_name_val)
        else:
            conv_lx_fusion(fm_shape, filter, pads, strides, bias_flag, dilations, data_flow,
                           vadd_flag, relu_flag, deq_vector, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset,
                           l1_fusion_type, fm_addr_type, out_addr_type, fm_valid_shape, fm_offset,
                           eltwise_shape, eltwise_valid_shape, eltwise_addr_type, eltwise_offset, eltwise_rs_stride,
                           ws_valid_shape, kernel_name_val)

    def run_testcase():
        testcases = tc.gen_dict_4_op("conv_lx_fusion_ut", platform="mini")
        testcase_for_mini = testcases["mini"]
        for key in testcase_for_mini:
            print("run the case:", key)
            test_conv(*testcase_for_mini[key])
            print("[success: %s]" % key)

    def test_conv_lx_fusion():
        set_L1_info("L1_fusion_enabled", False)
        set_L1_info("L2_fusion_enabled", True)
        print("[ UNITTEST START conv2d lx_fusion]")
        cce_conf.te_set_version("Ascend310", l2_fusion="true")
        run_testcase()
        cce_conf.te_set_version("Ascend310", l2_fusion="false")

    test_conv_lx_fusion()

# ut_case.add_cust_test_func(test_func=test_conv2d_lx)

if __name__ == '__main__':
    ut_case.add_cust_test_func(test_func=test_conv2d_lx)
    exit(0)
