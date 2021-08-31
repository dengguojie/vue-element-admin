#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_v200(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from tbe.dsl import auto_schedule
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.conv2d import _conv_layer_cce
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.eltwise import eltwise_compute

    sys.path.append("./llt/ops/ut/testcase_python/")

    testcases = {
        "op_name": "conv_v200",
        "all": {
            # case name: fm_shape, weight_shape, padding, stride, data_type, bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset)
            # data_flow
            #   -2: f16f16->f16
            #   -1: s8s8->s32
            #   0: s32->f16
            #   1: s32->s8
            #   2: s32->f16->s8
            #   3: s32->f16->s8/s16 double output
            "conv_v200_s32out": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], -2, 0, 0, 0, 0, 0, 0),
            "conv_v200_f16out": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], -1, 0, 0, 0, 0, 0, 0),
            "conv_v200_s32in_fp16out": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 0, 0, 0, 0, 0, 0, 0),
            "conv_v200_s32in_fp16out1": ((1, 32, 7, 7), (32, 32, 2, 2), [1, 1, 1, 1], [1, 1, 1, 1], 0, 1, 0, 0, 0, 0, 0),
            "conv_v200_s32in_fp16out2": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 0, 0, 1, 1, 0, 0, 0),
            "conv_v200_s32in_fp16out3": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 0, 1, 0, 1, 0, 0, 0),
            "conv_v200_s32in_fp16out4": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 0, 0, 1, 0, 0, 0, 0),

            "conv_v200_s32in_s8out": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 1, 0, 0, 1, 1, -14),
            "conv_v200_s32in_s8out1": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 0, 1, 1, 0, -0.5, 0),
            "conv_v200_s32in_s8out2": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 1, 0, 1, 1, 0.5, -3),
            "conv_v200_s32in_s8out3": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 1, 0, 1, 0, 0, 1, -128),

            "conv_v200_s32in_elt_s8out": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 2, 1, 0, 0, 1, 1, -14),
            "conv_v200_s32in_elt_s8out1": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 2, 0, 1, 1, 0, -0.5, 0),
            "conv_v200_s32in_elt_s8out2": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 2, 1, 0, 1, 1, 0.5, -3),
            "conv_v200_s32in_elt_s8out3": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 2, 0, 1, 0, 0, 1, -128),

            "conv_v200_s32in_elt_s8f16out": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 3, 1, 0, 0, 1, 1, -14),
            "conv_v200_s32in_elt_s8f16out1": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 3, 0, 1, 1, 0, -0.5, 0),
            "conv_v200_s32in_elt_s8f16out2": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 3, 1, 0, 1, 1, 0.5, -3),
            "conv_v200_s32in_elt_s8f16out3": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 3, 0, 1, 0, 0, 1, -128),

            "conv_v200_s32in_elt_f16out1": ((1, 32, 7, 7), (32, 32, 2, 2), [0, 0, 0, 0], [1, 1, 1, 1], 4, 0, 1, 0, 0, 0, 0),
        }
    }


    def conv_v200_fusion_case(shape_in, shape_w, pads, strides, data_flow,
                            bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset, orig_shape_w):
        with tvm.target.cce():
            # conv2d
            dilations = [1, 1, 1, 1]
            c_out = shape_w[1] * 16
            shape_c = (1, shape_w[1], 1, 1, 16)
            fm = tvm.placeholder(shape_in, name='fm', dtype='int8', attrs={'ori_format': 'NCHW'})
            filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='int8',
                                    attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW'})
            deq = tvm.placeholder(shape_c,
                                name='deq',
                                dtype='uint64',
                                attrs={'ori_shape': [shape_w[1] * 16]})

            bias = None
            if bias_flag:
                bias =  tvm.placeholder((c_out,), name='bias', dtype='int32')
            # s8 -> s32
            conv_res = conv2d_compute(fm, filter_w, bias, None, None, strides, pads, dilations, offset_x=q_offset)
            out = ascend_dequant_compute(conv_res, deq, None, sqrt_mode=deq_sqrt, relu_flag=deq_relu)
            fm2 = tvm.placeholder(out.shape, name='fmap2', dtype="float16", attrs={'ori_format': 'NCHW'})
            tensor_list = [fm, filter_w, deq]

            if data_flow == 1:
                out = ascend_quant_compute(out, None, q_scaler, q_offset, q_sqrt)

            if data_flow in (2,3):
                res_add = eltwise_compute([fm2, out], None)
                res_relu = leaky_relu_compute(res_add, None)
                res_quant = ascend_quant_compute(res_relu, None, q_scaler, q_offset, q_sqrt)
                out = res_quant
                tensor_list.append(fm2)
                if data_flow == 3:
                    out = [res_relu, res_quant]
            if data_flow == 4:
                res_add = eltwise_compute([fm2, out], None)
                res_relu = leaky_relu_compute(res_add, None)
                out = res_relu
                tensor_list.append(fm2)


            if bias_flag:
                tensor_list.append(bias)
            import collections.abc
            if isinstance(out, collections.abc.Sequence):
                tensor_list.extend(out)
            else:
                tensor_list.append(out)

            sch = auto_schedule(out)

        return sch, tensor_list


    def conv_v200(fm_shape, filter, pads, strides, data_flow, bias_flag,
                deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset, kernel_name_val):

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
        _conv_layer_cce(shape_in=fm_shape, shape_w=filter, in_dtype=fm_type,
            w_dtype=weight_type, res_dtype=output_type, padh=padh, padw=padw,
            strideh=strideh, stridew=stridew, bias=bias_tensor, kernel_name=kernel_name_val)

    def conv_v200_fusion(fm_shape, filter, pads, strides, data_flow,
                        bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset, kernel_name_val):
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

        if data_flow == 0:
            c_out = (out_channel + block_size_n - 1) // block_size_n
        else:
            c_out = (out_channel + block_size_k - 1) // block_size_k * block_size_k
            c_out = (c_out + block_size_n - 1) // block_size_n
        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                c_out, block_size_n, block_size_k)
        print("run the case:",kernel_name_val)
        sch, tensor_list = conv_v200_fusion_case(shape_in, shape_w, pads, strides, data_flow,
                                                bias_flag, deq_sqrt, deq_relu, q_sqrt, q_scaler, q_offset, filter)

        config = {"print_ir": False,
                "need_build": True,
                "name": kernel_name_val,
                "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(sch, config)


    def run_testcase():
        testcases_for_all = testcases["all"]
        for key in testcases_for_all:
            if testcases_for_all[key][4] in (-2, -1):
                conv_v200(*testcases_for_all[key], key)
            else:
                conv_v200_fusion(*testcases_for_all[key], key)
            print("[passed: %s]" % key)

    def set_ddk_version(version):
        if version == "v100":
            ddk_info = "Ascend310"
        else:
            ddk_info = "Ascend710"
        cce_conf.cce_conf.te_set_version(ddk_info)

    """
    The UT for cce Test_conv2d_v200
    """

    print("---------------------------------------------------")
    set_ddk_version("v200")
    print("[ UNITTEST START conv2d v200]")

    run_testcase()
    set_ddk_version("v100")

print("adding Conv2D v200 new ut testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_v200)

if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)