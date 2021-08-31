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
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_pooling(test_arg):
    import os
    import shutil
    import sys
    from te.platform import cce_conf
    from impl.conv2d import conv2d_compute
    from impl.pooling import pool_fuse_compute
    from impl.relu import relu_compute
    from te import tvm
    from tbe.dsl import auto_schedule
    import te.lang.cce
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    testcases = {
            "fm_224_224_conv2d_pooling_1batch": ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), 1, 1, (3, 3), (2, 2), "SAME"),
            "fm_224_224_conv2d_pooling_4batch": ((4, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), 1, 1, (3, 3), (2, 2), "SAME"),
            "fm_224_224_conv2d_pooling_8batch": ((8, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), 1, 1, (3, 3), (2, 2), "SAME"),
            "fm_224_224_conv2d_pooling_16batch": ((16, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), 1, 1, (3, 3), (2, 2), "SAME"),
            "fm_224_224_conv2d_pooling_1batch_no_bias": ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), 0, 1, (3, 3), (2, 2), "SAME"),
            "fm_224_224_conv2d_pooling_4batch_no_relu": ((4, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), 1, 0, (3, 3), (2, 2), "SAME"),
            "fm_224_224_conv2d_pooling_8batch_no_bias_no_relu": ((8, 3, 224, 224), (64, 3, 7, 7), (3, 3), (2, 2), 0, 0, (3, 3), (2, 2), "SAME"),
    }


    def _conv_pool_fusion_case(shape_in, shape_w, pads, strides, c_out, \
        orig_shape_w, bias_flag, relu_flag, window_pool, strides_pool, pad_mode):
        with tvm.target.cce():
            # conv2d
            dilations = [1, 1, 1, 1]
            shape_c = (c_out * 16,)
            fm = tvm.placeholder(shape_in, name='fm', dtype="float16", attrs={'ori_format': 'NCHW', 'format':'NC1HWC0_C04'})
            filter_w = tvm.placeholder(shape_w, name='filter_w', dtype="float16",
                                    attrs={'ori_shape': orig_shape_w, 'ori_format': 'NCHW', 'format':'FRACTAL_Z_C04'})
            # MMAD
            bias_tensor = None
            if bias_flag:
                bias_tensor = tvm.placeholder(shape_c, name='bias_tensor', dtype="float16")
            conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations)


            if relu_flag:
                conv_res = relu_compute(conv_res, None)
            window = window_pool
            padding = [0, 0, 0, 0]
            strides = strides_pool
            mode = pad_mode
            print("the conv shape is : ", conv_res.shape)
            out = pool_fuse_compute(conv_res, None, None, None, window, strides, pad=padding)
            print("the out shape is :", out.shape)
            auto_sch_res = AutoScheduleOp(out)
            sch = auto_schedule(out)
            if bias_flag:
                tensor_list = [fm, filter_w, bias_tensor, out]
            else:
                tensor_list = [fm, filter_w, out]
            if bias_flag and relu_flag:
                fusion_type = 45
            elif not bias_flag and relu_flag:
                fusion_type = 44
            elif bias_flag and not relu_flag:
                fusion_type = 43
            else:
                fusion_type = 42
            assert auto_sch_res.fusion_type == fusion_type

        return sch, tensor_list

    def conv_pooling_fusion(fm_shape, w_shape, padding, strides, \
        bias_flag, relu_flag, window_pool, strides_pool, pad_mode):
        from te.platform.cce_policy import disableL2
        disableL2()
        smalle_channel_flag = False
        if cce_conf.get_soc_spec("SOC_VERSION") in ("Ascend310"):
            platform = "mini"
        else:
            platform = "not_mini"
        block_size_k = 16
        C0 = 16
        block_size_n = 16
        batch, channel, height, weight = fm_shape

        if channel <= 4:
            smalle_channel_flag = True
            if platform == "mini":
                C0 = 16
            else:
                C0 = 4
            weight_c0 = 4

        C1 = (channel + C0 - 1) // C0
        shape_in = (batch, C1, height, weight, C0)

        out_channel = w_shape[0]

        in_channel_weight = ((w_shape[1] + block_size_k - 1) // block_size_k) * block_size_k

        if smalle_channel_flag:
            in_channel_weight = ((w_shape[1] + weight_c0 - 1) // weight_c0) * weight_c0
        filter_h = w_shape[2]
        filter_w = w_shape[3]

        c_out = (out_channel + block_size_n - 1) // block_size_n

        shape_w = ((in_channel_weight * filter_h * filter_w + block_size_k - 1) // block_size_k,
                c_out, block_size_n, block_size_k)
        if smalle_channel_flag:
            shape_w = ((weight_c0 * filter_h * filter_w + block_size_k - 1) // block_size_k,
                    c_out, block_size_n, block_size_k)

        padding_4d = [padding[0], padding[0], padding[1], padding[1]]
        strides_4d = [0, 0, strides[0], strides[1]]  # NCHW

        sch, tensor_list = _conv_pool_fusion_case(shape_in, shape_w, \
            padding_4d, strides_4d,c_out, w_shape, bias_flag, relu_flag, \
            window_pool, strides_pool, pad_mode)
        from te.platform.fusion_manager import get_fusion_build_cfg
        config = {"print_ir": False,
                "need_build": True,
                "name": "conv_pooling",
                "tensor_list": tensor_list,
                "fusion_build_config": get_fusion_build_cfg()}
        te.lang.cce.cce_build_code(sch, config)


    def run_testcase():
        for key in testcases:
            conv_pooling_fusion(*testcases[key])
            print("[passed: %s]" % key)

    def run_input_shape_size_error_case():
        fm = tvm.placeholder((1, 224, 224), name='fm', dtype="float16", attrs={'ori_format': 'NCHW', 'format':'NC1HWC0_C04'})
        try:
            res = te.lang.cce.max_pool_compute(fm, (2, 2), (2, 2), "SAME", (0, 0, 0, 0), 0)
        except RuntimeError:
            print("[Passed: run_input_shape_size_error_case pass]")
        else:
            print("[Passed: run_input_shape_size_error_case end]")
    """
    The UT for cce Test_conv2d_v200
    """

    print("---------------------------------------------------")
    print("[ UNITTEST START conv2d pooling v100]")

    if cce_conf.get_product() in ("1910"):
        run_testcase()
        run_input_shape_size_error_case()
    else:
        pass

print("adding Conv2D pooling testcases")
ut_case.add_cust_test_func(test_func=test_conv2d_pooling)
