#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_c04_bn1_impl(test_arg):
    from tbe import tvm
    from tbe.dsl import auto_schedule
    from impl.conv2d import conv2d_compute
    from tbe.dsl.compute.conv_compute import ConvParam
    from tbe import dsl
    from impl.bn_training_reduce import bn_training_reduce_compute
    
    def conv_c04_bn1_fusion():
        # special shape for resnet50 first conv2d layer to open c04
        fm_shape = [256, 224, 224, 3]
        batch, height, weight, channel = fm_shape
        c0 = 16
        c1 = (channel + c0 - 1)  // c0
        shape_in = (batch, c1, height, weight, c0)
        c04_weight_shape = [13, 4, 16, 16]
        stride = [1, 2, 2, 1]
        pads = [2, 3, 2, 2]
        dilations = [1,1,1,1]
        with tvm.target.cce():
            fmap = tvm.placeholder(shape_in, dtype="float16", name="fmap",
                                   attrs = {"format":"NC1HWC0",
                                            "ori_shape":[256, 224, 224, 3], "ori_format": "NHWC"})
            weight_c04 = tvm.placeholder(c04_weight_shape, dtype="float16", name="weight_c04",
                                         attrs={"ori_shape": [7, 7, 3, 64],
                                                "ori_format": "HWCN"})
            res_conv = conv2d_compute(fmap, weight_c04, None, None, None, stride, pads, dilations, 1, "NHWC",
                                      0, "fused_conv2d_c04_bn1")
            res_bn = bn_training_reduce_compute(res_conv, {"format": "NC1HWC0"}, res_conv)

            res = [output for output in res_bn]
            res.insert(0, res_conv)
            tiling = {'AL0_matrix':[4, 13, 16, 16, 1, 1], 'CL0_matrix': [4, 4, 16, 16, 1, 1], 'CUB_matrix': [4, 4, 16, 16, 1, 1], 
                      'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0, 'BL0_matrix': [13, 4, 16, 16, 1, 1],
                      'manual_pingpong_buffer': {'AL0_pbuffer': 2, 'AL1_pbuffer': 2, 'AUB_pbuffer': 1, 'BL0_pbuffer': 2, 
                      'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 2, 'CUB_pbuffer': 2, 'UBG_pbuffer': 2},
                      'n_bef_batch_flag': 0, 'AL1_shape': [1, 15, 1, 1], 'BL1_shape': None, 'block_dim': [16, 1, 2, 1], 'CUB_channel_wise_flag': False}
            ConvParam.tiling = tiling
            sch = auto_schedule(res)

            config = {"print_ir": False,
                      "need_build": True,
                      "name": "fused_conv2d_c04_bn1",
                      "tensor_list": [fmap, weight_c04, sch.cce_special["real_out_tensor"][0],
                                      sch.cce_special["real_out_tensor"][1],
                                      sch.cce_special["real_out_tensor"][2]]}
        dsl.build(sch, config)
    conv_c04_bn1_fusion()

print("test_conv2d_c04_bn1_impl")

ut_case.add_cust_test_func(test_func=test_conv2d_c04_bn1_impl)

if __name__ == "__main__":
    from tbe.common.context import op_context
    with op_context.OpContext("pre-static"):
            ut_case.add_cust_test_func(test_func=test_conv2d_c04_bn1_impl)
            ut_case.run(["Ascend910"])
    exit(0)
