#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from te import tvm
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.dsl.base import operation
from impl.dynamic.conv2d import _conv2d_compute
from impl.dynamic.bn_training_reduce import bn_training_reduce_compute


ut_case = OpUT("Conv2D", "impl.dynamic.conv2d", "conv2d")


def test_dynamic_conv2d_bn1_mix_l2(test_arg):
    # case name: ((fm_range), (weight_shape), (paddings), (strides), (dilations), group, bias_flag, dtype)
    testcase = {
        "conv2d_dynamic_conv2d_bn1_test_fp16_1": ([(1, 4), (256, 256), (56, 100), (56, 100)], (512, 256, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_dynamic_conv2d_bn1_test_fp16_2": ([(1, 1), (128, 128), (1, 30), (1, 30)], (512, 128, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_dynamic_conv2d_bn1_test_fp16_3": ([(1, 4096), (64, 64), (1, 56), (1, 56)], (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_dynamic_conv2d_bn1_test_fp16_4": ([(1, 4), (3, 3), (200, 224), (224, 300)], (64, 3, 7, 7), [2, 3, 2, 3], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_dynamic_conv2d_bn1_test_fp32_1": ([(1, 4), (256, 256), (56, 100), (56, 100)], (512, 256, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_dynamic_conv2d_bn1_test_fp32_2": ([(1, 1), (128, 128), (1, 30), (1, 30)], (512, 128, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_dynamic_conv2d_bn1_test_fp32_3": ([(1, 4096), (64, 64), (1, 56), (1, 56)], (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float32"),
        "conv2d_dynamic_conv2d_bn1_test_fp32_4": ([(1, 4), (3, 3), (200, 224), (224, 300)], (64, 3, 7, 7), [2, 3, 2, 3], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float32"),
    }

    def compile_dynamic_conv2d_bn1_mix_l2(fm_range, weight_shape, pads, strides, dilations, groups, bias_flag, input_dtype, kernel_name):

        def get_input_shape(input_ranges):
            dyn_input_shape = []
            for in_range in input_ranges:
                if in_range[0] == in_range[1]:
                    dyn_input_shape.append(in_range[0])
                else:
                    dyn_input_shape.append(-1)
            return dyn_input_shape

        fm_ori_shape = get_input_shape(fm_range)
        Cout, Cin_k, Hk, Wk = weight_shape
        weight_range = [(Cout, Cout), (Cin_k, Cin_k), (Hk, Hk), (Wk, Wk)]

        inputs = {"ori_shape": fm_ori_shape, "dtype": input_dtype, "ori_format": "NCHW", "range": fm_range}
        weights = {"ori_shape": weight_shape, "dtype": input_dtype, "ori_format": "NCHW", "range": weight_range}
        bias = {"dtype": input_dtype} if bias_flag else None
        outputs = {"dtype": input_dtype, "ori_format": "NCHW"}

        with operation.dynamic():
            with operation.ComputeContext():
                conv_res = _conv2d_compute(inputs, weights, bias, None, outputs, strides, pads, dilations, groups=groups, data_format="NCHW")
                conv_out = conv_res['op_res'][0]
                axis = [0, 2, 3]
                bn_out = bn_training_reduce_compute(conv_out, {"format": "NC1HWC0"}, None, axis)

                out = [conv_out, bn_out[0], bn_out[1]]
                with tvm.target.cce():
                    sch = auto_schedule(out)

                tensor_list = list(conv_res['op_placeholder'])
                real_outs = sch[0].addition["real_outs"]
                tensor_list.extend(real_outs)

                config = {"name": kernel_name,
                          "tensor_list": tensor_list,
                          "build_args": {"constant_realize_extent_in_infer_bound": False}}
            # build(sch, config)

    for key, value in testcase.items():
        print("test begin test_dynamic_conv2d_bn1_mix_l2 case:", key)
        compile_dynamic_conv2d_bn1_mix_l2(*value, key)
        print("test end test_dynamic_conv2d_bn1_mix_l2 case:", key)


print("test_dynamic_conv2d_bn1_mix_l2 running")
ut_case.add_cust_test_func("Ascend920A", test_func=test_dynamic_conv2d_bn1_mix_l2)
ut_case.run(["Ascend920A"])
