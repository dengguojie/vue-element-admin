# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT
from impl.dynamic.fix_pipe import fixpipe_compute
from impl.dynamic.fix_pipe import fix_pipe

ut_case = OpUT("FixPipe", "impl.dynamic.fix_pipe", "fix_pipe")

def coverage():
    from impl.fixpipe_op.fixpipe_util import create_placeholder
    input_dict = {
        "shape": [1, 1, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "float32",
        "ori_shape": [1, 1, 112, 112, 16]
    }
    create_placeholder(input_dict, "in")


def test_dynamic_fixpipe(test_arg):
    from te import tvm
    from tbe.dsl.base import operation
    from tbe.dsl import auto_schedule
    from tbe.dsl import build
    from impl.dynamic.fix_pipe import fixpipe_compute
    from impl.dynamic.conv2d import _conv2d_compute

    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend320", "AiCore")

    # case name: ((fm_range), (weight_shape), (paddings), (strides), (dilations), group, bias_flag, dtype)
    testcase = {
        # "conv2d_dynamic_fixpipe_test_fp16_1": ([(1, 4), (256, 256), (56, 100), (56, 100)], (512, 256, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        # "conv2d_dynamic_fixpipe_test_fp16_2": ([(1, 1), (128, 128), (1, 30), (1, 30)], (512, 128, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        # "conv2d_dynamic_fixpipe_test_fp16_3": ([(1, 4096), (64, 64), (1, 56), (1, 56)], (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_dynamic_fixpipe_test_fp16_4": ([(1, 4), (3, 3), (200, 224), (224, 300)], (64, 3, 7, 7), [2, 3, 2, 3], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16")}

    def compile_dynamic_fixpipe(fm_range, weight_shape, pads, strides, dilations, groups, bias_flag, input_dtype, kernel_name):

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

                Ni, Ci, Hi, Wi = inputs.get("ori_shape")
                Co, _, Hk, Wk = weights.get("ori_shape")

                ho = (Hi + (pads[0] + pads[1]) - Hk) // strides[0] + 1
                wo = (Wi + (pads[2] + pads[3]) - Wk) // strides[1] + 1

                shape_out = conv_out.shape
                shape_out_5hd = [shape_out[0], shape_out[1], ho, wo, shape_out[3]]
                output = {
                    "shape": shape_out_5hd,
                    "format": "NC1HWC0",
                    "dtype": "float16"
                }

                out = fixpipe_compute(conv_out, None, None, None, None, None, None, None, None, None, output, [], ["pre_act"], "")

                with tvm.target.cce():
                    sch = auto_schedule(out)

                tensor_list = list(conv_res['op_placeholder'])

                config = {"name": kernel_name,
                          "tensor_list": tensor_list}

    for key, value in testcase.items():
        print("test begin test_dynamic_fixpipe case:", key)
        compile_dynamic_fixpipe(*value, key)
        print("test end test_dynamic_fixpipe case:", key)

def test_dynamic_fixpipe_single(test_arg):
    from impl.dynamic.fix_pipe import fix_pipe
    from tbe.dsl.base import operation
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend320", "AiCore")

    testcase = {
        # "conv2d_dynamic_fixpipe_test_fp16_1": ([(1, 4), (256, 256), (56, 100), (56, 100)], (512, 256, 1, 1), [0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16"),
        # "conv2d_dynamic_fixpipe_test_fp16_2": ([(1, 1), (128, 128), (1, 30), (1, 30)], (512, 128, 1, 1), [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        # "conv2d_dynamic_fixpipe_test_fp16_3": ([(1, 4096), (64, 64), (1, 56), (1, 56)], (64, 64, 3, 3), [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1, False, "float16"),
        "conv2d_dynamic_fixpipe_test_fp16_4": ([(1, 4), (3, 3), (200, 224), (224, 300)], (64, 3, 7, 7), [2, 3, 2, 3], [1, 1, 2, 2], [1, 1, 1, 1], 1, False, "float16")}

    def compile_dynamic_fixpipe_single(fm_range, weight_shape, pads, strides, dilations, groups, bias_flag, input_dtype, kernel_name):

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

        with operation.dynamic():
            with operation.ComputeContext():
                Ni, Ci, Hi, Wi = inputs.get("ori_shape")
                Co, _, Hk, Wk = weights.get("ori_shape")

                ho = (Hi + (pads[0] + pads[1]) - Hk) // strides[0] + 1
                wo = (Wi + (pads[2] + pads[3]) - Wk) // strides[1] + 1

                shape_out = inputs['ori_shape']
                shape_out_5hd = [shape_out[0], shape_out[1], ho, wo, shape_out[3]]
                inputs = {"ori_shape": fm_ori_shape, "dtype": input_dtype, "ori_format": "NCHW",
                          "range": fm_range, "shape": shape_out}

                output = {
                    "shape": shape_out_5hd,
                    "format": "NC1HWC0",
                    "dtype": "float16"
                }
                try:
                    out = fix_pipe(inputs, None, None, None, None, None, None, None, None, None, output, [], ["pre_act"], "")
                except:
                    print("not support single fix_pipe compile")

    for key, value in testcase.items():
        print("test begin test_dynamic_fixpipe_single case:", key)
        compile_dynamic_fixpipe_single(*value, key)
        print("test end test_dynamic_fixpipe_single case:", key)


if __name__ == "__main__":
    print("test_dynamic_fixpipe running")
    print("====> conv2d v300 ut start")
    test_dynamic_fixpipe("")
    test_dynamic_fixpipe_single("")
    print("====> end v300 ut start")
    exit(0)

