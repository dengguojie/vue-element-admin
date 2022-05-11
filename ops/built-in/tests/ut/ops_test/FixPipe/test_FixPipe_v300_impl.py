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

# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from unittest.mock import MagicMock
from unittest.mock import patch
import os
from op_test_frame.ut import OpUT
from te import tvm
from tbe.common.utils import shape_to_list
import tbe

from impl.conv2d import conv2d
from impl.conv2d import conv2d_compute

from tbe.common.context import op_context
from te import platform as cceconf
from te import tvm
from tbe.dsl import auto_schedule
import tbe.common.context.op_info as operator_info
import numpy as np
import ctypes

ut_case = OpUT("FixPipe", "impl.fix_pipe", "fix_pipe")

vals = {("CORE_NUM", ): 1,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 524288,
        ("L0C_SIZE", ): 131072,
        ("Intrinsic_fix_pipe_unit_list",): True,
        ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True,
        ("AICORE_TYPE",): "AiCore"
        }

vals_220 = {("CORE_NUM", ): 1,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 524288,
        ("L0C_SIZE", ): 131072,
        ("Intrinsic_fix_pipe_unit_list",): True,
        ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): False,
        ("AICORE_TYPE",): "AiCore"
        }


def side_effects(*args):
    return vals[args]


def side_effects_220(*args):
    return vals_220[args]


v300_case = [
    # generate .o
    # dataflow,
    # conv_type,
    # shape_in,
    # shape_w,
    # pads,
    # strides,
    # dilation,
    # groups,
    # bias_flag,
    # (quant_scale),
    # (quant_offset),
    # relu_param

    ("conv2d_relu", "conv2d_relu", "float16", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
    ("conv2d_relu_transdata", "conv2d_relu", "float32", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
    ("conv2d_transdata", "conv2d_transdata", "float16", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
    ("conv2d_dequant_add_quant", "conv2d_dequant_add_quant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
    ("conv2d_dequant_add_quant_transdata", "conv2d_dequant_add_quant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
    ("conv2d_dequant_relu6_quant", "conv2d_dequant_relu6_quant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
    ("conv2d_relu", "conv2d_relu", "float32", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
    ("conv2d_dequant_add(anti_quant)_quant", "conv2d_dequant_add_quant", "int8", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
]

single_conv2d = [
    ("conv2d", "conv2d", "float16", (1, 64, 32, 32), (32, 64, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
]

aipp_conv2d = [
    ("conv2d", "conv2d", "float16", (1, 3, 32, 32), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1, False, 0, 0, 0),
]


def eltwise_exists(dataflow):
    if dataflow.find("add") != -1 or dataflow.find("sub") != -1:
        return True
    return False


def create_quant_scale_0(dataflow, shape, quant_scale, quant_offset):
    quant_scale_0 = None
    if dataflow.find("dequant") != -1:
        quant_scale_0 = tvm.placeholder(shape,  # [1, c1, 1, 1, c0]
                                        name='quant_scale_0',
                                        dtype='uint64',
                                        attrs={'ori_shape': [shape[1] * shape[-1]]})
        return quant_scale_0

    if dataflow.find("_quant") != -1 and eltwise_exists(dataflow):
        return

    return quant_scale_0


def conv_float_to_u32(data_f):
    fp = ctypes.pointer(ctypes.c_float(data_f))
    cp = ctypes.cast(fp, ctypes.POINTER(ctypes.c_uint))
    data_hex = cp.contents.value
    result = (data_hex // 8192) * 8192
    return result


def mask(val, width):
    val_mask = int('1' * width, 2)
    return val & val_mask


def gen_quant_scale_1(quant_scale_post, quant_offset_post):
    val = np.uint64(0)
    val += (conv_float_to_u32(quant_scale_post) >> 13) << 13  # M3 [31:13]
    val += mask(quant_offset_post, width=9)  # offset [0:8]
    val += (0b1 << 9)  # sign [9]
    return np.uint64(val)


def create_quant_scale_1(dataflow, shape, quant_scale, quant_offset):
    if dataflow.find("add_quant") != -1 or dataflow.find("sub_quant") != -1:
        quant_scale_1_value = gen_quant_scale_1(quant_scale, quant_offset)
        quant_scale_1 = tvm.placeholder([1],  # [1, c1, 1, 1, c0]
                                        name='quant_scale_1',
                                        dtype='uint64',
                                        attrs={'ori_shape': [1],
                                               "const_value": [quant_scale_1_value]})
        return quant_scale_1
    return None


def create_relu_weight_0(dataflow, shape, relu_param):
    relu_weight_0 = None
    if dataflow.find("_lrelu") != -1:
        if dataflow.find("dequant_lrelu") != -1:
            relu_weight_0 = tvm.placeholder(shape,  # [1, c1, 1, 1, c0]
                                            name='relu_weight_0',
                                            dtype='float32',
                                            attrs={'ori_shape': [shape[1] * shape[-1]]})
        else:
            relu_weight_0 = tvm.placeholder([1],  # [1]
                                            name='relu_weight_0',
                                            dtype='float32',
                                            attrs={"const_value": [relu_param],
                                                   "ori_shape": [1]})
    return relu_weight_0


def create_relu_weight_1(dataflow, shape):
    return None


def create_clip_value_0(dataflow, shape, qscale, qoffset):
    if dataflow.find("_relu6") != -1:
        if dataflow.find("_quant") == -1:
            qscale = 1
            qoffset = 0
        clip_value = round(6 * qscale) + qoffset
        print("===>clip_value:", clip_value)
        clip_value_0 = tvm.placeholder([1],  # [n, c1, hw, c0]
                                       name='clip_value_0',
                                       dtype="float16",
                                       attrs={"const_value": [clip_value],
                                              "ori_shape": [1]})
        return clip_value_0

    return None


def create_clip_value_1(dataflow, shape):
    return None


def create_anti_quant_scale(casename, dataflow, shape):
    if casename.find("anti_quant") != -1:
        anti_quant_scale = tvm.placeholder([1],  # [n, c1, hw, c0]
                                       name='anti_quant_scale',
                                       dtype="float16",
                                       attrs={"const_value": [1],
                                              "ori_shape": [1]})
        return anti_quant_scale
    return None


def create_anti_quant_offset(casename, dataflow, shape):
    if casename.find("anti_quant") != -1:
        anti_quant_offset = tvm.placeholder([1],  # [n, c1, hw, c0]
                                       name='anti_quant_offset',
                                       dtype="int8",
                                       attrs={"const_value": [1],
                                              "ori_shape": [1]})
        return anti_quant_offset
    return None


def create_eltwise_src(casename, dataflow, shape):
    if dataflow.find("add") == -1 and dataflow.find("sub") == -1:
        return None

    shepe_x2 = shape_to_list(shape)
    data_type = "float16"
    if casename.find("anti_quant") != -1:
        data_type = "int8"
        shepe_x2 = [shepe_x2[0], shepe_x2[1]//2, shepe_x2[2], shepe_x2[3] * 2]

    x1 = tvm.placeholder(shepe_x2,  # [n, c1, hw, c0]
                         name='x1',
                         dtype=data_type,
                         attrs={'ori_shape': shape_to_list(shape)})
    return x1


def get_eltwise_mode(dataflow):
    eltwise_mode = "None"
    if dataflow.find("add") != -1:
        return "ADD"
    if dataflow.find("sub") != -1:
        return "SUB"
    return ""


def get_unit_list(dataflow):
    unit_list = []
    op_list = dataflow.split("_")

    if dataflow == "conv2d_dequant_relu_quant":
        unit_list = ["pre_conv",
                     "pre_act"]  # pre_conv, pre_act, post_eltwise, post_act, post_quant, post_platform

    if "relu" in op_list or "relu6" in op_list:
        unit_list = ["pre_act"]

    if dataflow == "conv2d_relu6":
        unit_list = ["pre_act"]

    return unit_list


def get_output_dict(dataflow, shape, in_dtype):
    if dataflow.find("_quant") != -1:
        dtype = "int8"
        c1 = (shape[1] * shape[4] + 31) // 32
        shape = [shape[0], c1, shape[2], shape[3], 32]
    elif dataflow.find("dequant") != -1:
        dtype = "float16"
    else:
        dtype = in_dtype

    format = "NC1HWC0"
    if dataflow.find("_transdata") != -1:
        format = "NHWC"
        shape = [shape[0], shape[2], shape[3], shape[1] * shape[4]]  # [n, hw, c]

    output = {
        "shape": shape,
        "format": format,
        "dtype": dtype
    }

    return output


def append_tensor_list(tensor_list, x1, quant_scale_0, relu_weight_0, clip_value_0, quant_scale_1,
                       relu_weight_1, clip_value_1, anti_quant_scale, anti_quant_offset):
    param_tensor_list = [x1, quant_scale_0, relu_weight_0, clip_value_0, quant_scale_1,
                         relu_weight_1, clip_value_1, anti_quant_scale, anti_quant_offset]
    for tensor in param_tensor_list:
        if tensor is not None:
            tensor_list.insert(-1, tensor)


def conv_v300_fusion_case(casename,
                          dataflow,
                          conv_type,
                          shape_in,
                          shape_w,
                          pads,
                          strides,
                          groups,
                          bias_flag,
                          quant_scale=0,
                          quant_offset=0,
                          relu_param=0,
                          cout_real=0):
    from impl.fix_pipe import fixpipe_compute
    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci0_dict = {
        "float32": 8,
        "float16": 16,
        "int8": 32,
        "bfloat16": 16
    }
    Ci0 = Ci0_dict[conv_type]
    Ci1 = (Ci + Ci0 - 1) // Ci0

    Co0 = 16
    Co1 = (Co + Co0 - 1) // Co0

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)

    print("====>shape_in_5HD:", shape_in_5HD)
    print("====>shape_w_fracz:", shape_w_fracz)

    shape_scale = (1, Co1, 1, 1, 16)
    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    ho = (Hi + (pads[0] + pads[1]) - Hk) // strides[0] + 1
    wo = (Wi + (pads[2] + pads[3]) - Wk) // strides[1] + 1

    bias_dtype_dict = {
        "float32": "float32",
        "float16": "float32",
        "bfloat16": "float32",
        "int8": "int32"
    }
    bias_dtype = bias_dtype_dict[conv_type]

    with tvm.target.cce():
        fmap = tvm.placeholder(shape_in_5HD, name='fmap', dtype=conv_type)
        weight = tvm.placeholder(shape_w_fracz,
                                 name='weight',
                                 dtype=conv_type,
                                 attrs={
                                     'ori_shape': shape_w,
                                     'ori_format': "NCHW"
                                 })
        bias = tvm.placeholder((Co1 * Co0,), name='bias',
                               dtype=bias_dtype) if bias_flag else None

        conv_res = conv2d_compute(fmap,
                                  weight,
                                  bias,
                                  None,
                                  None,
                                  strides,
                                  pads,
                                  dilations,
                                  offset_x=0)
        shape_param = [1, Co1, 1, 1, Co0]

        x1 = create_eltwise_src(casename, dataflow, conv_res.shape)
        quant_scale_0 = create_quant_scale_0(dataflow, shape_param, quant_scale, quant_offset)
        quant_scale_1 = create_quant_scale_1(dataflow, shape_param, quant_scale, quant_offset)
        relu_weight_0 = create_relu_weight_0(dataflow, shape_param, relu_param)
        relu_weight_1 = create_relu_weight_1(dataflow, shape_param)
        clip_value_0 = create_clip_value_0(dataflow, shape_param, quant_scale, quant_offset)
        clip_value_1 = create_clip_value_1(dataflow, shape_param)
        anti_quant_scale = create_anti_quant_scale(casename, dataflow, shape_param)
        anti_quant_offset = create_anti_quant_offset(casename, dataflow, shape_param)


        shape_out = conv_res.shape
        shape_out_5hd = [shape_out[0], shape_out[1], ho, wo, shape_out[3]]
        output = get_output_dict(dataflow, shape_out_5hd, conv_type)
        print("====>output dict:", output)

        if dataflow == "conv2d":
            out = conv_res
            tensor_list = [fmap, weight, out]
        else:
            eltwise_mode = get_eltwise_mode(dataflow)

            unit_list = get_unit_list(dataflow)

            out = fixpipe_compute(conv_res,
                                  x1,
                                  quant_scale_0,
                                  relu_weight_0,
                                  clip_value_0,
                                  quant_scale_1,
                                  relu_weight_1,
                                  clip_value_1,
                                  anti_quant_scale,
                                  anti_quant_offset,
                                  output,
                                  [],
                                  unit_list,
                                  eltwise_mode)

            tensor_list = [fmap, weight, out]
            append_tensor_list(tensor_list, x1, quant_scale_0, relu_weight_0, clip_value_0,
                               quant_scale_1, relu_weight_1, clip_value_1, anti_quant_scale,
                               anti_quant_offset)

        sch = auto_schedule(out)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": casename,
        "tensor_list": tensor_list
    }

    # kernel_o = "./kernel_meta/" + casename + ".o"
    # if os.path.exists(kernel_o):
    #     os.remove(kernel_o)
    #
    # # tbe.dsl.build(sch, config)
    #
    # if os.path.exists(kernel_o):
    #     print("********** build kernel [{}] success! **********".format(kernel_o))
    # else:
    #     raise RuntimeError("********** build [{}] failed! **********".format(kernel_o))


def conv_v300_single_op_case(casename,
                             conv_type,
                             in_nd2nz_flag,
                             out_nz2nd_flag,
                             shape_in,
                             shape_w,
                             pads,
                             strides,
                             groups,
                             bias_flag,
                             c04_flag=False):
    Ni, Ci, Hi, Wi = shape_in
    Co, w_Ci, Hk, Wk = shape_w

    Co0 = 16
    Co1 = (Co + Co0 - 1) // Co0

    Ci0_dict = {
        "float32": 8,
        "float16": 16,
        "int8": 32,
        "bfloat16": 16
    }
    Ci0 = Ci0_dict[conv_type]
    Ci1 = (Ci + Ci0 - 1) // Ci0

    Ci0_dict = {
        "float32": 8,
        "float16": 16,
        "int8": 32,
        "bfloat16": 16
    }
    Ci0 = Ci0_dict[conv_type]
    Ci1 = (Ci + Ci0 - 1) // Ci0
    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)
    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    bias_dtype_dict = {
        "float32": "float32",
        "float16": "float32",
        "bfloat16": "float32",
        "int8": "int32"
    }
    bias_dtype = bias_dtype_dict[conv_type]
    w_format = "FRACTAL_Z_C04" if c04_flag else "FRACTAL_Z"

    res_dtype_dict = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "int8": "int32"
    }

    # ======================config conv2d parameters======================
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
        "format": w_format,
        "dtype": conv_type,
    }

    bias = {
        "ori_shape": (Co1 * Co0),
        "dtype": bias_dtype
    } if bias_flag else None
    offset_w = None
    outputs = {"dtype": res_dtype_dict[conv_type]}

    # data_format决定了strides和dilations怎么取， 默认"NCHW"

    print("==========conv2d inputs==========", inputs)
    print("==========conv2d weights==========", weights)
    conv2d(inputs,
           weights,
           bias,
           offset_w,
           outputs,
           strides,
           pads,
           dilations,
           groups=groups,
           offset_x=0,
           kernel_name=casename)


def run_testcase(config_dict):
    print("=" * 150)
    print("case {}".format(config_dict))
    print()

    casename, dataflow, conv_type, shape_in, shape_w, pads, strides, dilation, groups, bias_flag, quant_scale, quant_offset, relu_param = config_dict

    if dataflow == "conv2d":
        conv_v300_single_op_case(casename,
                                 conv_type,
                                 False,
                                 False,
                                 shape_in,
                                 shape_w,
                                 pads,
                                 strides,
                                 groups,
                                 bias_flag,
                                 c04_flag=False)
        return

    cout_real = shape_w[0]
    conv_v300_fusion_case(casename, dataflow,
                          conv_type,
                          shape_in,
                          shape_w,
                          pads,
                          strides,
                          groups,
                          bias_flag,
                          quant_scale=quant_scale,
                          quant_offset=quant_offset,
                          relu_param=relu_param,
                          cout_real=cout_real)


def set_impl_mode():
    cube_op_info = operator_info.OpInfo("conv2d", "Conv2D")
    cube_op_info.precision_mode = "high_performance"
    op_context.get_context().add_op_info(cube_op_info)


def run_v300_case(case_config, is_hf32_flag=False):
    with op_context.OpContext():
        if is_hf32_flag:
            set_impl_mode()
        run_testcase(case_config)


def run_v300_batch_cases(case_list, is_hf32_flag=False):
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend320", "AiCore")
    with op_context.OpContext():
        for case in case_list:
            if is_hf32_flag:
                set_impl_mode()
            run_testcase(case)


def check_FixpipeBase():
    from impl.fixpipe_op.fixpipe_base import FixpipeBase
    from impl.fixpipe_op.fixpipe_util import create_placeholder
    from impl.fix_pipe import fixpipe_compute

    input_dict = {
        "shape": [1, 1, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "float32",
        "ori_shape": [1, 1, 112, 112, 16]
    }

    output = {
        "shape": [1, 1, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 1, 112, 112, 16]
    }

    relu_vector_dict = {
        "shape": [1, 1, 1, 1, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 1, 1, 1, 16]
    }

    relu_scalar_dict = {
        "shape": [1],
        "format": "ND",
        "dtype": "float16",
        "ori_shape": [1]
    }

    anto_quant_scale_dict = relu_scalar_dict
    anto_quant_offset_dict = relu_scalar_dict

    x2_dict = {
        "shape": [1, 1, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 1, 112, 112, 16]
    }

    x2_int8_dict = {
        "shape": [1, 1, 112, 112, 32],
        "format": "NC1HWC0",
        "dtype": "int8",
        "ori_shape": [1, 1, 112, 112, 32]
    }

    try:
        input = create_placeholder(input_dict, "in")
        relu = create_placeholder(relu_vector_dict, "relu")
        fixpipe = FixpipeBase("test", input, None, None, relu, None, None, None, None, None, None, output, [], [], "None")
        fixpipe._get_pre_activation()
        fixpipe.fixpipe_compute()

        x2 = create_placeholder(x2_dict, "eltwise_src")
        fixpipe = FixpipeBase("test", input, x2, None, relu, None, None, relu, None, None, None, output, [], [], "ADD")
        fixpipe._get_pre_activation()
        fixpipe._get_post_activation()

        x2 = create_placeholder(x2_int8_dict, "eltwise_src")
        fixpipe = FixpipeBase("test", input, x2, None, relu, None, None, relu, None, None, None, output, [], [], "ADD")
        fixpipe._get_post_anti_quant()

        relu = create_placeholder(relu_scalar_dict, "relu")
        fixpipe = FixpipeBase("test", input, None, None, relu, None, None, relu, None, None, None, output, [], [], "None")
        fixpipe._get_pre_activation()
        fixpipe._get_post_activation()

    except:
        print("===============>catch error")

    x1_dict = {
        "shape": [1, 2, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "int32",
        "ori_shape": [1, 2, 112, 112, 16]
    }

    x2_dict = {
        "shape": [1, 2, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "int8",
        "ori_shape": [1, 2, 112, 112, 16]
    }

    scalar_dict = {
        "shape": [1],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1],
        "const_value": [1],
    }

    offset_dict = {
        "shape": [1],
        "format": "NC1HWC0",
        "dtype": "int8",
        "ori_shape": [1],
        "const_value": [1],
    }

    output = {
        "shape": [1, 2, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 2, 112, 112, 16]
    }

    x1 = create_placeholder(x1_dict, "x1")
    x2 = create_placeholder(x2_dict, "x2")
    anti_quant = create_placeholder(scalar_dict, "scalar")
    offset = create_placeholder(offset_dict, "offset")

    fixpipe = FixpipeBase("test", x1, x2, None, None, None, None, None, None, anti_quant, offset, output, [], [], "ADD")
    fixpipe.fixpipe_compute()


def check_fix_pipe():
    from impl.fix_pipe import fix_pipe
    from impl.fixpipe_op.fixpipe_util import is_scaler_input
    from impl.fixpipe_op.fixpipe_util import is_vector_input
    from impl.fixpipe_op.fixpipe_util import get_input_scalar_value
    from impl.fixpipe_op.fixpipe_util import get_op_type
    from impl.fixpipe_op.fixpipe_util import create_placeholder

    input_dict = {
        "shape": [1, 1, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 1, 112, 112, 16]
    }
    fix_pipe(input_dict, None, None, None, None, None, None, None, None, None, {}, [], [], "None")
    flag = is_scaler_input(input_dict)

    scalar_dict = {
        "shape": [1],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1],
        "const_value": [0],
    }
    value = get_input_scalar_value(scalar_dict)

    tensor = create_placeholder(scalar_dict, "TEST")
    op_type = get_op_type(tensor)

    try:
        scalar_dict = {
            "shape": [1],
            "format": "NC1HWC0",
            "dtype": "float16",
            "ori_shape": [1],
        }
        tensor = create_placeholder(scalar_dict, "TEST")
    except:
        print("=======>check _create_placeholder no const_value")

    try:
        scalar_dict = {
            "shape": [1],
            "format": "NC1HWC0",
            "dtype": "float16",
            "const_value": [0],
        }
        tensor = create_placeholder(scalar_dict, "TEST")
    except:
        print("=======>check _create_placeholder no ori_shape")

    scalar_input = {
        "shape": [1],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": (),
        "const_value": [0],
    }

    flag = is_scaler_input(scalar_input)
    assert flag == True

    tensor = create_placeholder(scalar_input, "TEST")
    flag = is_scaler_input(tensor)
    assert flag == True

    flag = is_vector_input(tensor)
    assert flag == False

    scalar_input = {
        "shape": [1,1,1,16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": (),
        "const_value": [0],
    }
    try:
        flag = is_scaler_input(scalar_input)
    except:
        print("catch error! [shape should be 1 when ori_shape is empty]")


def check_fixpipe_func_name():
    from impl.fixpipe_op.fixpipe_util import DTYPE_TRANS_MAP
    import impl.fixpipe_op.fixpipe_util as fixpipe_util
    from impl.fixpipe_op.fixpipe_util import ANTI_QUANT_MAP
    from impl.fixpipe_op.fixpipe_util import QUANT_SCALE_0_STR
    from impl.fixpipe_op.fixpipe_util import QUANT_SCALE_1_STR
    from impl.fixpipe_op.fixpipe_util import RELU_WEIGHT_0_STR
    from impl.fixpipe_op.fixpipe_util import RELU_WEIGHT_1_STR
    from impl.fixpipe_op.fixpipe_util import ELTWISE_SRC_STR
    from impl.fixpipe_op.fixpipe_util import FIXPIPE_OP_TAG
    from impl.fixpipe_op.fixpipe_util import FIXPIPE_REFORM_TAG
    from impl.fixpipe_op.fixpipe_util import PASS_PRE_CONVERT_MODE
    from impl.fixpipe_op.fixpipe_util import PRE_CONVERT_MODE
    from impl.fixpipe_op.fixpipe_util import POST_QUANT_MODE
    from impl.fixpipe_op.fixpipe_util import FIXPIPE_VECTOR_TENSOR_LIST
    from impl.fixpipe_op.fixpipe_util import NC1HWC0_C1_IDX
    from impl.fixpipe_op.fixpipe_util import NC1HWC0_C0_IDX
    from impl.fixpipe_op.fixpipe_util import DTYPE_FLOAT32
    from impl.fixpipe_op.fixpipe_util import DTYPE_FLOAT16
    from impl.fixpipe_op.fixpipe_util import DTYPE_INT32
    from impl.fixpipe_op.fixpipe_util import VECTOR_RELU_MODE
    from impl.fixpipe_op.fixpipe_util import SCALAR_RELU_MODE
    from impl.fixpipe_op.fixpipe_util import NORMAL_RELU_MODE
    from impl.fixpipe_op.fixpipe_util import PRE_ACT_UNIT_STR
    from impl.fixpipe_op.fixpipe_util import POST_ACT_UNIT_STR
    from impl.fixpipe_op.fixpipe_util import get_op_type
    from impl.fixpipe_op.fixpipe_util import get_op_info_from_attrs
    from impl.fixpipe_op.fixpipe_util import calc_shape_total_dim
    from impl.fixpipe_op.fixpipe_util import is_scaler_input
    from impl.fixpipe_op.fixpipe_util import is_vector_input
    from impl.fixpipe_op.fixpipe_util import get_input_scalar_value
    from impl.fixpipe_op.fixpipe_util import check_fixpipe_support
    from impl.fixpipe_op.fixpipe_util import FIXPIPE_SCOPE_MAP

    print(fixpipe_util.ANTI_QUANT_MAP)
    print(fixpipe_util.QUANT_SCALE_0_STR)
    print(fixpipe_util.QUANT_SCALE_1_STR)
    print(DTYPE_TRANS_MAP)

    input_dict = {
        "shape": [1, 1, 112, 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 1, 112, 112, 16]
    }

    from impl.fixpipe_op.fixpipe_conv2d import NC1MC0_C0_IDX
    from impl.fixpipe_op.fixpipe_conv2d import NC1MC0_C0_IDX
    from impl.fixpipe_op.fixpipe_conv2d import FixpipeConv2d
    from impl.fixpipe_op.fixpipe_factory import FixpipeFactory
    from impl.fixpipe_op.fixpipe_factory import FIXPIPE_OP_SUPPORT_MAP

    fixpip = FixpipeConv2d("conv2d", input_dict, None, None, None, None, None, None, None, None, None, {}, [], [], "None")

    try:
        fixpip = FixpipeFactory.get_fixpipe("error", input_dict, None, None, None, None, None, None, None, None, None, {}, [], [], "None")
    except:
        print("======> check error op_type")


def check_fixpipe_conv2d():
    from impl.fixpipe_op import fixpipe_conv2d
    print (fixpipe_conv2d.C0_16)
    print (fixpipe_conv2d.C0_32)
    print (fixpipe_conv2d.C0_64)

    def _create_placeholder(input_dict, name):
        if "ori_shape" not in input_dict:
            raise RuntimeError("ori_shape not in dict")

        attrs = {}
        if "const_valule" in input_dict:
            attrs["const_value"] = input_dict.get("const_value")

        attrs["ori_shape"] = input_dict.get("ori_shape")
        return tvm.placeholder(input_dict.get("shape"), input_dict.get("dtype"), name=name, attrs=attrs)

    x1_dict = {
        "shape": [1, 2, 112 * 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 2, 112 * 112, 16]
    }

    x2_dict = {
        "shape": [1, 1, 112 * 112, 32],
        "format": "NC1HWC0",
        "dtype": "int8",
        "ori_shape": [1, 1, 112 * 112, 32]
    }

    scalar_dict = {
        "shape": [1],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1],
        "const_value": [1],
    }

    offset_dict = {
        "shape": [1],
        "format": "NC1HWC0",
        "dtype": "int8",
        "ori_shape": [1],
        "const_value": [1],
    }

    output = {
        "shape": [1, 2, 112 * 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 2, 112 * 112, 16]
    }

    x1 = _create_placeholder(x1_dict, "x1")
    x2 = _create_placeholder(x2_dict, "x2")
    anti_quant = _create_placeholder(scalar_dict, "scalar")
    offset = _create_placeholder(offset_dict, "offset")

    # anti_quant
    fixpipe_obj = fixpipe_conv2d.FixpipeConv2d("conv2d",
                                               x1, x2, None, None, None, None, None, None,
                                               anti_quant, offset, output, [], [], "ADD")
    tvm.compute(x2_dict.get("shape"),
                lambda *indices:
                fixpipe_obj._x2_reform_generate_func(x2, x2_dict.get("shape"))(*indices),
                name="test")

    # default
    fixpipe_obj = fixpipe_conv2d.FixpipeConv2d("conv2d",
                                               x1, x2, None, None, None, None, None, None,
                                               None, None, output, [], [], "ADD")
    tvm.compute(x2_dict.get("shape"),
                lambda *indices:
                fixpipe_obj._x2_reform_generate_func(x2, x2_dict.get("shape"))(*indices),
                name="test")

    # anti_quant c0 error
    x2_dict = {
        "shape": [1, 1, 112 * 112, 16],
        "format": "NC1HWC0",
        "dtype": "float16",
        "ori_shape": [1, 1, 112 * 112, 16]
    }
    x2 = _create_placeholder(x2_dict, "x2")

    try:
        fixpipe_obj = fixpipe_conv2d.FixpipeConv2d("conv2d",
                                                   x1, x2, None, None, None, None, None, None,
                                                   anti_quant, offset, output, [], [], "ADD")
        tvm.compute(x2_dict.get("shape"),
                    lambda *indices:
                    fixpipe_obj._x2_reform_generate_func(x2, x2_dict.get("shape"))(*indices),
                    name="test")
    except:
        print("====>catch c0 error")


def test_aipp_conv2d(config_dict):
    """
    c04 + aipp
    """
    import json
    from impl.aipp import aipp_compute

    casename, dataflow, conv_type, shape_in, shape_w, pads, strides, dilation, groups, bias_flag, quant_scale, quant_offset, relu_param = config_dict
    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci0 = 4
    Co0 = 16
    Ci1 = 1
    Co1 = (Co + Co0 - 1) // Co0

    shape_w_fracz = ((Hk * Wk * Ci1 + Ci0 - 1) // Ci0, Co1, Co0, Ci0 * 4)
    weight_format = "FRACTAL_Z_C04"

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    aipp_format_dict = {
        "yuv": "YUV420SP_U8",
        "xrgb": "XRGB8888_U8",
        "rgb": "RGB888_U8"
    }
    aipp_format = "rgb"
    aipp_input_format = aipp_format_dict[aipp_format]

    h_after_crop = Hi
    w_after_crop = Wi
    output_data = {
        'shape': [Ni, 1, h_after_crop, w_after_crop, 4],
        'ori_shape': [Ni, 3, h_after_crop, w_after_crop],
        'format': 'NC1HWC0_C04',
        'ori_format': 'NCHW',
        'dtype': 'float16',
        'addr_type': 0,
        'valid_shape': (),
        'slice_offset': (),
        'use_L1_workspace': 0,
        'L1_workspace_size': -1,
        'L1_fusion_type': -1,
        'L1_addr_offset': 0,
        'total_shape': (),
        'split_index': 0
    }

    aipp_config_dict = {
        "cce_product": "2.1",
        "out_dtype": "float16",
        "out_format": "NC1HWC0",
        "aipp_mode": "static",
        "related_input_rank": 0,
        "input_format": aipp_input_format,
        "src_image_size_n": shape_in[0],
        "src_image_size_c": shape_in[1],
        "src_image_size_h": shape_in[2],
        "src_image_size_w": shape_in[3],
        "crop": False,
        "load_start_pos_h": 2,
        "load_start_pos_w": 2,
        "crop_size_h": h_after_crop,
        "crop_size_w": w_after_crop,
        "src_image_h": shape_in[2],
        "src_image_w": shape_in[3],
        "resize": 0,
        "resize_model": 0,
        "resize_output_h": 32,
        "resize_output_w": 32,
        "padding": 0,
        "left_padding_size": 7,
        "right_padding_size": 7,
        "top_padding_size": 0,
        "bottom_padding_size": 4,
        "csc_switch": 1,
        "rbuv_swap_switch": 0,
        "ax_swap_switch": 0,
        "matrix_r0c0": 298,
        "matrix_r0c1": 516,
        "matrix_r0c2": 0,
        "matrix_r1c0": 298,
        "matrix_r1c1": -100,
        "matrix_r1c2": -208,
        "matrix_r2c0": 298,
        "matrix_r2c1": 0,
        "matrix_r2c2": 409,
        "input_bias_0": 16,
        "input_bias_1": 128,
        "input_bias_2": 128,
        "mean_chn_0": 1,
        "mean_chn_1": 1,
        "mean_chn_2": 1,
        "mean_chn_3": 1,
        "var_reci_chn_0": 1.0,
        "var_reci_chn_1": 1.0,
        "var_reci_chn_2": 1.0,
        "min_chn_0": 0,
        "min_chn_1": 0,
        "min_chn_2": 0,
        "min_chn_3": 0
    }

    with tvm.target.cce():
        fmap = tvm.placeholder(shape_in,
                               name="params_0",
                               dtype="uint8",
                               attrs={
                                   "ori_shape": shape_in,
                                   "format": "NCHW",
                                   "ori_format": "NHWC"
                               })
        aipp_config_dict_json = json.dumps(aipp_config_dict)
        aipp_res = aipp_compute(fmap,
                                None,
                                output_data,
                                aipp_config_dict_json,
                                kernel_name="aipp")
        aipp_res.op.attrs["is_first_layer"] = True
        weight = tvm.placeholder(shape_w_fracz,
                                 name='weight',
                                 dtype=conv_type,
                                 attrs={
                                     'ori_shape': shape_w,
                                     'ori_format': "NCHW",
                                     'format': weight_format
                                 })
        conv_res = conv2d_compute(aipp_res,
                                  weight,
                                  None,
                                  None,
                                  None,
                                  strides,
                                  pads,
                                  dilations,
                                  offset_x=0)
        sch = auto_schedule(conv_res)


def run_v300_batch_cases_aipp(case_list, is_hf32_flag=False):
    from te.platform.cce_conf import te_set_version
    from tbe.common.platform.platform_info import get_soc_spec
    with op_context.OpContext():
        te_set_version("Ascend320", "AiCore")
        soc_version = get_soc_spec("SOC_VERSION")
        for case in case_list:
            if is_hf32_flag:
                set_impl_mode()
            test_aipp_conv2d(case)


def run_v300_cases(test_arg):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=side_effects)):
            check_fixpipe_conv2d()
            check_fixpipe_func_name()
            check_FixpipeBase()
            check_fix_pipe()
            run_v300_batch_cases(v300_case)
            #run_v300_batch_cases(single_conv2d)


def run_v300_aipp_fusion_cases(test_arg):
    run_v300_batch_cases_aipp(aipp_conv2d)


def run_v220_cases(test_arg):
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects_220)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=side_effects_220)):
            run_v300_batch_cases(single_conv2d)

print("====> conv2d v300 ut start")
ut_case.add_cust_test_func(test_func=run_v300_cases)
ut_case.add_cust_test_func(test_func=run_v300_aipp_fusion_cases)
print("====> end v300 ut start")






