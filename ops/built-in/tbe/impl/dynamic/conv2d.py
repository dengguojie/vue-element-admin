# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
dynamic conv2d
"""
from __future__ import absolute_import

import math
from te import tvm
import te.lang.cce as tbe
import te.lang.dynamic as dynamic
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils.error_manager import error_manager_cube as err_man
from te.utils.error_manager import error_manager_util
from impl.util import fusion_util
from impl.util import util_conv2d
from impl.util.util_cube_dynamic import Conv2dParaProcess

NONETYPE = type(None)
H_DIM = 2
W_DIM = 3


def set_default_para():
    """
    set default parameter value
    """
    default_para = {}
    default_para["res_dtype"] = "float16"
    default_para["optim_dict"] = {"c0_optim_flg": False}
    default_para["fusion_para"] = {"input_memory_type": 0, "output_memory_type": 0,
                                   "valid_shape": (), "slice_offset": (),
                                   "l1_fusion_type": -1}
    default_para["ori_shape"] = [0, 0, 0, 0]
    return default_para


@tbe_base.register_fusion_compute("Conv2D")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list),
                             int, str, int, str, str)
def conv2d_fusion_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                          groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                          dsl_flag=True):
    fusion_util.check_fusion_input([inputs])
    fusion_util.check_fusion_input([weights])

    # set fusion build config
    build_cfg = tbe_platform.get_fusion_build_cfg()
    build_cfg['constant_realize_extent_in_infer_bound'] = False

    return _conv2d_compute(
        inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
        groups, data_format, offset_x, kernel_name, dsl_flag)


def _conv2d_compute(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                    groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d",
                    dsl_flag=True):

    """
    conv2d compute

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset for fmap

    Returns
    -------
    tvm compute
    """

    default_para = set_default_para()
    if not outputs.get("ori_shape"):
        outputs["ori_shape"] = default_para["ori_shape"]
    ori_paras = {
        "inputs": inputs, "weights": weights, "bias": bias, "offset_w": offset_w,
        "outputs": outputs, "strides": strides, "pads": pads, "dilations": dilations,
        "groups": groups, "data_format": data_format, "offset_x": offset_x,
        "kernel_name": kernel_name, "optim_dict": default_para.get("optim_dict"),
    }

    conv_para = Conv2dParaProcess(ori_paras)
    paras = conv_para.config_paras()

    pad_t, pad_b, pad_l, pad_r = conv_para.pads
    op_res = tbe.conv(paras.get("input_tensor"), paras.get("weight_tensor"),
                      {"bias_tensor": paras.get("bias_tensor"),
                       "offset_w_tensor": offset_w,
                       "pad_h": [pad_t, pad_b], "pad_w": [pad_l, pad_r],
                       "stride_h": conv_para.strides[H_DIM], "stride_w": conv_para.strides[W_DIM],
                       "dilate_h": conv_para.dilations[H_DIM], "dilate_w": conv_para.dilations[W_DIM],
                       "filter_h": paras.get("w_shape")[H_DIM],
                       "filter_w": paras.get("w_shape")[W_DIM],
                       "offset_x": offset_x,
                       "res_dtype": default_para.get("res_dtype"),
                       "fusion_para": default_para.get("fusion_para"),
                       "kernel_name": kernel_name,
                       "group": conv_para.groups,
                       "enlarge": paras.get("group_para").get("enlarge"),
                       "c1_opt": paras.get("group_para").get("c1_opt"),
                       "cout1_opt": paras.get("group_para").get("cout1_opt"),
                       "group_opt": paras.get("group_para").get("group_opt"),
                       "a_shape": paras.get("in_shape_nc1hwc0"),
                       "weight_fracz_shape": paras.get("w_shape_frac_z"),
                       "weight_ori_shape_nchw": paras.get("w_shape")},
                      optim_dict=default_para.get("optim_dict"),
                      dsl_flag=dsl_flag)

    if conv_para.bias is not None:
        return {"op_placeholder": [paras.get("input_tensor"), paras.get("weight_tensor"), paras.get("bias_tensor")],
                "op_res": [op_res]}
    return {"op_placeholder": [paras.get("input_tensor"), paras.get("weight_tensor")], "op_res": [op_res]}


@tbe_base.register_operator("Conv2D")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list),
                             int, str, int, str, str)
def conv2d(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
           groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d"):
    """
    algorithm: conv2d

    Notice
    ------
    only used by framework combine with IR

    Parameters
    ----------
    inputs: dict with keys(shape and dtype and range)
        input 4d feature map tensor
    weights: dict with keys(shape and dtype)
        input 4d weight tensor
    bias: dict with keys(shape and dtype) or None
        input bias tensor
    offset_w: keys(shape and dtype) or None
        input offset_w tensor
    outputs: dict with keys(shape and dtype)
        output tensor, dtype must be assigned
    strides: tuple/list of 4 integers
        stride on H/W, format sensitive
    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right]
    dilations: tuple/list of 4 integers
        dilation on H/W, format sensitive
    groups: int
        param for group covolution
    data_format: string
        input data format
    offset_x: int
        offset of fmap
    kernel_name: str
        kernel name, default value is "conv2d"

    Returns
    -------
    None
    """

    with tbe_base.compute():
        res = _conv2d_compute(
            inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
            groups, data_format, offset_x, kernel_name, dsl_flag=False)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get("op_res"))

    tensor_list = res.get("op_placeholder") + res.get("op_res")
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }

    tbe.build(sch, config)
