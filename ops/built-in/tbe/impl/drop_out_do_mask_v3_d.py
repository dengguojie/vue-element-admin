#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
drop_out_do_mask_v3_d
"""
import operator
from functools import reduce as  functools_reduce

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi.cce import util

SHAPE_SIZE_LIMIT = 1 << 30


@fusion_manager.register("drop_out_do_mask_v3_d")
def drop_out_do_mask_v3_d_compute(input_tensor: tvm.tensor.Tensor,
                                  input_mask: tvm.tensor.Tensor,
                                  input_keep_prob: float,
                                  output,
                                  kernel_name="drop_out_do_mask_v3_d"):
    input_dtype = input_tensor.dtype
    input_mask = te.lang.cce.cast_to(input_mask, input_dtype)
    rec_keep_prob = 1 / input_keep_prob
    mul_input_mask = te.lang.cce.vmul(input_tensor, input_mask)
    output = te.lang.cce.vmuls(mul_input_mask,
                               tvm.const(rec_keep_prob, input_dtype))
    return output


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    for var, name in zip(dict_list, name_list):
        shape = var.get("shape")
        shape = util.scalar2tensor_one(shape)
        shape_refine = (functools_reduce(operator.mul, shape),)
        dtype = var.get("dtype").lower()
        if name == "input_tensor":
            input_shape = list(shape_refine)
        if input_shape != list(shape_refine):
            raise RuntimeError(
                "the shape of input_tensor and input_mask must be equal !")
        list_placeholder.append(
            tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder


@util.check_input_type(dict, dict, dict, float, str)
def drop_out_do_mask_v3_d(input_tensor, input_mask, output, input_keep_prob,
                          kernel_name="drop_out_do_mask_v3_d"):
    """
    algorithm: tf_drop_out_do_mask_v3_d
    scale_x = x*(1 / keep_prob)
    res = select(mask == 1, scale_x, 0)

    Parameters
    ----------
    input_tensor : dict,shape and dtype of input_tensor,only support float16 and float32
    input_mask : dict,shape and dtype of input_mask
        shape of mask,1D, dtype == uint8
        length=(size(shape_tensor)+tbe_platform.ELEMENTS_VECTOR_OP_FP16
        -1)/tbe_platform.ELEMENTS_VECTOR_OP_FP16*tbe_platform.ELEMENTS_VECTOR_OP_FP16
        eg. shape_tensor=[2,5,8] shape_mask=[16] shape_res=[2,5,8]
        shape_tensor=[15,17,19] shape_mask=[608] shape_res=[15,17,19]
    input_keep_prob : dict,shape and dtype of input_keep_prob
        shape of keep_prob, only 1 parament and equals to (1)
        prob scale (0.0,1.0] NOTICE: type same as dytpe
    output : dict,shape and dtype of output
    kernel_name : str
        cce kernel name, default value is "drop_out_do_mask_v3_d"

    Returns
    -------
    None
    """
    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(
        input_tensor.get('dtype').lower(), ("float16", "float32"))
    util.check_dtype_rule(
        input_mask.get('dtype').lower(), ("uint8"))
    util.check_shape_rule(input_tensor.get('shape'),
                          max_shape_num=SHAPE_SIZE_LIMIT)
    util.check_shape_rule(input_mask.get('shape'),
                          max_shape_num=SHAPE_SIZE_LIMIT)
    util.check_shape_size(input_tensor.get('shape'), SHAPE_SIZE_LIMIT)
    util.check_shape_size(input_mask.get('shape'), SHAPE_SIZE_LIMIT)
    input_name_list = ['input_tensor', 'input_mask']
    input_tensor, input_mask = _get_placeholder([input_tensor, input_mask],
                                                input_name_list)
    output = drop_out_do_mask_v3_d_compute(input_tensor, input_mask,
                                           input_keep_prob, output)

    build_list = [input_tensor, input_mask, output]
    config = {"name": kernel_name, "tensor_list": build_list}

    with tvm.target.cce():
        sch = te.lang.cce.auto_schedule(output)

    config = {"name": kernel_name,
              "tensor_list": build_list}
    te.lang.cce.cce_build_code(sch, config)
