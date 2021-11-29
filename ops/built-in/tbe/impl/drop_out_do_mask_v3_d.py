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
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util.platform_adapter import para_check

MATMUL_BATCH_SIZE = 0
BATCH_MATMUL_BATCH_SIZE1 = 1
BATCH_MATMUL_BATCH_SIZE2 = 2
BATCH_MATMUL_BATCH_SIZE3 = 3
BATCH_MATMUL_BATCH_SIZE4 = 4

SHAPE_SIZE_LIMIT = 1 << 30


def reshape_input_mask(input_tensor, input_mask, kernel_name):
    """
    Reshape mask shape ND to matmul shape FRACTAL_NZ,
    e.g. [batch1, batch2, K//16, M//16, 16, 16] -> [batch, K//16, M//16, 16, 16].

    Params
    ----------------
    input_tensor: matmul, tvm.tensor, fp16/fp32
    input_mask: dropout_gen_mask, tvm.tensor, uint8
    kernel_name: str

    Returns
    ----------------
    input_mask: reshaped mask
    """
    matmul_flag = "matmul" in input_tensor.op.tag \
        and input_tensor.op.attrs["format"] == "FRACTAL_NZ"
    matmul_shape = shape_util.shape_to_list(input_tensor.shape)
    mask_shape = shape_util.shape_to_list(input_mask.shape)
    batch_shape = mask_shape[:-4]

    if matmul_flag:
        lambda_expression = None
        if len(batch_shape) == MATMUL_BATCH_SIZE:
            lambda_expression = lambda *indices: input_mask(*indices)
        elif len(batch_shape) == BATCH_MATMUL_BATCH_SIZE1:
            lambda_expression = lambda *indices: input_mask(*indices)
        elif len(batch_shape) == BATCH_MATMUL_BATCH_SIZE2:
            lambda_expression = lambda *indices: input_mask(
                indices[0] // batch_shape[-1],
                indices[0] % batch_shape[-1],
                indices[-4],
                indices[-3],
                indices[-2],
                indices[-1]
            )
        elif len(batch_shape) == BATCH_MATMUL_BATCH_SIZE3:
            lambda_expression = lambda *indices: input_mask(
                indices[0] // batch_shape[-1] // batch_shape[-2],
                indices[0] // batch_shape[-1] % batch_shape[-2],
                indices[0] % batch_shape[-1],
                indices[-4],
                indices[-3],
                indices[-2],
                indices[-1]
            )
        elif len(batch_shape) == BATCH_MATMUL_BATCH_SIZE4:
            lambda_expression = lambda *indices: input_mask(
                indices[0] // batch_shape[-1] // batch_shape[-2] // batch_shape[-3],
                indices[0] // batch_shape[-1] // batch_shape[-2] % batch_shape[-3],
                indices[0] // batch_shape[-1] % batch_shape[-2],
                indices[0] % batch_shape[-1],
                indices[-4],
                indices[-3],
                indices[-2],
                indices[-1]
            )
        else:
            error_detail = ("Only support to adjust batch shape [2, 3, 4], " +
                "but the recent batch shape is [%d]." % (len(batch_shape)))
            error_manager_vector.raise_err_input_shape_invalid(
                kernel_name, "input_mask", error_detail
            )

        if lambda_expression:
            input_mask = tvm.compute(
                matmul_shape,
                lambda_expression,
                name="dropout_reshape",
                tag="dropout_broadcast"
            )

    return input_mask, batch_shape


@fusion_manager.register("drop_out_do_mask_v3_d")
def drop_out_do_mask_v3_d_compute(input_tensor: tvm.tensor.Tensor,
                                  input_mask: tvm.tensor.Tensor,
                                  output,
                                  input_keep_prob: float,
                                  kernel_name="drop_out_do_mask_v3_d"):
    """
    dropoutdomaskv3d compute
    """
    input_mask, batch_shape = reshape_input_mask(input_tensor, input_mask, kernel_name)
    input_dtype = input_tensor.dtype
    input_mask = te.lang.cce.cast_to(input_mask, input_dtype)
    rec_keep_prob = 1 / input_keep_prob
    mul_input_mask = te.lang.cce.vmul(input_tensor, input_mask)
    output = te.lang.cce.vmuls(mul_input_mask,
                               tvm.const(rec_keep_prob, input_dtype))
    if batch_shape:
        output.op.attrs["batch_shape"] = batch_shape
    return output


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    for var, name in zip(dict_list, name_list):
        shape = var.get("shape")
        shape = shape_util.scalar2tensor_one(shape)
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


@para_check.check_input_type(dict, dict, dict, float, str)
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
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(
        input_tensor.get('dtype').lower(), ("float16", "float32"))
    para_check.check_dtype_rule(
        input_mask.get('dtype').lower(), ("uint8"))
    para_check.check_shape_rule(input_tensor.get('shape'),
                          max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(input_mask.get('shape'),
                          max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_tensor.get('shape'), SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_mask.get('shape'), SHAPE_SIZE_LIMIT)
    input_name_list = ['input_tensor', 'input_mask']
    list_placeholder = _get_placeholder([input_tensor, input_mask],
                                         input_name_list)
    input_tensor = list_placeholder[0]
    input_mask = list_placeholder[1]

    output = drop_out_do_mask_v3_d_compute(input_tensor, input_mask,
                                           output, input_keep_prob)

    build_list = [input_tensor, input_mask, output]
    config = {"name": kernel_name, "tensor_list": build_list}

    with tvm.target.cce():
        sch = te.lang.cce.auto_schedule(output)

    config = {"name": kernel_name,
              "tensor_list": build_list}
    te.lang.cce.cce_build_code(sch, config)
    fusion_manager.set_current_op_pattern("DropOutDoMaskV3D")
