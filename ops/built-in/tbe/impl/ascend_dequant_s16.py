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
ascend_dequant_s16
"""
from functools import reduce as function_reduce

from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl import ascend_quant_util as util
from te.utils import shape_util


# pylint: disable=invalid-name,unused-argument,unnecessary-lambda,too-many-arguments,too-many-locals
@fusion_manager.register("ascend_dequant_s16")
def ascend_dequant_s16_compute(x0, deq_scale, x1, y, relu_flag=False, kernel_name="ascend_dequant_s16"):
    """
    int32 -> int16

    Parameters:
    ----------
    x0 : the placeholder of input
    deq_scale: the placeholder of deq_scale
    x1: the placeholder of add input tensor
    y: the dict of output
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_dequant_s16"

    Returns:
    -------
    res : the result of ascend_dequant_s16
    """
    x0_shape = x0.shape

    conv_flag = 0
    if len(x0.op.input_tensors) and x0.op.input_tensors[0].name in ('mad1', 'convolution_c_col_bias'):
        conv_flag = 1

    if conv_flag:
        align_shape = shape_util.shape_to_list(x0.op.input_tensors[0].shape)
        c1_index = 2
    else:
        x0_shape_list = shape_util.shape_to_list(x0_shape)
        align_shape = x0_shape_list.copy()
        c1_index = 1

    ori_shape_deq = deq_scale.op.attrs["ori_shape"]
    ori_shape_deq_list = shape_util.shape_to_list(ori_shape_deq)
    deq_dim = function_reduce(lambda x, y: x * y, ori_shape_deq_list[:])
    tensor_flag = False
    if deq_dim > 1:
        tensor_flag = True

    if util.is_nz_format(x0):
        c1_index = len(x0_shape) - 4

    align_shape[-2] = (align_shape[-2] + 15) // 16 * 16
    res_ub = _s32_to_s16_normal_compute(x0, deq_scale, x1, align_shape, c1_index, tensor_flag, relu_flag, conv_flag)

    if util.is_nz_format(x0):
        res = tvm.compute(align_shape, lambda *i: res_ub[i], name="res", tag="dequant_s16_NZ")
        return res

    res_shape = shape_util.shape_to_list(res_ub.shape)
    res_shape[-2] = x0.shape[-2]

    if conv_flag:
        group = align_shape[0]
        cout1_opt = align_shape[2]
        if group == 1:
            res = tvm.compute(
                x0_shape,
                lambda batch, cout1, howo, cout0: res_ub(0, batch, cout1, howo, cout0), name="dequant_s16_remove_pad",
                tag="dequant_s16_remove_pad")
        else:
            res = tvm.compute(x0_shape,
                lambda batch, cout1, howo, cout0:
                    res_ub(cout1 // cout1_opt, batch, cout1 % cout1_opt, howo, cout0), name="dequant_s16_remove_pad",
                tag="dequant_s16_remove_pad")
    else:
        res = tvm.compute(res_shape, lambda *indice: res_ub(*indice), name="dequant_s16_remove_pad",
                          tag="dequant_s16_remove_pad")
    return res


def _s32_to_s16_normal_compute(x0, deq_scale, x1, align_shape, c1_index, tensor_flag, relu_flag, conv_flag):
    """
    generate s32_to_s16 compute
    """
    if tensor_flag:
        res_ub = tvm.compute(align_shape,
            _deq_cast_compute(x0, deq_scale, x1, align_shape, c1_index, tensor_flag, relu_flag, conv_flag),
            name="s32_to_s16", tag="dequant_s16_vector")
    else:
        res_ub = tvm.compute(align_shape,
            _deq_cast_compute(x0, deq_scale, x1, align_shape, c1_index, tensor_flag, relu_flag, conv_flag),
            name="s32_to_s16", tag="dequant_s16_scale")

    return res_ub


def _deq_cast_compute(x0, deq_scale, x1, align_shape, c1_index, tensor_flag, relu_flag, conv_flag):
    """
    generate lambda func
    """
    if conv_flag:
        group = align_shape[0]
        x0_shape = x0.shape

        def lambda_func(g, i, j, k, l):
            deq_indice = [0] * 5
            if tensor_flag:
                deq_indice[4] = l
                deq_indice[1] = j

            if x1 is not None:
                x1_indice = [0] * len(x1.shape)
                if len(x1.shape) == 5:
                    x1_indice[4] = l
                    x1_indice[1] = j
                else:
                    x1_indice[0] = j * 16 + l

                if tensor_flag:
                    func = tvm.vdeq_cast(x0(g*(x0_shape[0]//group)+i, j, k, l),
                                         deq_scale(*deq_indice), "int16", do_relu=relu_flag) + x1(*x1_indice)
                else:
                    func = tvm.deq_cast(x0(g*(x0_shape[0]//group)+i, j, k, l),
                                        deq_scale(*deq_indice), "int16") + x1(*x1_indice)
            else:
                if tensor_flag:
                    func = tvm.vdeq_cast(x0(g*(x0_shape[0]//group)+i, j, k, l),
                                         deq_scale(*deq_indice), "int16", do_relu=relu_flag)
                else:
                    func = tvm.deq_cast(x0(g*(x0_shape[0]//group)+i, j, k, l), deq_scale(*deq_indice), "int16")
            return func
    else:
        n_dim = len(align_shape)
        c0_index = n_dim - 1

        def lambda_func(*indice):
            deq_indice = [0] * 5
            if tensor_flag:
                deq_indice[4] = indice[c0_index]
                deq_indice[1] = indice[c1_index]

            if x1 is not None:
                x1_indice = [0] * len(x1.shape)
                if len(x1.shape) == 5:
                    x1_indice[4] = indice[c0_index]
                    x1_indice[1] = indice[c1_index]
                else:
                    x1_indice[0] = indice[c1_index] * 16 + indice[c0_index]

                if tensor_flag:
                    func = tvm.vdeq_cast(x0(*indice),
                                         deq_scale(*deq_indice), "int16", do_relu=relu_flag) + x1(*x1_indice)
                else:
                    func = tvm.deq_cast(x0(*indice), deq_scale(*deq_indice), "int16") + x1(*x1_indice)
            else:
                if tensor_flag:
                    func = tvm.vdeq_cast(x0(*indice), deq_scale(*deq_indice), "int16", do_relu=relu_flag)
                else:
                    func = tvm.deq_cast(x0(*indice), deq_scale(*deq_indice), "int16")
            return func

    return lambda_func


def _check_params(x0, deq_scale, x1, y, relu_flag, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr
    """
    shape_x0 = x0.get("shape")
    format_x0 = x0.get("format")
    dtype_x0 = x0.get("dtype")

    shape_deq = deq_scale.get("shape")
    format_deq = deq_scale.get("format")
    dtype_deq = deq_scale.get("dtype")

    check_list = [("int32",), ("uint64",), ("int16",)]
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_dtype(dtype_x0, check_list[0], param_name="x0")
    para_check.check_dtype(dtype_deq, check_list[1], param_name="deq_scale")

    para_check.check_format(format_x0, format_list, param_name="x0")

    if format_x0 == "NC1HWC0":
        para_check.check_shape(shape_x0, min_rank=5, max_rank=5, param_name="x0")

    if format_x0 == "FRACTAL_NZ":
        para_check.check_shape(shape_x0, min_rank=4, param_name="x0")

    para_check.check_shape(shape_deq, min_rank=5, max_rank=5, param_name="deq_scale")

    para_check.check_format(format_deq, ("NC1HWC0",), param_name="deq_scale")

    if shape_deq[0] != 1 or shape_deq[2] != 1 or shape_deq[3] != 1:
        detail = "deq_scale shape must be 1 in n,h,w"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "deq_scale", detail)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ascend_dequant_s16(x0, deq_scale, x1, y, relu_flag=False, kernel_name="ascend_dequant_s16"):
    """
    int32 -> int16

    Parameters:
    ----------
    x0 : the placeholder of input
    deq_scale: the placeholder of deq_scale
    x1: the placeholder of add input tensor
    y: the dict of output
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_dequant_s16"

    Returns:
    -------
    None
    """
    _check_params(x0, deq_scale, x1, y, relu_flag, kernel_name)
    shape_x0 = x0.get("shape")
    format_x0 = x0.get("format")
    dtype_x0 = x0.get("dtype")
    shape_deq = deq_scale.get("shape")
    dtype_deq = deq_scale.get("dtype")

    if format_x0 == "NC1HWC0":
        # n, C1, H*W, C0
        shape_x0 = [shape_x0[0], shape_x0[1], shape_x0[2] * shape_x0[3], shape_x0[4]]

    ori_shape_deq = deq_scale.get("ori_shape")
    attr = {"ori_shape": ori_shape_deq}
    input_x0 = tvm.placeholder(shape_x0, dtype_x0, "x0")
    input_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr)
    input_x1 = None
    if x1:
        format_x1 = x1.get("format")
        shape_bias = x1.get("shape")
        if format_x1 == "ND" and len(shape_bias) != 1:
            detail = "when x1 format is ND, the length of x1 shape must be 1"
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x1", detail)

        input_x1 = tvm.placeholder(shape_bias, "int16", "x1")

    with tvm.target.cce():
        res = ascend_dequant_s16_compute(input_x0, input_deq, input_x1, relu_flag, kernel_name)
        generic.auto_schedule(res)
