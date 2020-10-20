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
ascend_requant
"""
from functools import reduce as function_reduce
from te import tvm
import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic
from te.utils import para_check
from te.utils import shape_util
from te.utils import error_manager
from impl import ascend_quant_util as util


# pylint: disable=invalid-name,unused-argument,unnecessary-lambda,too-many-arguments,too-many-locals
@fusion_manager.register("ascend_requant")
def ascend_requant_compute(x, req_scale, y, relu_flag=False, kernel_name="ascend_requant"):
    """
    int32 -> int8

    Parameters:
    ----------
    x: the placeholder of input
    req_scale: the placeholder of req_scale
    y: the dict of output
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_requant"

    Returns:
    -------
    res : the result of ascend_requant
    """

    x_shape = x.shape
    x_shape_list = shape_util.shape_to_list(x_shape)
    align_shape = x_shape_list.copy()

    # the tensor is a constant or vector based on the original shape
    ori_shape_req = req_scale.op.attrs["ori_shape"]
    ori_shape_req_list = shape_util.shape_to_list(ori_shape_req)
    req_dim = function_reduce(lambda x, y: x * y, ori_shape_req_list[:])
    tensor_flag = False
    if req_dim > 1:
        tensor_flag = True

    c1_index = 1
    if util.is_nz_format(x):
        c1_index = len(x_shape) - 4

    if x.op.tag == "depthwise_conv2d":
        align_shape[4] = 16
        align_shape[3] = (x_shape_list[3] + 15) // 16 * 16
        align_shape[2] = 1
        if tensor_flag:
            align_shape[1] = (x_shape_list[1] * x_shape_list[2] * 16 + 31) // 32 * 32 // 16
        else:
            align_shape[1] = x_shape_list[1] * x_shape_list[2]
        align_shape[0] = x_shape_list[0]

        if tensor_flag:
            res_ub = tvm.compute(align_shape,
                                 lambda i, j, a, k, l:
                                 tvm.vdeq_cast(x(i, j // 2, j % 2, k, l), req_scale(0, j, 0, 0, l), "int8",
                                               do_relu=relu_flag),
                                 name="s32_to_s8", tag="requant_vector")
        else:
            res_ub = tvm.compute(align_shape,
                                 lambda i, j, a, k, l:
                                 tvm.deq_cast(x(i, j // 2, j % 2, k, l), req_scale(0, 0, 0, 0, 0), "int8"),
                                 name="s32_to_s8", tag="requant_scale")
    else:
        align_shape[c1_index] = (align_shape[c1_index] + 1) // 2 * 2
        align_shape[-2] = (align_shape[-2] + 15) // 16 * 16
        res_ub = _s32_to_s8_normal_compute(x, req_scale, align_shape, c1_index, tensor_flag, relu_flag)

    if util.is_nz_format(x):
        res = _format_transfer_nz(align_shape, res_ub, c1_index)
        return res

    res_ub_reform = _format_transfer(align_shape, res_ub, c1_index)
    res_shape = shape_util.shape_to_list(res_ub_reform.shape)

    res_shape[-2] = x.shape[-2]

    res = tvm.compute(res_shape, lambda *indice: res_ub_reform(*indice),
                      name="requant_remove_pad", tag="requant_remove_pad")
    return res


def _s32_to_s8_normal_compute(x, req_scale, align_shape, c1_index, tensor_flag, relu_flag):
    """
    generate s32_to_s8 compute
    """
    if tensor_flag:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(x, req_scale, align_shape, c1_index, tensor_flag, relu_flag),
                             name="s32_to_s8", tag="requant_vector")
    else:
        res_ub = tvm.compute(align_shape,
                             _deq_cast_compute(x, req_scale, align_shape, c1_index, tensor_flag, relu_flag),
                             name="s32_to_s8", tag="requant_scale")
    return res_ub


def _deq_cast_compute(x, req_scale, align_shape, c1_index, tensor_flag, relu_flag):
    """
    generate lambda func
    """
    n_dim = len(align_shape)
    c0_index = n_dim - 1
    x_shape_list = shape_util.shape_to_list(x.shape)

    def lambda_func(*indice):
        new_indice = [0] * 5
        if tensor_flag:
            new_indice[4] = indice[c0_index]
            new_indice[1] = indice[c1_index]

        if tensor_flag:
            return tvm.select(indice[c1_index] < x_shape_list[c1_index],
                              tvm.vdeq_cast(x(*indice), req_scale(*new_indice), "int8", do_relu=relu_flag),
                              tvm.const(0, dtype="int8"))
        return tvm.select(indice[c1_index] < x_shape_list[c1_index],
                          tvm.deq_cast(x(*indice), req_scale(*new_indice), "int8"),
                          tvm.const(0, dtype="int8"))

    return lambda_func


def _format_compute(tensor, trans_shape, c1_index):
    """
    generate lambda func
    """
    n_dim = len(trans_shape)
    c0_index = n_dim - 1

    def lambda_func(*indice):
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * 32 + indice[c0_index]) % 16
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * 32 + indice[c0_index]) // 16
            else:
                new_indice[i] = indice[i]
        return tensor(*new_indice)

    return lambda_func


def _format_transfer_nz(shape, x, c1_index):
    """
    C0 from 16 to 32 for FRACTAL_NZ
    """
    trans_shape = shape[:]
    trans_shape[c1_index] = trans_shape[c1_index] // 2
    trans_shape[-1] = trans_shape[-1] * 2
    res = tvm.compute(trans_shape,
                      _format_compute(x, trans_shape, c1_index),
                      name="data_transfer", tag="requant_data_transfer")
    res = tvm.compute(trans_shape, lambda *i: res[i], name="res", tag="requant_NZ")
    return res


def _format_transfer(shape, x, c1_index):
    """
    C0 from 16 to 32 for NC1HWC0
    """
    trans_shape = shape[:]
    trans_shape[c1_index] = trans_shape[c1_index] // 2
    trans_shape[-1] = trans_shape[-1] * 2
    res = tvm.compute(trans_shape,
                      _format_compute(x, trans_shape, c1_index),
                      name="data_transfer", tag="data_transfer")
    return res


def _check_params(x, req_scale, y, relu_flag, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr
    """
    shape_x = x.get("shape")
    format_x = x.get("format")

    shape_req = req_scale.get("shape")
    format_req = req_scale.get("format")

    dtype_x = x.get("dtype")
    dtype_req = req_scale.get("dtype")

    check_list = [("int32",), ("uint64",)]
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_dtype(dtype_x, check_list[0], param_name="x")
    para_check.check_dtype(dtype_req, check_list[1], param_name="req_scale")
    para_check.check_format(format_x, format_list, param_name="x")

    if format_x == "NC1HWC0":
        para_check.check_shape(shape_x, min_rank=5, max_rank=5, param_name="x")

    if format_x == "FRACTAL_NZ":
        para_check.check_shape(shape_x, min_rank=4, param_name="x")

    para_check.check_shape(shape_req, min_rank=5, max_rank=5, param_name="req_scale")

    para_check.check_format(format_req, ("NC1HWC0",), param_name="req_scale")

    if shape_req[0] != 1 or shape_req[2] != 1 or shape_req[3] != 1:
        detail = "req_scale shape must be 1 in n,h,w"
        error_manager.error_manager_vector.raise_err_input_shape_invalid(kernel_name, "req_scale", detail)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ascend_requant(x, req_scale, y, relu_flag=False, kernel_name="ascend_requant"):
    """
    int32 -> int8

    Parameters:
    ----------
    x: the placeholder of input
    req_scale: the placeholder of req_scale
    y: the dict of output
    relu_flag: the relu mode, default value is False
    kernel_name: cce kernel name, default value is "ascend_requant"

    Returns:
    -------
    None
    """
    _check_params(x, req_scale, y, relu_flag, kernel_name)
    shape_x = x.get("shape")
    format_x = x.get("format")
    shape_req = req_scale.get("shape")
    dtype_x = x.get("dtype")
    dtype_req = req_scale.get("dtype")

    if format_x == "NC1HWC0":
        # n, C1, H*W, C0
        shape_x = [shape_x[0], shape_x[1], shape_x[2] * shape_x[3], shape_x[4]]

    ori_shape_req = req_scale.get("ori_shape")
    attr = {"ori_shape": ori_shape_req}
    input_x = tvm.placeholder(shape_x, dtype_x, "x")
    input_req = tvm.placeholder(shape_req, name="req_scale", dtype=dtype_req, attrs=attr)

    with tvm.target.cce():
        res = ascend_requant_compute(input_x, input_req, relu_flag,
                                     kernel_name)
        generic.auto_schedule(res)
