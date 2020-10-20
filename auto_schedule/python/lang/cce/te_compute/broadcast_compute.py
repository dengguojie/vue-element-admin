# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
broadcat compute
"""
from te import tvm
from .util import dtype_check_decorator
from .util import judge_var
from .util import shape_to_list
from .util import check_input_tensor_shape

NAME_INDEX = [0]


@dtype_check_decorator
def broadcast(var, shape, output_dtype=None):
    """
    broadcast scalar to tensor, only support float16

    Parameters
    ----------
    var : can be python instance of int and float, or tvm.const

    shape : tensor shape

    output_dtype : tensor dtype , default : var.dtype

    Returns
    -------
    wrapped_tensor : broadcast tensor
    """
    if not isinstance(shape, (list, tuple, tvm.container.Array)):
        raise RuntimeError(
            "the input parameter shape must be list or tuple, "
            "while type of input is %s" % (type(shape)))

    if isinstance(var, tvm.tensor.Tensor):
        return tensor_broadcast(var, shape)

    var_type = judge_var(var)
    tmp_args = var
    if var_type == "python_const":
        if isinstance(tmp_args, float):
            tmp_args = tvm.const(tmp_args, dtype="float16")
        else:
            tmp_args = tvm.const(tmp_args, dtype="int32")

    if not output_dtype:
        output_dtype = tmp_args.dtype

    tmp_args = tmp_args.astype(output_dtype)

    lambda_func = lambda *indice: tmp_args

    name = "broadcast_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    _op = 'broadcast'
    with tvm.tag_scope(_op):
        out = tvm.compute(shape, lambda_func, name=name)
    return out


def tensor_broadcast(var, shape) -> tvm.tensor.Tensor:
    """
    broadcast tensor to tensor

    Parameters
    ----------
    var : can be tvm.tensor.Tensor

    shape : tensor shape

    Returns
    -------
    wrapped_tensor : broadcast tensor
    """
    tensor = var
    orig_shape = shape_to_list(tensor.shape)
    check_input_tensor_shape(orig_shape)
    if len(orig_shape) > len(shape):
        raise RuntimeError(
            "Length of original shape must be smaller than target shape, "
            "but src shape is %s, and dst shape is %s" % (
                str(orig_shape), str(shape)))
    difference = len(shape) - len(orig_shape)
    orig_shape = difference * [1] + orig_shape
    check_equal = 0
    for src_shape, dst_shape in zip(orig_shape,
                                    shape):
        if src_shape == dst_shape:
            check_equal += 1
            continue
        if src_shape == 1:
            continue
        raise RuntimeError(
            "For tensor broadcasting, shape must be the same or "
            "corresponding shape of src tensor is 1"
            "while src shape is %s, and dst shape is %s" % (
                str(orig_shape), str(shape)))
    if check_equal == len(shape):
        return tensor

    name = "broadcast_tensor_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1
    _op = 'broadcast_for_tensor'

    def lambda_func(*indice):
        return tensor(
            *([0 if orig_shape[i] == 1 else indice[i] for i in
               range(len(orig_shape))][difference:]))

    with tvm.tag_scope(_op):
        out = tvm.compute(shape, lambda_func, name=name)
    return out
