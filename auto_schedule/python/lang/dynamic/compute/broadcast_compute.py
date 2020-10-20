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
broadcast compute
"""
from te import tvm

from . import util

NAME_INDEX = [0]


@util.dtype_check_decorator
def broadcast(var, shape, dtype=None):
    """
    broadcast scalar to tensor, only support float16

    Parameters
    ----------
    var : can be python instance of int and float, or tvm.const

    shape : tensor shape

    dtype : tensor dtype , default : var.dtype

    Returns
    -------
    wrapped_tensor : broadcast tensor
    """
    if not isinstance(shape, (list, tuple, tvm.container.Array)):
        raise RuntimeError(
            "the input parameter shape must be list or tuple, "
            "while type of input is %s" % (type(shape)))

    valid_types = (int, float, tvm.expr.ConstExpr, tvm.expr.Var,
                   tvm.tensor.Tensor)
    if not isinstance(var, valid_types):
        raise RuntimeError("Only accept int, float, var, tensor.")

    if isinstance(var, tvm.tensor.Tensor):
        return _tensor_broadcast(var, shape)

    var0 = var
    if isinstance(var, (int,)):
        dtype0 = "int32" if dtype is None else dtype
        var0 = tvm.const(var, dtype=dtype0)
    elif isinstance(var, (float,)):
        dtype0 = "float16" if dtype is None else dtype
        var0 = tvm.const(var, dtype=dtype0)
    else:
        if dtype is not None:
            var0.astype(dtype)

    name = "broadcast_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    _op = 'broadcast'
    with tvm.tag_scope(_op):
        out = tvm.compute(shape, lambda *indices: var0, name=name)
    return out


def _tensor_broadcast(var, shape) -> tvm.tensor.Tensor:
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
    orig_shape = util.shape_to_list(tensor.shape)
    util.check_input_tensor_shape(orig_shape)
    if len(orig_shape) > len(shape):
        raise RuntimeError(
            "Length of original shape must be smaller than target shape, "
            "but src shape is %s, and dst shape is %s" %
            (str(orig_shape), str(shape)))

    valid_types = (tvm.expr.ConstExpr, tvm.expr.Var)
    difference = len(shape) - len(orig_shape)
    orig_shape = difference * [1] + orig_shape
    check_equal = 0
    is_unknown_broadcast = False
    for src_shape, dst_shape in zip(orig_shape, shape):
        if util.equal(src_shape, dst_shape):
            check_equal += 1
            continue
        if util.equal(src_shape, 1):
            continue
        if isinstance(src_shape, valid_types) or \
                isinstance(dst_shape, valid_types):
            is_unknown_broadcast = True
            continue
        raise RuntimeError(
            "For tensor broadcasting, shape must be the same or "
            "corresponding shape of src tensor is 1"
            "while src shape is %s, and dst shape is %s" %
            (str(orig_shape), str(shape)))
    if check_equal == len(shape):
        return tensor

    name = "broadcast_tensor_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1
    if is_unknown_broadcast:
        _op = 'unknown_broadcast'
    else:
        _op = 'unified_broadcast'

    def lambda_func(*indices):
        return tensor(*([0 if orig_shape[i] == 1 else indices[i]
                         for i in range(len(orig_shape))][difference:]))

    with tvm.tag_scope(_op):
        out = tvm.compute(shape, lambda_func, name=name)

    return out
