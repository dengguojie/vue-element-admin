#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
array
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings
import types

from tbe import tvm
from tbe.common.testing.dsl_source_info import source_info_decorator
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.compute.transdata import transdata_compute
from tbe.dsl.base.expr_compare import expr_equal
from .util import dtype_check_decorator
from .util import check_input_tensor_shape
from .util import shape_to_list
from .util import in_dynamic_and_static_unify

NAME_INDEX = [0]


def _index_offset(shape, axis, offset, *index):
    """Compute the offset of index along one dimension.

    Parameters
    ----------
    shape: list
        shape of tensor.
    axis: int
        the dimension along which to split.
    offset: int
        axis offset.
    index: list or tuple
        index value list.

    Returns
    -------
    output_index: list
        output index with one input index value add axis offset.
    """
    input_index = list(index)
    output_index = []
    for i, _ in enumerate(shape):
        if i == axis:
            input_index[i] = input_index[i] + offset
        output_index.append(input_index[i])

    return output_index


@source_info_decorator()
def split_compute_com(data, split_dim, size_splits):
    """
    Split a tensor into len(size_splits) tensors along one dimension
    """
    warnings.warn("split_compute_com is expired, please replace it with the func split",
                  DeprecationWarning)
    return split(data, split_dim, size_splits)


@source_info_decorator()
def split(data, split_dim, size_splits):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    data: TVM tensor
        input tensor.
    split_dim: int
        the dimension along which to split.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor along `split_dim`.

    Returns
    -------
    output_shape_list: list
        the list of output shapes.
    output_tensor_list: list
        the list of output tensors, output tensor type is TVM tensor.
    """
    input_shape = shape_to_list(data.shape)

    output_shape_list = []
    for size in size_splits:
        input_shape[split_dim] = size
        output_shape_list.append(list(input_shape))

    offset = 0
    output_shape = None
    output_tensor_list = []
    for i, _ in enumerate(output_shape_list):
        output_shape = output_shape_list[i]
        name = 'tensor{}'.format(str(i))
        output_tensor = tvm.compute(
            output_shape,
            lambda *index: data(
                *_index_offset(output_shape, split_dim, offset, *index)),
            name=name, tag="split_com|compute_{}".format(str(i)))
        output_tensor_list.append(output_tensor)
        offset = offset + output_shape[split_dim]

    return output_shape_list, output_tensor_list


@dtype_check_decorator
def concat(raw_tensors, axis):
    """
    concat shapes at axis,  support int8, uint8, int16, int32 float16, float32
    Parameters
    ----------
    raw_tensors : list of tensors
    axis : concat axis
    Returns
    -------
    concat tensor :
    """
    if axis < 0:
        axis = axis + len(raw_tensors[0].shape)
    _concat_para_check(raw_tensors, axis)

    if in_dynamic_and_static_unify():
        return _unify_concat(raw_tensors, axis)

    def _get_input_tensors():
        shapes = []
        for in_tensor in list(raw_tensors):
            shape = [int(in_tensor.shape[i].value) for i in range(len(in_tensor.shape))]
            shapes.append(shape)

        _shapes = list(shapes)
        return _shapes

    shapes = _get_input_tensors()

    res_shape = shapes[0][:]
    for i in range(1, len(shapes)):
        res_shape[axis] += shapes[i][axis]

    sel = []
    n_tensor = len(raw_tensors)

    def compute_func(*indice):
        """
        concat compute expr
        """
        if n_tensor > 1:
            for tensor_i in range(n_tensor - 1):
                if tensor_i == 0:
                    tensor_a = raw_tensors[0]
                    tensor_b = raw_tensors[1]
                    shape_c = shapes[0][:]
                    indice2 = list(indice[:])
                    indice2[axis] = indice[axis] - tensor_a.shape[axis]
                    sel.append(
                        tvm.select(indice[axis] < shape_c[axis],
                                   tensor_a[indice], tensor_b[tuple(indice2)]))
                    shape_c[axis] += shapes[1][axis]
                else:
                    tensor_a = sel[tensor_i - 1]
                    tensor_b = raw_tensors[tensor_i + 1]
                    indice2 = list(indice[:])
                    indice2[axis] = indice[axis] - shape_c[axis]
                    sel.append(tvm.select(indice[axis] < shape_c[axis],
                                          tensor_a, tensor_b[tuple(indice2)]))
                    shape_c[axis] += shapes[tensor_i + 1][axis]
        else:
            return raw_tensors[0][indice]

        return sel[-1]

    res = tvm.compute(res_shape, compute_func, name="concat", tag="concat")

    return res


def _unify_concat(input_tensors, axis):
    """
    concat shapes at axis,  support int8, uint8, int16, int32, float16, float32, int64, uint64
    :param input_tensors: list[tvm.tensor]
    list of input tensors
    :param axis: int
    concat axis
    :return: tvm.tensor: A concat Tensor
    """

    def concat_func(*indices):
        func = None
        concat_axis_size = sum(t.shape[axis] for t in input_tensors)
        for tensor in reversed(input_tensors):
            index = []
            for i, _ in enumerate(dst_shape):
                if i == axis:
                    index.append(indices[i] - (concat_axis_size - tensor.shape[axis]))
                else:
                    index.append(indices[i])
            if func is None:
                func = tensor(*index)
            else:
                func = tvm.select(indices[axis] < concat_axis_size, tensor(*index), func)
            concat_axis_size -= tensor.shape[axis]
        return func

    with tvm.tag_scope("concat"):
        dst_shape = list(input_tensors[0].shape)
        concat_axis_size = sum(t.shape[axis] for t in input_tensors)
        dst_shape[axis] = concat_axis_size
        concat = tvm.compute(dst_shape, concat_func, name="concat")
    return concat


def _concat_para_check(raw_tensors, axis):
    """
    concat parameter check

    Parameters
    ----------
    raw_tensors : list of tensors
    axis : concat axis

    Returns
    -------
    rasie runtime error
    """
    if not isinstance(axis, int):
        raise RuntimeError("The axis type must be int")

    # check shape
    if axis < 0 or axis >= len(raw_tensors[0].shape):
        raise RuntimeError(
            "concat axis must be in [-%d - %d), actual is %d"
            % (len(raw_tensors[0].shape), len(raw_tensors[0].shape), axis))
    check_input_tensor_shape(raw_tensors[0])
    for i in range(1, len(raw_tensors)):
        if not isinstance(raw_tensors[i], tvm.tensor.Tensor):
            raise RuntimeError("The each element of input type must be tvm.tensor")
        check_input_tensor_shape(raw_tensors[i])
        if raw_tensors[i].dtype != raw_tensors[0].dtype:
            raise RuntimeError("dtype must be the same to each other")
        if in_dynamic_and_static_unify():
            for j in range(len(raw_tensors[0].shape)):
                if (j != axis) and not expr_equal(raw_tensors[i].shape[j], raw_tensors[0].shape[j]):
                    raise RuntimeError(
                        "concat input shape value must be the same to each other except concat axis")
        else:
            for j in range(len(raw_tensors[0].shape)):
                if (j != axis) and (raw_tensors[i].shape[j].value != raw_tensors[0].shape[j].value):
                    raise RuntimeError(
                        "concat input shape len must be the same to each other except concat axis")


@source_info_decorator()
@dtype_check_decorator
def set_value(tensor, condition, value):
    """
    set specified value
    Parameters
    ----------
    tensor: tvm.tensor

    condition: lambda expr

    value: const, variable or lambda expr
    Returns
    -------
    wrapped_tensor: updated tensor
    """
    shape = shape_to_list(tensor.shape)
    if isinstance(value, types.FunctionType):
        lambda_func = lambda *indice: tvm.select(condition(*indice), value(*indice), tensor(*indice))
    else:
        lambda_func = lambda *indice: tvm.select(condition(*indice), value, tensor(*indice))

    name = "set_value_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    with tvm.tag_scope("set_value"):
        out = tvm.compute(shape, lambda_func, name=name)

    return out


@source_info_decorator()
@dtype_check_decorator
def transpose(tensor, axes):
    """
    transpose a tensor by permute

    Parameters
    ----------
    tensor : tvm.tensor
        Original tensor
    axes : list[int]
        Permutes the dimensions according to the value of axes
    Returns
    -------
    tvm.tensor: A transposed Tensor
    """

    def check_input():
        input_shape = shape_to_list(tensor.shape)
        for shape in input_shape:
            if (isinstance(shape, int) and shape < 0) or not isinstance(shape, (tvm.expr.Expr, int)):
                dict_args = {"errCode": "E90001",
                             "detailed_cause": "The input shape value [%s] must be a positive integer or tvm expr"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
        if not isinstance(axes, (list, tuple)):
            dict_args = {"errCode": "E90001", "detailed_cause": "The axes must be list or tuple"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        sorted_axes = sorted(axes)
        base_axes = [i for i, _ in enumerate(axes)]
        if sorted_axes != base_axes:
            dict_args = {"errCode": "E90001", "detailed_cause": "The input axes error, cannot transpose"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    check_input()
    with tvm.tag_scope("transpose"):
        src_shape = tensor.shape
        dst_shape = tuple(src_shape[i] for i in axes)
        attrs = {"permute": axes}
        transpose = tvm.compute(dst_shape,
                                lambda *index: tensor(*(x for _, x in sorted(zip(axes, index)))),
                                attrs=attrs, name="transpose")
    return transpose


@source_info_decorator()
@dtype_check_decorator
def gather(params, indices, axis=None, batch_dims=0):
    """
    :param params:
    :param indices:
    :param axis:
    :param batch_dims:
    :return:
    """
    p_shape = params.shape
    i_shape = indices.shape
    p_name = params.name
    i_name = indices.name
    i_dtype = indices.dtype

    # check indices dytpe
    _check_gather_indices_dtype(i_dtype)

    g_shape = p_shape[:axis] + i_shape[batch_dims:] + p_shape[axis + 1:]

    def index(i):
        end = axis + len(i_shape) - batch_dims
        idx_pos = i[:batch_dims] + i[axis:end]
        return list(i[:axis]) + [indices[idx_pos]] + list(i[end:])

    op_name = "gather"
    compute_name = op_name + str(_get_next_name_index())
    with tvm.tag_scope(op_name):
        g_tensor = tvm.compute(g_shape, lambda *i: params(*index(i)), name=compute_name,
                               attrs={"params_name": p_name, "indices_name": i_name})

    return g_tensor


@source_info_decorator()
@dtype_check_decorator
def gather_nd(params, indices, batch_dims=0):
    """
    :param params:
    :param indices:
    :param batch_dims:
    :return:
    """
    p_shape = params.shape
    i_shape = indices.shape
    p_name = params.name
    i_name = indices.name
    i_dtype = indices.dtype

    # check indices dytpe
    _check_gather_indices_dtype(i_dtype)

    rank_value = i_shape[-1].value if isinstance(i_shape[-1], tvm.expr.ConstExpr) else i_shape[-1]

    if rank_value > len(p_shape):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "indices last dim value [%u] > params shape length [%u]! " % (
            rank_value, len(p_shape))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    g_shape = i_shape[:-1] + p_shape[batch_dims + rank_value:]

    def index(i):
        p_pos = []
        p_pos.extend(i[:batch_dims])
        axis = len(i_shape) - 1
        for j in range(rank_value):
            p_pos.append(indices(*i[:axis], j))
        p_pos.extend(i[axis:])
        return p_pos

    op_name = "gather_nd"
    compute_name = op_name + str(_get_next_name_index())
    with tvm.tag_scope(op_name):
        g_tensor = tvm.compute(g_shape, lambda *i: params(*index(i)), name=compute_name,
                               attrs={"params_name": p_name, "indices_name": i_name})

    return g_tensor


def _check_gather_indices_dtype(i_dtype):
    """
    :param i_dtype: indices dtype str
    :return:
    """
    if i_dtype not in ("int32, int64"):
        dict_args = {}
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "indices tensors dtype must be int32 or int64! " \
                                      "while dtype is [%s], " % (i_dtype,)
        raise RuntimeError(dict_args, get_error_message(dict_args))


def _get_next_name_index():
    """
    :return: next name index as string
    """
    NAME_INDEX[0] += 1
    return NAME_INDEX[0]

@source_info_decorator()
@dtype_check_decorator
def transdata(tensor, dst_shape, axes_map, pad_value=0):
    """
    transdata a tensor by axes_map and dst_shape

    Parameters
    ----------
    tensor : tvm.tensor
        Original tensor
    dst_shape : list[int]
        Shape of dst_tensor after transdata
    axes_map : dict
        Permutes the dimensions according to the axes_map
    pad_value : int
        Determine the padding value when padding is required
    Returns
    -------
    tvm.tensor: A transdata tensor that shape is dst_shape
    """

    def forward_get_permute(_axes_map):
        permute = []
        for _, v in _axes_map.items():
            if isinstance(v, int):
                permute.append(v)
            elif isinstance(v, (tuple, list)):
                permute.extend(v)
        return [x for _, x in sorted(zip(permute, range(0, len(permute))))]

    def backward_get_permute(_axes_map):
        permute = []
        for k, _ in _axes_map.items():
            if isinstance(k, int):
                permute.append(k)
            elif isinstance(k, (tuple, list)):
                permute.extend(k)
        return permute

    def check():
        input_shape = shape_to_list(tensor.shape)
        for shape in input_shape:
            if (isinstance(shape, int) and shape < 0) or not isinstance(shape, (tvm.expr.Expr, int)):
                dict_args = {"errCode": "E90001",
                             "detailed_cause": "The input shape value [%s] must be a positive integer or tvm expr"}
                raise RuntimeError(dict_args, get_error_message(dict_args))
        if not isinstance(axes_map, dict):
            dict_args = {"errCode": "E90001", "detailed_cause": "The axes_map must dict"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        is_forward = False
        is_backward = False
        for k, v in axes_map.items():
            if isinstance(k, (list, tuple)) and isinstance(v, int):
                is_backward = True
            if isinstance(k, int) and isinstance(v, (list, tuple)):
                is_forward = True

        if is_forward and is_backward:
            # Pad and DePad existed together
            dict_args = {"errCode": "E90001",
                         "detailed_cause": "Pad and DePad existed together is invalid"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if not is_forward and not is_backward:
            # Pad and DePad not existed
            dict_args = {"errCode": "E90001",
                         "detailed_cause": "Pad or DePad is required in transdata, "
                                           "but axes_map not exist Pad or DePad"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if is_forward:
            _axes_map = dict(sorted(axes_map.items()))
            permute = forward_get_permute(_axes_map)
            # check ele of axes_map
            for k, v in _axes_map.items():
                if not isinstance(k, int):
                    dict_args = {"errCode": "E90001", "detailed_cause": "In forward, key of axes_map must be int"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                if not isinstance(v, (int, tuple, list)):
                    dict_args = {"errCode": "E90001", "detailed_cause": "In forward, value of axes_map "
                                                                        "should in [int, tuple, list]"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                if isinstance(v, (list, tuple)) and len(v) > 2:
                    dict_args = {"errCode": "E90001", "detailed_cause":
                        "In forward, only support split DimX to [DimX0, DimX1],"
                        f"but value of axes_map is {v}"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
        else:
            _axes_map = dict(sorted(axes_map.items(), key=lambda x: x[1]))
            permute = backward_get_permute(_axes_map)
            # check ele of axes_map
            for k, v in _axes_map.items():
                if not isinstance(v, int):
                    dict_args = {"errCode": "E90001", "detailed_cause": "In backward, value of axes_map must be int"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                if not isinstance(k, (int, tuple, list)):
                    dict_args = {"errCode": "E90001", "detailed_cause": "In backward, key of axes_map "
                                                                        "should in [int, tuple, list]"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                if isinstance(k, (list, tuple)) and len(k) > 2:
                    dict_args = {"errCode": "E90001", "detailed_cause":
                        "In forward, only support fuse [DimX0, DimX1] to DimX,"
                        f"but key of axes_map is {k}"}
                    raise RuntimeError(dict_args, get_error_message(dict_args))

        sorted_axes = sorted(permute)
        base_axes = [i for i, _ in enumerate(permute)]
        if sorted_axes != base_axes:
            dict_args = {"errCode": "E90001", "detailed_cause": "The input axes_map error, cannot do transdata"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    check()
    return transdata_compute.transdata(tensor, dst_shape, axes_map, pad_value)