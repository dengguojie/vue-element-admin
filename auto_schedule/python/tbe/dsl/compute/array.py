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

from tbe import tvm
from tbe.common.testing.dsl_source_info import source_info_decorator
from .util import dtype_check_decorator
from .util import check_input_tensor_shape
from .util import shape_to_list


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
        name = 'tensor' + str(i)
        output_tensor = tvm.compute(
            output_shape,
            lambda *index: data(
                *_index_offset(output_shape, split_dim, offset, *index)),
            name=name, tag="split_com|compute_" + str(i))
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
        for j in range(len(raw_tensors[0].shape)):
            if (j != axis) and (raw_tensors[i].shape[j].value != raw_tensors[0].shape[j].value):
                raise RuntimeError(
                    "concat input shape len must be the same to each other except concat axis")
