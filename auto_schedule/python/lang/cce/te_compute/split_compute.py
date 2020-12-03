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
split compute
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings

import te.lang.cce
from te import tvm


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


def split_compute_com(data, split_dim, size_splits):
    """
    Split a tensor into len(size_splits) tensors along one dimension
    """
    warnings.warn("split_compute_com is expired, please replace it with the func split",
                  DeprecationWarning)
    return split(data, split_dim, size_splits)


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
    input_shape = te.lang.cce.util.shape_to_list(data.shape)

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
