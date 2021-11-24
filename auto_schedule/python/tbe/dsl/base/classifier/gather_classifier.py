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
classifier of shape in gather
"""
from itertools import chain

from tbe.dsl.base import operation
from . import util


def classify_gather(ins: list):
    """
    GatherClassifier
    :param ins:
    :return:
    """
    return GatherClassifier(ins).classify()


def classify_gather_nd(ins: list):
    """
    GatherNdClassifier
    :param ins:
    :return:
    """
    return GatherNdClassifier(ins).classify()


class GatherClassifier:
    def __init__(self, ins: list):
        self.is_zeros_shape = False
        self.is_zeros_range = False
        self.is_binary_shape = False

        # params
        self.params_shape = list(ins[0]["shape"])
        self.params_range = list(ins[0]["range"])
        self.params_dtype = ins[0]["dtype"]

        # indices
        self.indices_shape = list(ins[1]["shape"])
        self.indices_range = list(ins[1]["range"])
        self.indices_dtype = ins[1]["dtype"]

        # check status dynamic or static
        self.is_static = operation.get_op_mode() == "static"

        # gather ins like [params, indices, axis, batch_dims]
        self.axis = ins[2]
        if isinstance(ins[2], dict):
            self.axis = self._get_axis_in_const() if self.is_static else ins[2]

        self.batch_dims = ins[3] + len(self.indices_shape) if ins[3] < 0 else ins[3]

        # gather axes rank
        self.gather_rank = 1

        # binary condition
        if -2 in chain(self.params_shape + self.indices_shape):
            self.params_shape = [-1, -1, -1, -1]
            self.params_range = [[1, None], [1, None], [1, None], [1, None], ]

            self.indices_shape = [-1, -1]
            self.indices_range = [[1, None], [1, None]]

            self.batch_dims = 1
            self.axis = 2

            self.is_zeros_range = True
            self.is_binary_shape = True

        self.org_batch_dims = ins[3] if self.is_binary_shape else self.batch_dims

        operation.get_context().add("_batch_dims", self.batch_dims)
        operation.get_context().add("_org_batch_dims", self.org_batch_dims)
        operation.get_context().add("_is_binary_shape", self.is_binary_shape)
        operation.get_context().add("_gather_mode", "gather")

        self._check_zero_shape()

    def _get_axis_in_const(self):
        """
        get const axis value
        :return:
        """
        axis = self.axis["value"] if isinstance(self.axis["value"], int) else self.axis["value"][0]
        return axis + len(self.params_shape) if axis < 0 else axis

    def _check_zero_shape(self):
        # shape value zero
        for dim_value in chain(self.params_shape + self.indices_shape):
            if 0 == dim_value:
                self.is_zeros_shape = True
                break

        # range value zero
        if not self.is_zeros_shape:
            for dim_range in chain(self.params_range + self.indices_range):
                if 0 == dim_range[0]:
                    self.is_zeros_range = True
                    break

    def classify(self):
        """
        classify
        :return:
        """
        # zeros shape
        gather_instances = []
        if self.is_zeros_shape:
            gather_instances.append(_classify_gather_zero_shape(self.params_dtype, self.indices_dtype, "gather"))
            return gather_instances

        # zeros range
        if self.is_zeros_range:
            gather_instances.append(_classify_gather_zero_shape(self.params_dtype, self.indices_dtype, "gather"))

            # change range 0 to 1
            for i, v in enumerate(self.params_range):
                if 0 == v[0]:
                    self.params_range[i] = [1, v[1]]
            for i, v in enumerate(self.indices_range):
                if 0 == v[0]:
                    self.indices_range[i] = [1, v[1]]

        # normal classify
        if self.is_static:
            gather_instances.extend(self._classify_in_static())
        else:
            gather_instances.extend(self._classify_in_dynamic())

        return gather_instances

    def _handle_indices_loops(self):
        # indices loop
        if self.batch_dims == len(self.indices_shape):
            self.indices_loop_shape = 1
            self.indices_loop_range = (1, 1)
        else:
            self.indices_loop_shape = util.combine_dim(self.indices_shape[self.batch_dims:])
            self.indices_loop_range = util.combine_range(self.indices_range[self.batch_dims:])

    def _handle_batch_dims(self):
        # fuse batch dims
        if self.batch_dims == 0:
            self.batch_shape = 1
            self.batch_range = (1, 1)
        else:
            self.batch_shape = util.combine_dim(self.params_shape[:self.batch_dims])
            self.batch_range = util.combine_range(self.params_range[:self.batch_dims])

    def _classify_in_static(self):
        self._handle_indices_loops()

        self._handle_batch_dims()

        # pre loops params
        if self.axis == self.batch_dims:
            pre_loop_shape = 1
            pre_loop_range = (1, 1)
        else:
            pre_loop_shape = util.combine_dim(self.params_shape[self.batch_dims:self.axis])
            pre_loop_range = util.combine_range(self.params_range[self.batch_dims:self.axis])

        # gather axis
        gather_axis_shape = list(self.params_shape[self.axis:self.axis + self.gather_rank])
        gather_axis_range = list(self.params_range[self.axis:self.axis + self.gather_rank])

        # gather after index
        if self.axis + self.gather_rank >= len(self.params_shape):
            after_loop_shape = 1
            after_loop_range = (1, 1)
        else:
            after_loop_shape = util.combine_dim(self.params_shape[self.axis + self.gather_rank:])
            after_loop_range = util.combine_range(self.params_range[self.axis + self.gather_rank:])

        # assemble
        gather_ins = []

        params_dict = _asseble_params_info(
            [self.batch_shape, pre_loop_shape, gather_axis_shape, after_loop_shape],
            [self.batch_range, pre_loop_range, gather_axis_range, after_loop_range],
            self.params_dtype)

        indices_dict = _asseble_indices_info(
            [self.batch_shape, self.indices_loop_shape],
            [self.batch_range, self.indices_loop_range],
            self.indices_dtype)

        gather_ins.append(params_dict)
        gather_ins.append(indices_dict)
        gather_ins.append(2)
        gather_ins.append(1)

        return [gather_ins]

    def _classify_in_dynamic(self):
        gather_instances = []

        self._handle_indices_loops()

        self._handle_batch_dims()

        if isinstance(self.axis, int):
            # gather axis pre loop
            if self.axis == self.batch_dims:
                pre_loop_shape = 1
                pre_loop_range = (1, 1)
            else:
                pre_loop_shape = util.combine_dim(self.params_shape[self.batch_dims:self.axis])
                pre_loop_range = util.combine_range(self.params_range[self.batch_dims:self.axis])

            # params hand
            # gather axis
            gather_axis_shape = list(self.params_shape[self.axis:self.axis + self.gather_rank])
            gather_axis_range = list(self.params_range[self.axis:self.axis + self.gather_rank])

            # gather after index
            if self.axis + self.gather_rank >= len(self.params_shape):
                after_loop_shape = 1
                after_loop_range = (1, 1)
            else:
                after_loop_shape = util.combine_dim(self.params_shape[self.axis + self.gather_rank:])
                after_loop_range = util.combine_range(self.params_range[self.axis + self.gather_rank:])

            # assemble
            gather_ins = []

            params_dict = _asseble_params_info(
                [self.batch_shape, pre_loop_shape, gather_axis_shape, after_loop_shape],
                [self.batch_range, pre_loop_range, gather_axis_range, after_loop_range],
                self.params_dtype)

            indices_dict = _asseble_indices_info(
                [self.batch_shape, self.indices_loop_shape],
                [self.batch_range, self.indices_loop_range],
                self.indices_dtype)

            gather_ins.append(params_dict)
            gather_ins.append(indices_dict)
            gather_ins.append(2)
            gather_ins.append(1)

            gather_instances.append(gather_ins)
        else:
            pre_loop_shape = -1
            pre_loop_range = (1, None)

            gather_axis_shape = [-1, ]
            gather_axis_range = [[1, None]]

            after_loop_shape = -1
            after_loop_range = [1, None]

            # assemble info
            gather_ins = []
            params_dict = _asseble_params_info(
                [self.batch_shape, pre_loop_shape, gather_axis_shape, after_loop_shape],
                [self.batch_range, pre_loop_range, gather_axis_range, after_loop_range],
                self.params_dtype)

            indices_dict = _asseble_indices_info(
                [self.batch_shape, self.indices_loop_shape],
                [self.batch_range, self.indices_loop_range],
                self.indices_dtype)

            gather_ins.append(params_dict)
            gather_ins.append(indices_dict)
            gather_ins.append(2)
            gather_ins.append(1)

            gather_instances.append(gather_ins)
        return gather_instances


class GatherNdClassifier:
    def __init__(self, ins: list):
        self.is_zeros_shape = False
        self.is_zeros_range = False
        self.is_broadcast_shape = False
        self.is_broadcast_range = False
        self.is_binary_shape = False

        # params
        self.params_shape = list(ins[0]["shape"])
        self.params_range = list(ins[0]["range"])
        self.params_dtype = ins[0]["dtype"]

        # indices
        self.indices_shape = list(ins[1]["shape"])
        self.indices_range = list(ins[1]["range"])
        self.indices_dtype = ins[1]["dtype"]

        if len(self.indices_shape) == 1:
            self.indices_shape.insert(0, 1)
            self.indices_range.insert(0, (1, 1))

        # check status dynamic or static
        self.is_static = operation.get_op_mode() == "static"

        # binary condition
        if -2 in chain(self.params_shape + self.indices_shape):
            self.params_shape = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
            self.params_range = [[1, None], [1, None], [1, None], [1, None], [1, None],
                                 [1, None], [1, None], [1, None], [1, None]]

            self.indices_shape = [-1, -1, -1]
            self.indices_range = [[1, None], [1, None], [1, None]]

            self.is_zeros_range = True
            self.is_broadcast_range = True
            self.is_binary_shape = True

        self._check_zero_shape()

        # gather axis
        if self.is_binary_shape:
            self.batch_dims = 1
            self.org_batch_dims = ins[2]
        else:
            self.batch_dims = ins[2] + len(self.indices_shape) if ins[2] < 0 else ins[2]
            self.org_batch_dims = self.batch_dims

        operation.get_context().add("_batch_dims", self.batch_dims)
        operation.get_context().add("_org_batch_dims", self.org_batch_dims)
        operation.get_context().add("_is_binary_shape", self.is_binary_shape)
        operation.get_context().add("_gather_mode", "gather_nd")

        self.axis = self.batch_dims

        # gather axes rank
        self.gather_rank = self.indices_shape[-1]

    def _check_zero_shape(self):
        # shape value zero
        for dim_value in chain(self.params_shape + self.indices_shape[:-1]):
            if 0 == dim_value:
                self.is_zeros_shape = True
                break

        # range value zero
        if not self.is_zeros_shape:
            for range_value in chain(self.params_range + self.indices_range[:-1]):
                if 0 == range_value[0]:
                    self.is_zeros_range = True
                    break

        # shape value broadcast
        if self.indices_shape[-1] == 0:
            self.is_broadcast_shape = True

        # range value broadcast
        if self.indices_range[-1][0] == 0:
            self.is_broadcast_range = True

    def classify(self):
        """
        classify
        :return:
        """
        # zeros shape
        gather_instances = []
        if self.is_zeros_shape:
            gather_instances.append(_classify_gather_zero_shape(self.params_dtype, self.indices_dtype, "gather_nd"))
            return gather_instances

        # broadcast shape
        if self.is_broadcast_shape:
            gather_instances.append(
                _classify_gather_broadcast_shape(self.batch_dims, self.params_shape, self.indices_shape,
                                                 self.params_range, self.indices_range, self.params_dtype,
                                                 self.indices_dtype))
            return gather_instances

        if self.is_zeros_range:
            gather_instances.append(_classify_gather_zero_shape(self.params_dtype, self.indices_dtype, "gather_nd"))

        if self.is_broadcast_range:
            gather_instances.append(
                _classify_gather_broadcast_shape(self.batch_dims, self.params_shape, self.indices_shape,
                                                 self.params_range, self.indices_range, self.params_dtype,
                                                 self.indices_dtype))

        # change range 0 to 1
        if self.is_zeros_range or self.is_broadcast_range:
            for i, v in enumerate(self.params_range):
                if 0 == v[0]:
                    self.params_range[i] = [1, v[1]]
            for i, v in enumerate(self.indices_range):
                if 0 == v[0]:
                    self.indices_range[i] = [1, v[1]]

        # normal classify
        if self.is_static:
            gather_instances.extend(self._classify_in_static())
        else:
            gather_instances.extend(self._classify_in_dynamic())

        return gather_instances

    def _handle_indices_loop(self):
        # indices loop
        self.indices_loop_shape = util.combine_dim(self.indices_shape[self.batch_dims:-1])
        self.indices_loop_range = util.combine_range(self.indices_range[self.batch_dims:-1])

    def _handle_batch_dims(self):
        # fuse batch dims
        if self.batch_dims == 0:
            self.batch_shape = 1
            self.batch_range = (1, 1)
        else:
            self.batch_shape = util.combine_dim(self.params_shape[:self.batch_dims])
            self.batch_range = util.combine_range(self.params_range[:self.batch_dims])

    def _classify_in_static(self):
        self._handle_indices_loop()
        self._handle_batch_dims()

        gather_axis_shape = list(self.params_shape[self.axis:self.axis + self.gather_rank])
        gather_axis_range = list(self.params_range[self.axis:self.axis + self.gather_rank])

        # gather after index
        if self.axis + self.gather_rank >= len(self.params_shape):
            after_loop_shape = 1
            after_loop_range = (1, 1)
        else:
            after_loop_shape = util.combine_dim(self.params_shape[self.axis + self.gather_rank:])
            after_loop_range = util.combine_range(self.params_range[self.axis + self.gather_rank:])

        # assemble
        gather_ins = []
        params_dict = _asseble_params_info(
            [self.batch_shape, gather_axis_shape, after_loop_shape],
            [self.batch_range, gather_axis_range, after_loop_range],
            self.params_dtype)

        # indices gather axis
        indices_index_shape = self.gather_rank
        indices_index_range = (self.gather_rank, self.gather_rank)

        indices_dict = _asseble_indices_info(
            [self.batch_shape, self.indices_loop_shape, indices_index_shape],
            [self.batch_range, self.indices_loop_range, indices_index_range],
            self.indices_dtype)

        gather_ins.append(params_dict)
        gather_ins.append(indices_dict)
        gather_ins.append(1)

        return [gather_ins]

    def _classify_in_dynamic(self):
        gather_instance = []

        self._handle_indices_loop()
        self._handle_batch_dims()

        # know rank
        if self.gather_rank != -1:
            return self._classify_in_static()
        else:
            # gather nd
            # gather rank
            rank_range = len(self.params_shape) - self.axis + 1
            for one_rank in range(1, rank_range):
                gather_axis_shape = list(self.params_shape[self.axis:self.axis + one_rank])
                gather_axis_range = list(self.params_range[self.axis:self.axis + one_rank])

                # gather after index
                if self.axis + one_rank == len(self.params_shape):
                    after_loop_shape = 1
                    after_loop_range = (1, 1)
                else:
                    after_loop_shape = util.combine_dim(self.params_shape[self.axis + one_rank:])
                    after_loop_range = util.combine_range(self.params_range[self.axis + one_rank:])

                # indices gather axis update
                indices_index_shape = one_rank
                indices_index_range = (one_rank, one_rank)

                # assemble info
                gather_ins = []
                params_dict = _asseble_params_info(
                    [self.batch_shape, gather_axis_shape, after_loop_shape],
                    [self.batch_range, gather_axis_range, after_loop_range],
                    self.params_dtype)

                indices_dict = _asseble_indices_info(
                    [self.batch_shape, self.indices_loop_shape, indices_index_shape],
                    [self.batch_range, self.indices_loop_range, indices_index_range],
                    self.indices_dtype)

                gather_ins.append(params_dict)
                gather_ins.append(indices_dict)
                gather_ins.append(1)

                gather_instance.append(gather_ins)
            return gather_instance


def _classify_gather_zero_shape(params_dtype, indices_dtype, gather_type="gather"):
    params_dict = {
        "shape": (0, 0),
        "range": ((0, 0), (0, 0)),
        "dtype": params_dtype
    }

    indices_dict = {
        "shape": (0, 0, 0),
        "range": ((0, 0), (0, 0), (0, 0)),
        "dtype": indices_dtype
    }

    if gather_type == "gather_nd":
        return [params_dict, indices_dict, 1]

    return [params_dict, indices_dict, 1, 1]


def _classify_gather_broadcast_shape(batch_dims, params_shape, indices_shape, params_range,
                                     indices_range, params_dtype, indices_dtype):
    # params like [parmas_batch, params_data]
    # indices like [indices_batch, indices_loops, 0]

    # fuse batch dims
    if batch_dims == 0:
        batch_shape = 1
        batch_range = (1, 1)
    else:
        batch_shape = util.combine_dim(params_shape[:batch_dims])
        batch_range = util.combine_range(params_range[:batch_dims])

    params_data_shape = util.combine_dim(params_shape[batch_dims:])
    params_data_range = util.combine_range(params_range[batch_dims:])

    if batch_dims == len(indices_shape):
        indices_loop_shape = 1
        indices_loop_range = (1, 1)
    else:
        indices_loop_shape = util.combine_dim(indices_shape[batch_dims:-1])
        indices_loop_range = util.combine_range(indices_range[batch_dims:-1])

    params_dict = {
        "shape": (batch_shape, params_data_shape),
        "range": (batch_range, params_data_range),
        "dtype": params_dtype
    }

    indices_dict = {
        "shape": (batch_shape, indices_loop_shape, 0),
        "range": (batch_range, indices_loop_range, (0, 0)),
        "dtype": indices_dtype
    }

    return [params_dict, indices_dict, 1]


def _asseble_params_info(shape_info, range_info, dtype_info):
    params_info = {}
    params_info["dtype"] = dtype_info

    params_shape = []
    params_range = []
    for one_shape, one_range in zip(shape_info, range_info):
        if isinstance(one_shape, list):
            for gather_idx, gather_range in zip(one_shape, one_range):
                params_shape.append(gather_idx)
                params_range.append(gather_range)
        else:
            params_shape.append(one_shape)
            params_range.append(one_range)

    params_info["shape"] = params_shape
    params_info["range"] = params_range

    return params_info


def _asseble_indices_info(shape_info, range_info, dtype_info):
    indices_info = {}
    indices_info["dtype"] = dtype_info
    indices_info["shape"] = shape_info
    indices_info["range"] = range_info

    return indices_info
