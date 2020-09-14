#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

vector schedule
"""
import abc
from te import platform as cceconf
from te import tvm



# 'pylint: disable=no-self-use, too-many-instance-attributes, too-few-public-methods, redefined-builtin, useless-object-inheritance
class VectorSchedule(object):
    """
    Base class of cce vector schedule

    Parameters
    ----------
    None

    Returns
    -------
    VectorSchedule_instance : instance of VectorSchedule
    """

    def __init__(self):

        self._schedule = None
        self._schedule_valid = True
        self._need_db = False
        self._need_multi_core = True
        self._spec_node_list = []
        self._multi_core_bind_tensor = None
        self._multi_core_fused_axis = None
        self._out_tensors = []

        # cache read para map
        self._cache_read_tensors_and_readers_map = {}

        # cache read result map
        self._cache_read_tensors_and_buffer_map = {}

        # cache write para list
        self._cache_write_tensors = []

        # cache write result map
        self._cache_write_tensors_and_buffer_map = {}

        # compute inline para list
        self._compute_inline_tensors = []

        # double buffer para list
        self._double_buffer_tensors = []

        # record double buffer map[read] = write
        self._double_buffer_map = {}

        self._tiling_tensor = None

        self._insn_map = {}

        self._reg_insn_map = {}

        self._tiling_para = {"block_tiling": {"axis": 0, "factor": 1},
                             "ub_tiling": {"axis": 0, "factor": 1}}

        self._tiling_result = {}
        self._compute_at_map = {}

        self._emit_insn_map = {}

        self._scope = "local.UB"

    def _do_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        self._double_buffer_tensors.clear()

        for i in self._cache_read_tensors_and_readers_map:
            readers = self._cache_read_tensors_and_readers_map[i]
            read_buffer = self._schedule.cache_read(i, self._scope, readers)

            self._cache_read_tensors_and_buffer_map[i] = read_buffer

            self._double_buffer_tensors.append(read_buffer)

    def _do_cache_write(self):
        """
        cache write operations

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for i in self._cache_write_tensors:
            write_buffer = self._schedule.cache_write(i, self._scope)
            self._cache_write_tensors_and_buffer_map[i] = write_buffer

    def _do_tiling(self):
        res = self._tiling_tensor
        block_tiling_para = self._tiling_para["block_tiling"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner_size = block_tiling_para["factor"]

        ub_tiling_para = self._tiling_para["ub_tiling"]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        res_block_outer, res_block_inner = self._schedule[res].split(
            res.op.axis[block_split_axis], factor=block_split_inner_size)

        block_tiling_result = {"axis": block_split_axis,
                               "parent_itervar": res.op.axis[block_split_axis],
                               "outer_itervar": res_block_outer,
                               "inner_itervar": res_block_inner}

        if block_split_axis == ub_split_axis:
            res_ub_outer, res_ub_inner = self._schedule[res].split(
                res_block_inner, factor=ub_split_inner)
            ub_tiling_result = {"axis": ub_split_axis,
                                "parent_itervar": res_block_inner,
                                "outer_itervar": res_ub_outer,
                                "inner_itervar": res_ub_inner}

        else:
            res_ub_outer, res_ub_inner = self._schedule[res].split(
                res.op.axis[ub_split_axis], factor=ub_split_inner)
            ub_tiling_result = {"axis": ub_split_axis,
                                "parent_itervar": res.op.axis[ub_split_axis],
                                "outer_itervar": res_ub_outer,
                                "inner_itervar": res_ub_inner}

        self._tiling_result = {"block_tiling": block_tiling_result,
                               "ub_tiling": ub_tiling_result}

    def _do_compute_inline(self):
        """
        compute inline operations
        """
        for i in self._compute_inline_tensors:
            self._schedule[i].compute_inline()

    def _do_multi_core(self):
        if self._need_multi_core:
            res = self._multi_core_bind_tensor
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[res].bind(self._multi_core_fused_axis, block)

    def _do_compute_at(self):
        for stage in self._compute_at_map:
            parent_stage = self._compute_at_map[stage]["parent"]
            scope_iter_var = self._compute_at_map[stage]["scope"]
            self._schedule[stage].compute_at(parent_stage, scope_iter_var)

    def _do_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        temp_write_buffer = []
        if self._need_db:
            for i in self._double_buffer_tensors:
                self._schedule[i].double_buffer()
                # just for ternary instruction
                if i in self._double_buffer_map:
                    buffers = list(set(self._double_buffer_map[i]))
                    for buffer in buffers:
                        temp_write_buffer.append(buffer)
                        self._schedule[buffer].double_buffer()
            if temp_write_buffer:
                self._recursive_double_buffer(temp_write_buffer)

    def _recursive_double_buffer(self, write_buffer):
        """
        open cache write double buffer for ternary instruction by recursive
        """
        if not write_buffer:
            return

        temp_write_buffer = []
        for i in write_buffer:
            if i in self._double_buffer_map:
                buffers = list(set(self._double_buffer_map[i]))
                for buffer in buffers:
                    temp_write_buffer.append(buffer)
                    self._schedule[buffer].double_buffer()
        self._recursive_double_buffer(temp_write_buffer)

    def _do_emit_insn(self):
        for stage in self._emit_insn_map:
            scope_iter_var = self._emit_insn_map[stage]["scope"]
            instruction = self._emit_insn_map[stage]["instruction"]
            self._schedule[stage].emit_insn(scope_iter_var, instruction)

    @abc.abstractmethod
    def _construct_compute_graph(self, out_tensors, spec_node_list):
        return

    @abc.abstractmethod
    def _calculate_cache_read(self):
        return

    @abc.abstractmethod
    def _calculate_cache_write(self):
        return


    @abc.abstractmethod
    def _calculate_compute_inline(self):
        return

    @abc.abstractmethod
    def _calculate_multi_core(self):
        return

    @abc.abstractmethod
    def _calculate_compute_at(self):
        return

    @abc.abstractmethod
    def _calculate_double_buffer(self):
        return

    @abc.abstractmethod
    def _calculate_emit_insn(self):
        return


    def _get_block_num(self):
        return cceconf.get_soc_spec("CORE_NUM")


    def _map_apend(self, input_map, key, value):
        if input_map.get(key):
            if isinstance(value, list):
                for tmp_value in value:
                    if tmp_value not in input_map[key]:
                        input_map[key].append(tmp_value)
            else:
                if value not in input_map[key]:
                    input_map[key].append(value)
        else:
            if isinstance(value, list):
                input_map[key] = value
            else:
                input_map[key] = [value]

    def _shape_to_list(self, shape):
        """
        translate tvm.shape to list type in python
        """
        tmp = []
        for i in shape:
            if isinstance(i, tvm.expr.Var):
                tmp.append(i)
            else:
                tmp.append(i.value)
        return tmp

    def get_dst_tensor_map(self, reslist, tensor_map):
        """
        get the dst_tensor list of the tensor with more than one dst_tensor
        tensor_map = {input: outputlist}
        """
        for out_tensor in reslist:
            for in_tensor in list(out_tensor.op.input_tensors):
                if in_tensor in tensor_map:
                    if out_tensor not in tensor_map[in_tensor]:
                        tensor_map[in_tensor].append(out_tensor)
                else:
                    tensor_map[in_tensor] = [out_tensor]
                    self.get_dst_tensor_map([in_tensor], tensor_map)

    def get_align_factor(self, dtype):
        """
        get_align_factor
        """
        # base on the diff data type, get the align_factor
        align_factor = 16
        dtype_bytes = 2
        if dtype in ('int8', 'uint8'):
            align_factor = 32
            dtype_bytes = 1
        elif dtype in ('float16', 'int16', 'uint16'):
            align_factor = 16
            dtype_bytes = 2
        else:
            align_factor = 8
            dtype_bytes = 4
        return align_factor, dtype_bytes

