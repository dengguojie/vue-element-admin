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

reduce atomic schedule
"""
import copy

from te import platform as cceconf
from te import tvm
from te.platform import operation
from te.platform.operation import register_tiling_case, register_schedule

from . import Pattern, INSN_MAPPING, DTYPE_BYTE_MAPPING
from .vector_schedule import VectorSchedule
from .reduce_atomic_schedule import ReduceAtomicSchedule

BLOCK_SIZE_BYTE = 32

# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.REDUCE)
# 'pylint: disable=unused-argument
def calc_tiling_case(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

    reduce_sch = ReduceSchedule()
    is_success = reduce_sch.init(outs, [])
    if not is_success:
        return []
    tiling_cases = reduce_sch.get_tiling_cases()

    operation.get_context().add("reduce_schedule", reduce_sch)

    return tiling_cases


@register_schedule(pattern=Pattern.REDUCE)
# 'pylint: disable=unused-argument
def schedule(outs, tiling_case):
    """
    :param outs:
    :param tiling_case:
    :return:
    """
    reduce_sch = operation.get_context().get("reduce_schedule")
    return reduce_sch.do_schedule(outs, tiling_case)


# 'pylint: disable=too-many-locals, too-many-return-statements,too-few-public-methods,too-many-arguments,too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches,
class ReduceSchedule(VectorSchedule):
    """
    class of cce reduce schedule

    Parameters
    ----------
    VectorSchedule: base class of reduce schedule

    Returns
    -------
    ReduceSchedule : instance of ReduceSchedule
    """

    def __init__(self):
        VectorSchedule.__init__(self)
        # record ops
        self._op = []
        # record origin op
        self._origin_op = []

        self._res_tensor = None
        self._last_output_tensors = []
        self._input_tensors = []
        self._input_tensor_dst_tensor_map = {}
        self._mid_tensors = []  # exclude _input_tensors and last_output_tensor
        self._mid_tensor_dst_tensor_map = {}  # {mid_tensor->dst_tensor}

        self._mid_output_tensors = []
        self._mid_output_tensors_dst_tensor_map = {}

        self._cache_write_exclude_tensors = []

        self._broadcast_last_axis_tensors = []
        self._broadcast_scalars = []
        self._broadcast_scalar_dst_tensor_map = {}
        self._broadcast_not_last_axis_tensors = []

        self._tuple_reduce_tensor_out_list = []

        self._tensor_list_before_reduce = []
        self._tensor_list_after_reduce = []  # include reduce tensor

        self._reduce_tensors = []
        self._broadcast_tensors = []
        self._vector_dup_tensors = []  # broadcast scalar in ub
        self._tensor_dst_tensor_map = {}  # {tensor->dst_tensor(next_tensor)}

        self._tiling_factor_vars = []

        self._spec_node_list = []
        self._is_last_axis_broadcast = False
        self._total_size = 0
        self._is_muti_output = False
        self._have_reduce = False
        self._max_ub_count = None

        self._need_multi_core = True

        self.block_split_axis = None

        # reduce_axis_map: key:reduce_axis_index, value:reduce_axis_var
        # reduce_index_map: key:reduce_axis_index in original index,
        #                   value:reduce_axis_index in reduce axis
        self._reduce_info = {"reduce_tensor": None,
                             "reduce_axis_map": {},
                             "reduce_axis_index": [],
                             "reduce_index_map": [],
                             "shape_before_reduce": None,
                             "shape_after_reduce": None,
                             "keep_dims": True,
                             "dtype": None}

        self._reduce_tiling_para = {
            "block_tiling": {"tiling_tensor": None, "axis": 0, "axis_var": None,
                             "factor": 1},
            "ub_tiling": [{"tiling_tensor": None, "axis": 0, "axis_var": None,
                           "factor": 1}]}

        self._reduce_tiling_result = {"block_tiling": {}, "ub_tiling": [{}]}

        self._storage_align_para = {}
        self._tiling_case_index = 0
        self._tiling_case_list = [{}]
        self._reduce_tiling_key_map = {}
        self._produce_atomic_sch = False
        self._atomic_tiling_case_list = []
        self._atomic_sch = None

    def init(self, out_tensors, spec_node_list):
        """
        :param out_tensors:
        :param spec_node_list:
        :return:
        """
        self._spec_node_list = spec_node_list
        self._out_tensors = copy.copy(out_tensors)

        is_success = self._construct_compute_graph(out_tensors, spec_node_list)
        if not is_success:
            return False

        self._calculate_tiling_cases()
        self._calculate_cache_read()
        self._calculate_cache_write()
        self._calculate_compute_inline()

        atomic_sch = ReduceAtomicSchedule()
        is_success = atomic_sch.init(out_tensors, [])
        if is_success:
            self._atomic_tiling_case_list = atomic_sch.get_tiling_case_list()
            self._produce_atomic_sch = True
            self._atomic_sch = atomic_sch

        self._add_compile_info()

        return True

    def do_schedule(self, outs, tiling_case):
        """
        auto_schedule for cce AI-CORE

        Parameters
        ----------
        outTensors : the out tvm.tensor

        sch_list : schedule, the computation schedule for the op

        spec_node_list : special node list

        Returns
        -------
        Bool, now is true

        """

        if tiling_case in range(0, len(self._tiling_case_list)):
            sch = self._do_reduce_schedule(tiling_case)
        elif tiling_case in range(len(self._tiling_case_list),
                                  len(self._tiling_case_list) + len(
                                      self._atomic_tiling_case_list)):
            sch = self._atomic_sch.do_schedule(outs, tiling_case - len(
                self._tiling_case_list))
        else:
            return None
        sch.tiling_key = tiling_case

        return sch

    def _do_reduce_schedule(self, tiling_case):

        self._schedule = tvm.create_schedule([self._res_tensor.op])

        self._schedule.disable_allocate(cceconf.scope_ubuf)

        self._select_tiling_case(tiling_case)

        self._do_cache_read()
        self._do_cache_write()
        self._do_compute_inline()

        self._do_storage_bound()

        self._calculate_tiling()
        self._do_tiling()

        self._do_reorder()

        self._do_set_constraint()

        self._caculate_storage_align()
        self._do_storage_align()

        self._calculate_multi_core()
        self._do_multi_core()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn()
        self._do_emit_insn()

        self._calculate_double_buffer()
        self._do_double_buffer()

        return self._schedule

    def check_support(self, out_tensors, spec_node_list):
        """
        :param out_tensors:
        :param spec_node_list:
        :return:
        """

        out_tensors = list(out_tensors) if isinstance(out_tensors, (list, tuple)) \
            else [out_tensors]

        # find the last out tensor
        mid_output_tensors_dst_tensor = {}
        last_output_tensor = None
        last_output_tensors = []
        if hasattr(out_tensors, 'index'):
            if len(out_tensors) > 1:
                self._is_muti_output = True
                self.get_dst_tensor_map(out_tensors,
                                        mid_output_tensors_dst_tensor)
                for out in out_tensors:
                    if out not in mid_output_tensors_dst_tensor.keys():
                        last_output_tensors.append(out)
                        if last_output_tensor is None:
                            last_output_tensor = out
            else:
                last_output_tensor = out_tensors[0]
                last_output_tensors.append(out_tensors[0])
        else:
            last_output_tensor = out_tensors
            last_output_tensors.append(out_tensors[0])

        self._last_output_tensors = last_output_tensors

        # record tensor list and tensor->dst_tensor(next_tensor) map
        visited_list = []
        tensor_list = []

        visited_list.append(last_output_tensor)
        tensor_list.append(last_output_tensor)
        self.__gen_reversed_subgraph_list(last_output_tensor, tensor_list,
                                          visited_list, spec_node_list)

        # tensor classification
        self._tensor_classify(out_tensors, tensor_list)

        is_supported = self._check_reduce_support()
        if not is_supported:
            return False

        reduce_tensor = self._reduce_tensors[0]
        self._res_tensor = self._last_output_tensors[0]
        self._record_reduce_info(reduce_tensor)

        _, tensor_list_after_reduce = \
            self._get_tensors_before_after_reduce(
                reduce_tensor, self._res_tensor, spec_node_list)

        is_supported = self._check_broadcast_support(tensor_list_after_reduce)
        if not is_supported:
            return False

        return True

    def get_tiling_cases(self):
        """
        :return:
        """

        return range(0, len(self._tiling_case_list) + len(
            self._atomic_tiling_case_list))

    def _select_tiling_case(self, index):
        """
        :param index:
        :return:
        """

        self._tiling_case_index = index

    def _calculate_tiling_cases(self):
        """
        :return:
        """

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        self._tiling_case_list.clear()
        if self._is_reduce_not_last_axis():
            self._gen_tiling_case_not_last_axis(shape_before_reduce,
                                                reduce_axis_index)
        else:
            self._gen_tiling_case_last_axis(shape_before_reduce,
                                            reduce_axis_index)
            if not self._tiling_case_list:
                self._gen_tiling_case_reduce_all(shape_before_reduce,
                                                 reduce_axis_index)

    def _add_compile_info(self):
        """
        :return:
        """
        for i in range(0, len(self._tiling_case_list)):
            self._reduce_tiling_key_map[i] = self._tiling_case_list[i]

        reduce_info = {}
        reduce_info["reduce_axis"] = self._reduce_info["reduce_axis_index"]
        reduce_info["keep_dims"] = self._reduce_info["keep_dims"]
        reduce_info["dtype"] = self._reduce_info["dtype"]
        reduce_info["out_dtype"] = self._res_tensor.dtype

        max_ub_count = self._get_max_ub_count()

        operation.add_compile_info("pattern", Pattern.REDUCE)
        operation.add_compile_info("reduce_tiling_key_map",
                                   self._reduce_tiling_key_map)
        operation.add_compile_info("reduce_info", reduce_info)
        operation.add_compile_info("max_ub_count", max_ub_count)
        operation.add_compile_info("core_num",
                                   cceconf.get_soc_spec("CORE_NUM"))

        if self._produce_atomic_sch:
            atomic_tiling_key_map = {}
            for i in range(0, len(self._atomic_tiling_case_list)):
                atomic_tiling_key_map[i + len(self._tiling_case_list)] = \
                    self._atomic_tiling_case_list[i]

            operation.add_compile_info("atomic_tiling_key_map",
                                       atomic_tiling_key_map)

    def _gen_tiling_case_not_last_axis(self, shape_before_reduce,
                                       reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        reordered_shape, reorder_to_orignal_axis_map, _ = \
            self._reorder_reduce_nlast_shape(shape_before_reduce,
                                             reduce_axis_index)
        is_keep_dims = self._reduce_info["keep_dims"]
        for i in range(0, len(reordered_shape)):
            orignal_axis = reorder_to_orignal_axis_map[i]
            if orignal_axis not in reduce_axis_index:
                if is_keep_dims:
                    block_split_axis = orignal_axis
                else:
                    none_reduce_index_map = self._find_none_reduce_axis_map(
                        shape_before_reduce,
                        reduce_axis_index)
                    block_split_axis = none_reduce_index_map[orignal_axis]

                for j in range(0, len(reordered_shape)):
                    orignal_axis = reorder_to_orignal_axis_map[j]
                    if orignal_axis not in reduce_axis_index and j < i:
                        continue
                    ub_split_axis = reorder_to_orignal_axis_map[j]
                    tiling_case = {"block_split_axis": block_split_axis,
                                   "block_factor": None,
                                   "ub_split_axis": ub_split_axis,
                                   "ub_factor": None}
                    self._tiling_case_list.append(tiling_case)

        dtype = self._reduce_info["dtype"]
        if self._need_special_tiling_case_not_last_axis(dtype,
                                                        shape_before_reduce,
                                                        reduce_axis_index):
            self._gen_special_tiling_case_not_last_axis(shape_before_reduce,
                                                        reduce_axis_index)

    def _need_special_tiling_case_not_last_axis(self, dtype,
                                                shape_before_reduce,
                                                reduce_axis_index):
        """
        :param dtype:
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # do not need special tiing:
        # (r1, a1), (...,a1) when a1 is larger than 32B
        r1_start_index, r1_end_index = self._find_last_reduce_axis(
            shape_before_reduce, reduce_axis_index)
        if r1_end_index is None or r1_start_index == 0:
            return False

        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
            shape_before_reduce, reduce_axis_index)
        if a1_start_index is None:
            return False

        return self._is_last_axis_less_than_32b(shape_before_reduce, dtype,
                                                a1_start_index, a1_end_index)

    def _is_last_axis_less_than_32b(self, shape, dtype, last_axis_start_index,
                                    last_axis_end_index):
        """
        :param shape:
        :param dtype:
        :param last_axis_start_index:
        :param last_axis_end_index:
        :return:
        """
        size = DTYPE_BYTE_MAPPING[dtype]
        for i in range(last_axis_start_index, last_axis_end_index + 1):
            if isinstance(shape[i], tvm.expr.Var):
                return False
            size = size * shape[i]

        return size < 32

    def _gen_special_tiling_case_not_last_axis(self, shape_before_reduce,
                                               reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """

        r1_start_index, r1_end_index = self._find_last_reduce_axis(
            shape_before_reduce, reduce_axis_index)

        if r1_end_index is None:
            return

        a2_end_index = r1_start_index - 1

        is_keep_dims = self._reduce_info["keep_dims"]
        if is_keep_dims:
            res_ub_split_axis = a2_end_index
        else:
            none_reduce_index_map = self._find_none_reduce_axis_map(
                shape_before_reduce,
                reduce_axis_index)
            res_ub_split_axis = none_reduce_index_map[a2_end_index]

        tiling_case = {"res_ub_split_axis": res_ub_split_axis,
                       "res_ub_factor": None,
                       "ub_split_axis": r1_end_index,
                       "ub_factor": None}

        self._tiling_case_list.append(tiling_case)

    def _gen_tiling_case_last_axis(self, shape_before_reduce,
                                   reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        reordered_shape, reorder_to_orignal_axis_map, _ = \
            self._reorder_reduce_last_shape(shape_before_reduce,
                                            reduce_axis_index)
        is_keep_dims = self._reduce_info["keep_dims"]
        for i in range(0, len(reordered_shape)):
            orignal_axis = reorder_to_orignal_axis_map[i]
            if orignal_axis not in reduce_axis_index:
                if is_keep_dims:
                    block_split_axis = orignal_axis
                else:
                    none_reduce_index_map = self._find_none_reduce_axis_map(
                        shape_before_reduce,
                        reduce_axis_index)
                    block_split_axis = none_reduce_index_map[orignal_axis]

                for j in range(i, len(reordered_shape)):
                    ub_split_axis = reorder_to_orignal_axis_map[j]
                    tiling_case = {"block_split_axis": block_split_axis,
                                   "block_factor": None,
                                   "ub_split_axis": ub_split_axis,
                                   "ub_factor": None}
                    self._tiling_case_list.append(tiling_case)

    def _gen_tiling_case_reduce_all(self, shape_before_reduce,
                                    reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        reordered_shape, reorder_to_orignal_axis_map, _ = \
            self._reorder_reduce_last_shape(shape_before_reduce,
                                            reduce_axis_index)
        block_split_axis = 0
        for i in range(0, len(reordered_shape)):
            ub_split_axis = reorder_to_orignal_axis_map[i]
            tiling_case = {"block_split_axis": block_split_axis,
                           "block_factor": None,
                           "ub_split_axis": ub_split_axis,
                           "ub_factor": None}
            self._tiling_case_list.append(tiling_case)
        self._need_multi_core = False

    # 'pylint: disable=too-many-locals
    def _construct_compute_graph(self, out_tensors, spec_node_list):
        """
        record relate context imformations of operations

        """
        # find the last out tensor
        mid_output_tensors_dst_tensor = {}
        last_output_tensor = None
        last_output_tensors = []
        if hasattr(out_tensors, 'index'):
            if len(out_tensors) > 1:
                self._is_muti_output = True
                self.get_dst_tensor_map(out_tensors,
                                        mid_output_tensors_dst_tensor)
                for out in out_tensors:
                    if out not in mid_output_tensors_dst_tensor.keys():
                        last_output_tensors.append(out)
                        if last_output_tensor is None:
                            last_output_tensor = out
            else:
                last_output_tensor = out_tensors[0]
                last_output_tensors.append(out_tensors[0])
        else:
            last_output_tensor = out_tensors
            last_output_tensors.append(out_tensors[0])

        self._last_output_tensors = last_output_tensors

        # record tensor list and tensor->dst_tensor(next_tensor) map
        visited_list = []
        tensor_list = []

        visited_list.append(last_output_tensor)
        tensor_list.append(last_output_tensor)
        self.__gen_reversed_subgraph_list(last_output_tensor, tensor_list,
                                          visited_list, spec_node_list)

        # tensor classification
        self._tensor_classify(out_tensors, tensor_list)

        is_supported = self._check_reduce_support()
        if not is_supported:
            return False

        reduce_tensor = self._reduce_tensors[0]
        self._res_tensor = self._last_output_tensors[0]
        self._record_reduce_info(reduce_tensor)

        tensor_list_before_reduce, tensor_list_after_reduce = \
            self._get_tensors_before_after_reduce(reduce_tensor,
                                                  self._res_tensor,
                                                  spec_node_list)

        is_supported = self._check_broadcast_support(tensor_list_after_reduce)
        if not is_supported:
            return False

        self._tensor_list_before_reduce = tensor_list_before_reduce
        self._tensor_list_after_reduce = tensor_list_after_reduce
        self._record_broadcast_info()
        # calculate cache_write_exclude_tensors
        for i in self._broadcast_not_last_axis_tensors:
            self._cache_write_exclude_tensors.append(i)

        # record info in order to calculate ub tiling
        for tensor in reversed(tensor_list):
            tmp_op = self.__split_tensor(tensor)
            if tmp_op["effective_op"]:
                self._op.append(tmp_op)
            self._origin_op.append(tmp_op)

        return True

    def _reorder_reduce_nlast_shape(self, shape_before_reduce,
                                    reduce_axis_index):
        """
        reorder shape (r4,a4,r3,a3,r2,a2,r1,a1) to (a4,a3,a2, r4,r3,r2,,r1,a1)
        :param shape_before_reduce: like (r4,a4,r3,a3,r2,a2,r1,a1)
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce: (ak+1,rk,..,r2,a2,r1,a1)
        # find the last none-reduce axis a1

        a1_start_index, _ = self._find_last_none_reduce_axis(
            shape_before_reduce, reduce_axis_index)

        last_none_reduce_axis = a1_start_index

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}
        #  (ak+1,ak,...,a2, rk,..,r2,,r1,a1)
        reordered_shape = list(shape_before_reduce)
        temp_axis = last_none_reduce_axis - 1
        for i in range(len(reduce_axis_index) - 1, -1, -1):
            reordered_shape[temp_axis] = shape_before_reduce[
                reduce_axis_index[i]]
            reorder_to_orignal_axis_map[temp_axis] = reduce_axis_index[i]
            orignal_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
            temp_axis = temp_axis - 1
        for i in range(last_none_reduce_axis - 1, -1, -1):
            if i not in reduce_axis_index:
                reordered_shape[temp_axis] = shape_before_reduce[i]
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis - 1

        for i in range(last_none_reduce_axis, len(shape_before_reduce)):
            reorder_to_orignal_axis_map[i] = i
            orignal_to_reorder_axis_map[i] = i

        return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map

    def _reorder_reduce_last_shape(self, shape_before_reduce,
                                   reduce_axis_index):
        """
        reorder shape (a4,r4,a3,r3,a2,r2,a1,r1) to (a4,a3,a2,a1,r4,r3,r2,,r1)
        :param shape_before_reduce: like (a4,r4,a3,r3,a2,r2,a1,r1)
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce: (a4,r4,a3,r3,a2,r2,a1,r1)

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}

        reordered_shape = []
        temp_axis = 0
        for i, ele in enumerate(shape_before_reduce):
            if i not in reduce_axis_index:
                reordered_shape.append(ele)
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis + 1

        for i, ele in enumerate(shape_before_reduce):
            if i in reduce_axis_index:
                reordered_shape.append(ele)
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis + 1

        return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map

    def _get_tensors_before_after_reduce(self, reduce_tensor, res_tensor,
                                         spec_node_list):
        """
        :param reduce_tensor:
        :param res_tensor:
        :param spec_node_list:
        :return:
        """

        visited_list = [reduce_tensor]
        tensor_list_before_reduce = []
        self.__gen_reversed_subgraph_list(reduce_tensor,
                                          tensor_list_before_reduce,
                                          visited_list, spec_node_list)
        tensor_list_after_reduce = []
        if res_tensor != reduce_tensor:
            visited_list = []
            self.__gen_reversed_subgraph_list(res_tensor,
                                              tensor_list_after_reduce,
                                              visited_list,
                                              [reduce_tensor])
        tensor_list_after_reduce.append(res_tensor)

        return tensor_list_before_reduce, tensor_list_after_reduce

    def _tensor_classify(self, out_tensors, tensor_list):
        """
        :param out_tensors:
        :param tensor_list:
        :return:
        """

        for tensor in tensor_list:
            if isinstance(tensor.op,
                          tvm.tensor.PlaceholderOp) or tensor in self._spec_node_list:
                self._input_tensors.append(tensor)
                if tensor in self._tensor_dst_tensor_map.keys():
                    self._input_tensor_dst_tensor_map[tensor] = \
                        self._tensor_dst_tensor_map[tensor]
            else:
                if tensor.op.tag.find("reduce") != -1:
                    self._reduce_tensors.append(tensor)
                if tensor.op.tag.find("broadcast") != -1:
                    if tensor.op.tag == "broadcast_for_tensor":
                        self._broadcast_tensors.append(tensor)
                    else:
                        self._vector_dup_tensors.append(tensor)
                if tensor in out_tensors:
                    if tensor in self._tensor_dst_tensor_map.keys():
                        self._mid_output_tensors.append(tensor)
                        self._mid_output_tensors_dst_tensor_map[tensor] = \
                            self._tensor_dst_tensor_map[tensor]
                        self._mid_tensors.append(tensor)
                else:
                    self._mid_tensors.append(tensor)

    def _is_reduce_not_last_axis(self):
        """
        :return:
        """

        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]

        return reduce_axis_index[-1] != len(shape_before_reduce) - 1

    def _check_broadcast_support(self, tensor_list_after_reduce):
        """
        :return: Bool
        """
        # broadcast axis must be a sub-set of reduce axis
        is_supported = self._check_broadcast_axis_support()
        if not is_supported:
            return False
        # do not support that broadcast tensor in tensor_list_after_reduce
        return self._check_broadcast_tensor_support(tensor_list_after_reduce)

    def _is_tuple_reduce_output(self):
        """
        :param output_tensors:
        :return:
        """

        if len(self._last_output_tensors) > 1:
            for tensor in self._last_output_tensors:
                if tensor.op.tag.find("tuple_reduce_sum") == -1:
                    return False
            return True

        return False

    def _check_reduce_support(self):
        """
        :return: Bool
        """
        if len(self._last_output_tensors) > 1:
            for tensor in self._last_output_tensors:
                if tensor.op.tag.find("tuple_reduce_sum") == -1:
                    return False

            for tensor in self._reduce_tensors:
                if tensor not in self._last_output_tensors:
                    return False
        else:
            return len(self._reduce_tensors) < 2
        return True

    def _check_broadcast_axis_support(self):
        """
        :return: Bool
        """
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        if not reduce_axis_index:
            return False
        broadcast_index = self._find_broadcast_axis_index()
        for ele in broadcast_index:
            if ele not in reduce_axis_index:
                return False

        return True

    def _check_broadcast_tensor_support(self, tensor_list_after_reduce):
        """
        :param tensor_list_after_reduce:
        :return:
        """

        # do not support that broadcast tensor in tensor_list_after_reduce
        for tensor in self._broadcast_tensors:
            if tensor in tensor_list_after_reduce:
                return False

        return True

    def _record_broadcast_info(self):
        """
        :return:
        """
        for tensor in self._broadcast_tensors:
            if self._is_broadcast_orignal_scalar(tensor):
                self._broadcast_scalars.append(tensor)
                if tensor in self._tensor_dst_tensor_map.keys():
                    dst_tensor = self._tensor_dst_tensor_map[tensor]
                    self._map_apend(self._broadcast_scalar_dst_tensor_map,
                                    tensor, dst_tensor)
            elif self._is_broadcast_not_last_axis_tensor(tensor):
                self._broadcast_not_last_axis_tensors.append(tensor)

    def _find_broadcast_axis_index(self):
        """
        :return:
        """
        index = []
        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        for tensor in self._broadcast_tensors:
            shape = self._shape_to_list(tensor.op.input_tensors[0].shape)
            for i, ele in enumerate(shape):
                # 'pylint: disable=unsubscriptable-object
                if ele != shape_before_reduce[i]:
                    if ele not in index:
                        index.append(i)

        return index

    def _is_broadcast_not_last_axis_tensor(self, tensor):
        """
        Check if the non-last axis broadcast scene

        Parameters:
        ----------
        tensors :  tensor to be checked

        Returns
        -------
        Bool: True or False
        """
        if tensor.op.tag == "broadcast_for_tensor":
            # broadcast not last axis
            if list(tensor.op.input_tensors):
                broadcast_before = self._shape_to_list(
                    tensor.op.input_tensors[0].shape)
                shape = self._shape_to_list(tensor.shape)
                for i in range(len(shape) - 1, -1, -1):
                    if shape[i] != 1:
                        return broadcast_before[i] == shape[i] and i != 0
        return False

    def _is_broadcast_orignal_scalar(self, tensor):
        """
        Check if the scaler broadcast scene

        Parameters:
        ----------
        tensors :  tensor to be checked

        Returns
        -------
        Bool: True or False
        """
        if tensor.op.tag == "broadcast_for_tensor":
            # broadcast scalar
            if list(tensor.op.input_tensors):
                shape = self._shape_to_list(tensor.op.input_tensors[0].shape)
                flag = True
                for i in range(0, len(shape), 1):
                    if shape[i] != 1:
                        flag = False
                        break
                return flag
        return False

    def __gen_reversed_subgraph_list(self, tensor, tensor_list, visited_list,
                                     spec_node_list):
        """traverse tensors by Depth-First-Search

        Parameters
        ----------
        tensor : tensor
            traverse tensors from this tensor,
            traversing its input tensors recursively.

        tensor_list : list
            record tensors in the order of Depth-First-Search.

        visited_list : list
            record tensors which has been visited.
        """
        for in_tensor in list(tensor.op.input_tensors):
            self._map_apend(self._tensor_dst_tensor_map, in_tensor,
                            tensor)
            if in_tensor not in visited_list:
                visited_list.append(in_tensor)
                tensor_list.append(in_tensor)
            if in_tensor in spec_node_list:
                continue

            self.__gen_reversed_subgraph_list(in_tensor, tensor_list,
                                              visited_list, spec_node_list)

    def _calculate_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._input_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i,
                            self._input_tensor_dst_tensor_map[i])

        for i in self._mid_output_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i,
                            self._mid_output_tensors_dst_tensor_map[i])

    def _calculate_cache_write(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._mid_tensors:
            if i not in self._cache_write_exclude_tensors:
                self._cache_write_tensors.append(i)

        if self._is_tuple_reduce_output():
            self._cache_write_tensors.append(self._last_output_tensors)
        else:
            if self._res_tensor not in self._cache_write_exclude_tensors:
                self._cache_write_tensors.append(self._res_tensor)



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
            if isinstance(i, list):
                self._cache_write_tensors_and_buffer_map[i[0]] = write_buffer[0]
            else:
                self._cache_write_tensors_and_buffer_map[i] = write_buffer

    def _is_reduce_all_axis(self, shape_before_reduce, reduce_axis_index):
        """
        :return:
        """
        # (1,1..,r1..rk,1,1)
        for i, _ in enumerate(shape_before_reduce):
            if i not in reduce_axis_index:
                if shape_before_reduce[i] != 1:
                    return False
        return True

    def _is_continuous_reduce(self, reduce_axis_index):
        """
        :param reduce_axis_index:
        :return:
        """
        for i, _ in enumerate(reduce_axis_index):
            if i > 0:
                if reduce_axis_index[i] != reduce_axis_index[i - 1] + 1:
                    return False
        return True

    def _support_storage_align(self):
        """
        :return:
        """
        # do not support mid output
        if self._mid_output_tensors:
            return False
        if not self._need_multi_core:
            return False
        if self._broadcast_tensors:
            return False
        return True

    def _need_storage_align(self):
        """
        :return:
        """
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]
        ub_tiling_para = ub_tiling_para_list[0]
        ub_split_axis = ub_tiling_para["axis"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        # for shape(r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, do not need storage_align
        if self._is_reduce_not_last_axis():
            a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if a1_end_index is None:
                return False
            if a1_start_index <= ub_split_axis <= a1_end_index:
                return False

        else:
            r1_start_index, r1_end_index = self._find_last_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if r1_end_index is None:
                return False
            # for shape(a4,r4,a3,r3,a2,r2,a1,r1), if ub split r1, do not need storage_align
            if r1_start_index <= ub_split_axis <= r1_end_index:
                return False

        return True

    def _caculate_storage_align(self):
        """
        :return:
        """
        self._storage_align_para.clear()

        if not self._need_storage_align():
            return

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
            shape_before_reduce,
            reduce_axis_index)

        if a1_end_index is None:
            return

        def _construct_storage_align_para(tensor_list, align_axis,
                                          mid_out_align_axis):
            """
            :param tensor_list:
            :param align_axis:
            :param align_factor:
            :param mid_out_align_axis:
            :return:
            """
            for i in self._cache_read_tensors_and_buffer_map:
                if i in tensor_list:
                    read_buffer = self._cache_read_tensors_and_buffer_map[i]
                    align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[
                        read_buffer.dtype]
                    para = {"align_axis_var": read_buffer.op.axis[align_axis],
                            "align_factor": align_factor,
                            "offset": 0
                            }
                    self._storage_align_para[read_buffer] = para
            for i in self._cache_write_tensors_and_buffer_map:
                if i in tensor_list:
                    write_buffer = self._cache_write_tensors_and_buffer_map[i]
                    align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[
                        write_buffer.dtype]
                    para = {"align_axis_var": write_buffer.op.axis[align_axis],
                            "align_factor": align_factor,
                            "offset": 0
                            }
                    self._storage_align_para[write_buffer] = para
            for tensor in self._mid_output_tensors:
                if tensor in tensor_list:
                    align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[
                        tensor.dtype]
                    para = {
                        "align_axis_var": tensor.op.axis[mid_out_align_axis],
                        "align_factor": align_factor,
                        "offset": 0
                    }
                    self._storage_align_para[tensor] = para

        def _reduce_tensor_storage_align_para(reduce_tensor, reduce_align_axis,
                                              mid_out_align_axis):
            """
            :param reduce_tensor:
            :param reduce_align_axis:
            :param mid_out_align_axis:
            :return:
            """

            if reduce_tensor in self._cache_write_tensors_and_buffer_map.keys():
                write_buffer = self._cache_write_tensors_and_buffer_map[
                    reduce_tensor]
                align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[
                    write_buffer.dtype]
                para = {
                    "align_axis_var": write_buffer.op.axis[reduce_align_axis],
                    "align_factor": align_factor,
                    "offset": 0
                }
                self._storage_align_para[write_buffer] = para

            for tensor in self._mid_output_tensors:
                if tensor == reduce_tensor and mid_out_align_axis:
                    align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[
                        tensor.dtype]
                    para = {
                        "align_axis_var": tensor.op.axis[mid_out_align_axis],
                        "align_factor": align_factor,
                        "offset": 0
                    }
                    self._storage_align_para[tensor] = para

        if self._is_reduce_not_last_axis():
            align_axis = a1_start_index - 1
            if align_axis < 0:
                align_axis = a1_end_index

            tensor_list_before_reduce = self._tensor_list_before_reduce
            _construct_storage_align_para(tensor_list_before_reduce, align_axis,
                                          align_axis)
            res_a1_start_index = a1_start_index
            is_keep_dims = self._reduce_info["keep_dims"]
            if not is_keep_dims:
                res_a1_start_index = a1_start_index - len(reduce_axis_index)

            if res_a1_start_index == 0:
                return
            res_align_axis = res_a1_start_index - 1

            reduce_tensor = self._reduce_info["reduce_tensor"]
            _reduce_tensor_storage_align_para(reduce_tensor,
                                              res_align_axis,
                                              res_align_axis)
            tensor_list_after_reduce = []
            # exclude reduce tensor
            for i in self._tensor_list_after_reduce:
                if i != reduce_tensor:
                    tensor_list_after_reduce.append(i)
            _construct_storage_align_para(tensor_list_after_reduce,
                                          res_align_axis,
                                          res_align_axis)
        else:
            align_axis = a1_end_index
            tensor_list_before_reduce = self._tensor_list_before_reduce
            _construct_storage_align_para(tensor_list_before_reduce, align_axis,
                                          align_axis)

    def _do_storage_align(self):
        """
        :param hape_before_reduce:
        :param reduce_axis_index:
        :return:
        """

        for stage in self._storage_align_para:
            scope_iter_var = self._storage_align_para[stage]["align_axis_var"]
            align_factor = self._storage_align_para[stage]["align_factor"]
            offset = self._storage_align_para[stage]["offset"]
            self._schedule[stage].storage_align(
                scope_iter_var, align_factor, offset)

    def _is_need_double_buffer(self, shape, block_start_axis, block_end_axis,
                               block_inner,
                               ub_split_axis, ub_inner):
        """
        :param shape:
        :param block_start_axis:
        :param block_end_axis:
        :param block_inner:
        :param ub_split_axis:
        :param ub_inner:
        :return:
        """

        loop = 1
        for i in range(0, block_start_axis):
            loop *= shape[i]
        if loop > 2:
            return True
        if block_end_axis == ub_split_axis:
            loop *= block_inner // ub_inner
        else:
            for i in range(block_end_axis + 1, ub_split_axis):
                loop *= shape[i]
            loop *= shape[ub_split_axis] // ub_inner
        if loop > 2:
            self._need_db = True
            return True
        return False

    @staticmethod
    def _find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find a1 position, a1 may contain continues axis
        a1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i not in reduce_axis_index:
                a1_end_index = i
                break
        a1_start_index = a1_end_index
        if a1_end_index is None:
            return a1_start_index, a1_end_index
        for i in range(a1_end_index, -1, -1):
            if i in reduce_axis_index:
                a1_start_index = i + 1
                break
            if i == 0:
                a1_start_index = i

        return a1_start_index, a1_end_index

    @staticmethod
    def _find_last_reduce_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find r1 position, r1 may contain continues axis
        r1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i in reduce_axis_index:
                r1_end_index = i
                break
        r1_start_index = r1_end_index
        if r1_end_index is None:
            return r1_start_index, r1_end_index
        for i in range(r1_end_index, -1, -1):
            if i not in reduce_axis_index:
                r1_start_index = i + 1
                break
            if i == 0:
                r1_start_index = i

        return r1_start_index, r1_end_index

    def _is_need_fused(self, shape, block_tiling_axis):
        """
        :param shape:
        :param block_tiling_axis:
        :return:
        """
        if block_tiling_axis == 0:
            return False
        for i in range(0, block_tiling_axis):
            if shape[i] != 1:
                return True
        return False


    def _do_storage_bound(self):
        """
        :return:
        """
        max_ub_count = self._get_max_ub_count()
        tensor_space = max_ub_count
        for tensor in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[tensor]
            self._schedule[read_buffer].set_storage_bound(tensor_space)

        for tensor in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            self._schedule[write_buffer].set_storage_bound(tensor_space)

        for tensor in self._mid_output_tensors:
            self._schedule[tensor].set_storage_bound(tensor_space)


    # 'pylint: disable=too-many-locals
    def _calculate_tiling(self):
        """
        calculate tiling strategy

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self._tiling_factor_vars.clear()

        tiling_case = self._tiling_case_list[self._tiling_case_index]

        if "block_split_axis" in tiling_case.keys():
            block_split_axis = tiling_case["block_split_axis"]
            block_factor = tiling_case["block_factor"]
        else:
            self._need_multi_core = False
            block_split_axis = tiling_case["res_ub_split_axis"]
            block_factor = tiling_case["res_ub_factor"]

        ub_split_axis = tiling_case["ub_split_axis"]
        ub_factor = tiling_case["ub_factor"]

        if block_factor is None:

            block_inner = operation.var("block_factor", (1, None))
            self._tiling_factor_vars.append(block_inner)
        else:
            block_inner = block_factor

        if ub_factor is None:
            ub_inner = operation.var("ub_factor", (1, None))
            self._tiling_factor_vars.append(ub_inner)
        else:
            ub_inner = ub_factor

        reduce_tensor = self._reduce_info["reduce_tensor"]
        reduce_ub_buffer = self._cache_write_tensors_and_buffer_map[
            reduce_tensor]

        res_tensor = self._res_tensor
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        block_tiling_para = {"tiling_tensor": res_tensor,
                             "axis": block_split_axis, "axis_var": None,
                             "factor": block_inner}

        if ub_split_axis in reduce_axis_index:
            reduce_axis = self._reduce_info["reduce_axis_map"]
            axis_var = reduce_axis[ub_split_axis]
            ub_tiling_para = [
                {"tiling_tensor": reduce_ub_buffer, "axis": ub_split_axis,
                 "axis_var": axis_var, "factor": ub_inner}]
        else:
            ub_tiling_para = [
                {"tiling_tensor": reduce_ub_buffer, "axis": ub_split_axis,
                 "axis_var": None, "factor": ub_inner}]

        self._reduce_tiling_para["block_tiling"] = block_tiling_para
        self._reduce_tiling_para["ub_tiling"] = ub_tiling_para

    def _do_tiling(self):
        """
        :return:
        """
        self._do_block_tiling()
        self._do_ub_tiling()

    def _do_block_tiling(self):
        """
        :return:
        """
        block_tiling_para = self._reduce_tiling_para["block_tiling"]
        block_tiling_tensor = block_tiling_para["tiling_tensor"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner = block_tiling_para["factor"]

        if block_split_axis < 0:
            raise RuntimeError("Should use positive number to represent axis!")

        if "axis_var" in block_tiling_para.keys() and \
                block_tiling_para["axis_var"] is not None:
            axis_var = block_tiling_para["axis_var"]
        else:
            axis_var = block_tiling_tensor.op.axis[block_split_axis]

        res_block_outer, res_block_inner = \
            self._schedule[block_tiling_tensor].split(axis_var,
                                                      factor=block_split_inner)
        block_tiling_result = {"tiling_tensor": block_tiling_tensor,
                               "axis": block_split_axis,
                               "parent_itervar": axis_var,
                               "outer_itervar": res_block_outer,
                               "inner_itervar": res_block_inner}
        self._reduce_tiling_result["block_tiling"] = block_tiling_result

    # 'pylint: disable=too-many-locals
    def _do_ub_tiling(self):
        """
        :return:
        """
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_tiling_tensor = block_tiling_result["tiling_tensor"]
        block_split_axis = block_tiling_result["axis"]
        res_block_inner = block_tiling_result["inner_itervar"]

        ub_tiling_result_list = []
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]

        ub_tiling_para = ub_tiling_para_list[0]
        ub_tiling_tensor = ub_tiling_para["tiling_tensor"]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        if ub_split_axis < 0:
            raise RuntimeError("Should use positive number to represent axis!")

        if ub_tiling_tensor is not None:
            if block_tiling_tensor is not None and block_split_axis == ub_split_axis \
                    and ub_tiling_tensor == block_tiling_tensor:
                res_ub_outer, res_ub_inner = self._schedule[
                    ub_tiling_tensor].split(res_block_inner,
                                            factor=ub_split_inner)
                ub_tiling_result = {"tiling_tensor": ub_tiling_tensor,
                                    "axis": ub_split_axis,
                                    "parent_itervar": res_block_inner,
                                    "outer_itervar": res_ub_outer,
                                    "inner_itervar": res_ub_inner}
            else:
                # if the axis_var is not empty,
                # the axis_var is used as the split parameter first,
                # otherwise the split_axis of the tilting_tensor is used as
                # the split parameter
                if "axis_var" in ub_tiling_para.keys() and \
                        ub_tiling_para["axis_var"] is not None:
                    axis_var = ub_tiling_para["axis_var"]
                else:
                    axis_var = self._get_axis_var(ub_split_axis,
                                                  ub_tiling_tensor)

                res_ub_outer, res_ub_inner = self._schedule[
                    ub_tiling_tensor].split(axis_var, factor=ub_split_inner)

                ub_tiling_result = {"tiling_tensor": ub_tiling_tensor,
                                    "axis": ub_split_axis,
                                    "parent_itervar": axis_var,
                                    "outer_itervar": res_ub_outer,
                                    "inner_itervar": res_ub_inner}
            ub_tiling_result_list.append(ub_tiling_result)

        self._reduce_tiling_result["ub_tiling"] = ub_tiling_result_list

    def _get_axis_var(self, axis, tensor):
        """
        :param axis:
        :param tensor:
        :return:
        """

        if self._is_reduce_tensor(tensor):
            reduce_axis_index = self._reduce_info["reduce_axis_index"]
            shape = self._reduce_info["shape_before_reduce"]
            if axis >= len(shape):
                raise RuntimeError("Axis index out of range!")
            if axis in reduce_axis_index:
                reduce_axis_map = self._reduce_info["reduce_axis_map"]
                return reduce_axis_map[axis]
            else:
                is_keep_dims = self._reduce_info["keep_dims"]
                if is_keep_dims:
                    return tensor.op.axis[axis]
                else:
                    none_reduce_index_map = self._find_none_reduce_axis_map(
                        shape,
                        reduce_axis_index)
                    return tensor.op.axis[none_reduce_index_map[axis]]

        else:
            if axis >= len(tensor.op.axis):
                raise RuntimeError("Axis index out of range!")
            return tensor.op.axis[axis]

    def _do_reorder(self):
        """
        :return:
        """

        if self._is_reduce_not_last_axis():
            self._reorder_reduce_not_last_axis()
        else:
            self._reorder_reduce_last_axis()

    def _do_set_constraint(self):
        """
        :return:
        """
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]
        ub_tiling_para = ub_tiling_para_list[0]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        shape_before_reduce = self._reduce_info["shape_before_reduce_expr"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        max_ub_count = self._get_max_ub_count()

        if self._is_reduce_not_last_axis():
            reordered_shape, _, orignal_to_reorder_axis_map = \
                self._reorder_reduce_nlast_shape(shape_before_reduce,
                                                 reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            self._schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                self._schedule.set_constraint(
                    reordered_shape[i] <= max_ub_count)

        else:
            reordered_shape, _, orignal_to_reorder_axis_map = \
                self._reorder_reduce_last_shape(shape_before_reduce,
                                                reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            self._schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis + 1, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                self._schedule.set_constraint(
                    reordered_shape[i] <= max_ub_count)

        self._schedule.set_constraint(shape_in_ub <= max_ub_count)

    def _reorder_reduce_last_axis(self):
        """
        :return:
        """
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]

        # reorder tensor before reduce,
        # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
        # the orignal ir is(a4,r4,a3,r3,a2,r2,a1,r1),
        # reorder orignal ir to (a4,a3,a2,a1,r4,r3,r2,r1)

        tensor_list_before_reduce = self._tensor_list_before_reduce
        tensor_list = tensor_list_before_reduce

        for tensor in self._cache_read_tensors_and_buffer_map:
            if tensor in tensor_list:
                read_buffer = self._cache_read_tensors_and_buffer_map[tensor]
                reordered_axis_list = []
                for i in range(0, len(shape_before_reduce)):
                    if i not in reduce_axis_index:
                        reordered_axis_list.append(read_buffer.op.axis[i])

                for i in range(0, len(shape_before_reduce)):
                    if i in reduce_axis_index:
                        reordered_axis_list.append(read_buffer.op.axis[i])

                self._schedule[read_buffer].reorder(*(reordered_axis_list))

        for tensor in self._cache_write_tensors_and_buffer_map:
            if tensor in tensor_list:
                write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
                reordered_axis_list = []
                for i in range(0, len(shape_before_reduce)):
                    if i not in reduce_axis_index:
                        reordered_axis_list.append(write_buffer.op.axis[i])
                self._schedule[write_buffer].reorder(*(reordered_axis_list))

        for tensor in self._mid_output_tensors:
            if tensor in tensor_list:
                reordered_axis_list = []
                for i in range(0, len(shape_before_reduce)):
                    if i not in reduce_axis_index:
                        reordered_axis_list.append(tensor.op.axis[i])
                self._schedule[tensor].reorder(*(reordered_axis_list))

    def __reorder_reduce_not_last_axis_reduce_ub(self, a1_start_index, a1_end_index):
        """
        :param a1_start_index:
        :param a1_end_index:
        :return:
        """

        reduce_tensor = self._reduce_info["reduce_tensor"]
        reduce_ub_buffer = self._cache_write_tensors_and_buffer_map[
            reduce_tensor]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        ub_outer = ub_tiling_result["outer_itervar"]
        ub_inner = ub_tiling_result["inner_itervar"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]


        is_keep_dims = self._reduce_info["keep_dims"]
        reduce_a1_start_index = a1_start_index
        reduce_a1_end_index = a1_end_index

        if not is_keep_dims:
            reduce_a1_start_index = a1_start_index - len(reduce_axis_index)
            reduce_a1_end_index = a1_end_index - len(reduce_axis_index)

        reduce_ub_buffer_reordered_axis_list = []
        # for shape(r4,a4,r3,a3,r2,a2,r1,a1), the orignal ir is
        # (a4,a3,a2,a1,r4,r3,r2,r1)
        # reorder orignal ir to (a4,a3,a2,r4,r3,r2,r1,a1) if ub do not split a1

        if ub_split_axis < a1_start_index or ub_split_axis > a1_end_index:
            for i, ele in enumerate(reduce_axis_index):
                if ub_split_axis != ele:
                    reduce_ub_buffer_reordered_axis_list.append(
                        reduce_ub_buffer.op.reduce_axis[i])
                else:
                    reduce_ub_buffer_reordered_axis_list.append(ub_outer)
                    reduce_ub_buffer_reordered_axis_list.append(ub_inner)

            is_keep_dims = self._reduce_info["keep_dims"]
            reduce_a1_start_index = a1_start_index
            reduce_a1_end_index = a1_end_index

            if not is_keep_dims:
                reduce_a1_start_index = a1_start_index - len(reduce_axis_index)
                reduce_a1_end_index = a1_end_index - len(reduce_axis_index)

            for i in range(reduce_a1_start_index, reduce_a1_end_index + 1):
                reduce_ub_buffer_reordered_axis_list.append(
                    reduce_ub_buffer.op.axis[i])
        else:
            # for shape(r4,a4,r3,a3,r2,a2,r1,a1), the orignal ir is
            # (a4,a3,a2,a1_outer,a1_inner, r4,r3,r2,r1),
            # reorder orignal ir to (a4,a3,a2,a1_outer, r4,r3,r2,r1,a1_inner)
            # if ub split a1
            for i, ele in enumerate(reduce_axis_index):
                reduce_ub_buffer_reordered_axis_list.append(
                    reduce_ub_buffer.op.reduce_axis[i])
            # when a1 is continous axis
            reduce_ub_buffer_reordered_axis_list.append(ub_inner)

            for i in range(ub_split_axis + 1, a1_end_index + 1):
                index = i
                if not is_keep_dims:
                    index = i - len(reduce_axis_index)
                reduce_ub_buffer_reordered_axis_list.append(reduce_ub_buffer.op.axis[index])

        self._schedule[reduce_ub_buffer].reorder(
            *(reduce_ub_buffer_reordered_axis_list))

    def __reorder_reduce_not_last_axis_before_reduce(self, a1_start_index):

        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        # reorder tensor before reduce,
        # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
        # the orignal ir is (r4,a4,r3,a3,r2,a2,r1,a1),
        # reorder orignal ir to (a4,a3,a2,r4,r3,r2,r1,a1)
        tensor_list_before_reduce = self._tensor_list_before_reduce
        tensor_list = tensor_list_before_reduce

        def __get_reorder_list(tensor):
            """
            :param tensor:
            :return:
            """
            reordered_axis_list = []
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    reordered_axis_list.append(tensor.op.axis[i])

            for i in reduce_axis_index:
                reordered_axis_list.append(tensor.op.axis[i])

            return reordered_axis_list

        for tensor in self._cache_read_tensors_and_buffer_map:
            if tensor in tensor_list:
                read_buffer = self._cache_read_tensors_and_buffer_map[tensor]
                reordered_axis_list = __get_reorder_list(read_buffer)
                self._schedule[read_buffer].reorder(*(reordered_axis_list))

        for tensor in self._cache_write_tensors_and_buffer_map:
            if tensor in tensor_list:
                write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
                reordered_axis_list = __get_reorder_list(write_buffer)
                self._schedule[write_buffer].reorder(*(reordered_axis_list))

        for tensor in self._mid_output_tensors:
            if tensor in tensor_list:
                reordered_axis_list = __get_reorder_list(tensor)
                self._schedule[tensor].reorder(*(reordered_axis_list))

    def _reorder_reduce_not_last_axis(self):
        """
        :return:
        """
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]

        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
            shape_before_reduce,
            reduce_axis_index)

        if a1_end_index is None:
            raise RuntimeError("a1_end_index can not be none!")

        self.__reorder_reduce_not_last_axis_reduce_ub(a1_start_index,
                                                      a1_end_index)

        self.__reorder_reduce_not_last_axis_before_reduce(a1_start_index)


    @staticmethod
    def _find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        none_reduce_index_map = {}
        count = 0
        for i in range(0, len(shape_before_reduce)):
            if i not in reduce_axis_index:
                none_reduce_index_map[i] = count
                count += 1
        return none_reduce_index_map

    def _calculate_compute_inline(self):
        """
        Calculate the tensor that needs compute inline

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        for i in self._mid_tensors:
            if i not in self._mid_output_tensors:
                self._compute_inline_tensors.append(i)

    def _calculate_multi_core(self):
        """
        Calculate fuse and bind axis of multicore

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        if self._need_multi_core:
            block_tiling_result = self._reduce_tiling_result["block_tiling"]
            tensor = block_tiling_result["tiling_tensor"]
            block_split_axis = block_tiling_result["axis"]
            res_block_outer = block_tiling_result["outer_itervar"]

            need_fuse_list = [res_block_outer]
            for i in range(block_split_axis - 1, -1, -1):
                # if i not in reduce_axis_index:
                need_fuse_list.append(tensor.op.axis[i])
            fused_axis = need_fuse_list[0]
            for i in range(1, len(need_fuse_list)):
                fused_axis = self._schedule[tensor].fuse(fused_axis,
                                                         need_fuse_list[i])

            self._multi_core_fused_axis = fused_axis
            self._multi_core_bind_tensor = tensor

            # to update block tiling result
            block_tiling_result["outer_itervar"] = fused_axis
            self._reduce_tiling_result["block_tiling"] = block_tiling_result

    def _calculate_compute_at(self):
        """
        Calculate the tensor that needs compute at

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """

        self._compute_at_map.clear()

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]

        reduce_ub_tiling_result = ub_tiling_result_list[0]
        reduce_ub_tiling_tensor = reduce_ub_tiling_result["tiling_tensor"]
        reduce_ub_outer = reduce_ub_tiling_result["outer_itervar"]

        def _construct_compute_at_para(parent_tenosr, ub_outer, tensor_list):
            """
            :param parent_tenosr:
            :param ub_outer:
            :param tensor_list:
            :return:
            """
            for i in self._cache_read_tensors_and_buffer_map:
                if i in tensor_list:
                    read_buffer = self._cache_read_tensors_and_buffer_map[i]
                    para = {"parent": self._schedule[parent_tenosr],
                            "scope": ub_outer}
                    self._compute_at_map[read_buffer] = para
            for i in self._cache_write_tensors_and_buffer_map:
                if i in tensor_list:
                    write_buffer = self._cache_write_tensors_and_buffer_map[i]
                    para = {"parent": self._schedule[parent_tenosr],
                            "scope": ub_outer}
                    self._compute_at_map[write_buffer] = para
            for i in self._mid_output_tensors:
                if i in tensor_list:
                    para = {"parent": self._schedule[parent_tenosr],
                            "scope": ub_outer}
                    self._compute_at_map[i] = para

        if self._is_reduce_not_last_axis():
            ub_split_axis = reduce_ub_tiling_result["axis"]
            shape_before_reduce = self._reduce_info["shape_before_reduce"]
            # for shape (r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, compute at r1
            # when a1 is continous,
            reduce_axis_index = self._reduce_info["reduce_axis_index"]
            shape_before_reduce = self._reduce_info["shape_before_reduce"]
            a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if a1_end_index is None:
                raise RuntimeError("a1_end_index can not be none!")

            if a1_start_index <= ub_split_axis <= a1_end_index:
                reduce_ub_outer = reduce_ub_tiling_tensor.op.reduce_axis[-1]

        tensor_list_before_reduce = self._tensor_list_before_reduce
        _construct_compute_at_para(reduce_ub_tiling_tensor, reduce_ub_outer,
                                   tensor_list_before_reduce)

        if len(ub_tiling_result_list) > 1:
            res_tiling_result = ub_tiling_result_list[1]
        else:
            res_tiling_result = self._reduce_tiling_result["block_tiling"]

        res_ub_tiling_tensor = res_tiling_result["tiling_tensor"]
        res_ub_outer = res_tiling_result["outer_itervar"]

        tensor_list_after_reduce = self._tensor_list_after_reduce
        _construct_compute_at_para(res_ub_tiling_tensor, res_ub_outer,
                                   tensor_list_after_reduce)

    def _calculate_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        if self._is_reduce_not_last_axis():
            self._need_db = True
        else:
            self._need_db = False

    def _calculate_emit_insn(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        self._emit_insn_map.clear()

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]

        def get_insn(tensor_):
            tag = tensor_.op.tag
            if tensor_.op.tag.find("|") != -1:
                insn = tag.split("|")[0]
            else:
                insn = tag
            return INSN_MAPPING.get(insn, insn)

        def _construct_emit_insn_para(tensor_list):
            """
            :param tensor_list:
            :return:
            """
            ub_tiling_result = ub_tiling_result_list[0]
            ub_split_axis = ub_tiling_result["axis"]
            for i in self._cache_read_tensors_and_buffer_map:
                if i in tensor_list:
                    read_buffer = self._cache_read_tensors_and_buffer_map[i]
                    para = {"scope": read_buffer.op.axis[ub_split_axis],
                            "instruction": 'dma_copy'}
                    self._emit_insn_map[read_buffer] = para

            for i in self._cache_write_tensors_and_buffer_map:
                if i in tensor_list:
                    write_buffer = self._cache_write_tensors_and_buffer_map[i]
                    insn = get_insn(write_buffer)
                    para = {"scope": write_buffer.op.axis[0],
                            "instruction": insn}
                    self._emit_insn_map[write_buffer] = para

            for out_tensor in self._mid_output_tensors:
                if out_tensor in tensor_list:
                    para = {"scope": out_tensor.op.axis[0],
                            "instruction": 'dma_copy'}
                    self._emit_insn_map[out_tensor] = para

        def _tensors_before_reduce_emit_insn_para():
            """
            :return:
            """
            tensor_list_before_reduce = self._tensor_list_before_reduce
            _construct_emit_insn_para(tensor_list_before_reduce)

        def _reduce_ub_emit_insn_para():
            """
            :return:
            """
            ub_tiling_result = ub_tiling_result_list[0]
            reduce_ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
            reduce_ub_inner = ub_tiling_result["inner_itervar"]

            if self._is_reduce_not_last_axis():
                ub_split_axis = ub_tiling_result["axis"]
                shape_before_reduce = self._reduce_info["shape_before_reduce"]
                reduce_axis_index = self._reduce_info["reduce_axis_index"]
                # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
                # the ir order (a4,a3,a2,r4,r3,r2,r1,a1)
                # if ub split a2,a3 or a4, emit insn should target at r4
                # when a1 is continous
                reduce_axis_index = self._reduce_info["reduce_axis_index"]
                shape_before_reduce = self._reduce_info["shape_before_reduce"]
                a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                    shape_before_reduce,
                    reduce_axis_index)
                if a1_end_index is None:
                    raise RuntimeError("a1_end_index can not be none!")
                if ub_split_axis < a1_start_index and\
                        ub_split_axis not in reduce_axis_index:
                    reduce_ub_inner = reduce_ub_tiling_tensor.op.reduce_axis[0]
            else:
                reduce_axis_index = self._reduce_info["reduce_axis_index"]
                ub_split_axis = ub_tiling_result["axis"]
                # ub cut ak (none reduce axis),
                if ub_split_axis not in reduce_axis_index:
                    reduce_ub_inner = reduce_ub_tiling_tensor.op.reduce_axis[0]

            insn = get_insn(reduce_ub_tiling_tensor)
            if insn == "vector_reduce_prod":
                if self._is_reduce_not_last_axis():
                    insn = "vector_mul"

            para = {"scope": reduce_ub_inner,
                    "instruction": insn}
            self._emit_insn_map[reduce_ub_tiling_tensor] = para

        def _tensors_after_reduce_emit_insn_para():
            """
            :return:
            """
            tensor_list_after_reduce = []
            # exclude reduce tensor
            reduce_tensor = self._reduce_info["reduce_tensor"]
            for i in self._tensor_list_after_reduce:
                if i != reduce_tensor:
                    tensor_list_after_reduce.append(i)

            _construct_emit_insn_para(tensor_list_after_reduce)

        def _res_tensor_emit_insn_para():
            """
            :return:
            """
            if len(ub_tiling_result_list) > 1:
                res_tiling_result = ub_tiling_result_list[1]
            else:
                res_tiling_result = self._reduce_tiling_result["block_tiling"]

            res_ub_tiling_tensor = res_tiling_result["tiling_tensor"]
            res_ub_inner = res_tiling_result["inner_itervar"]

            para = {"scope": res_ub_inner,
                    "instruction": 'dma_copy'}
            self._emit_insn_map[res_ub_tiling_tensor] = para

        _tensors_before_reduce_emit_insn_para()
        _reduce_ub_emit_insn_para()
        _tensors_after_reduce_emit_insn_para()
        _res_tensor_emit_insn_para()

    def _calculate_emit_insn_map(self, tensor):
        """
        Get the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        Instruction map string
        """
        if tensor.op.tag.find("|") != -1:
            str_list = tensor.op.tag.split("|")
            insn = self._insn_map.get(str_list[0])
            if insn and self._check_cast_support(tensor):
                return insn
            insn = self._reg_insn_map.get(str_list[0])
        else:
            insn = self._insn_map.get(tensor.op.tag)
            if insn and self._check_cast_support(tensor):
                return insn
            insn = self._reg_insn_map.get(tensor.op.tag)

        return insn

    def _check_cast_support(self, tensor):
        """
        Judge if tensor supports cast instruction operations

        Parameters:
        ----------
        tensors :  input tensor

        Returns
        -------
        Bool: True or False
        """
        cache_buffer = tensor
        read_buffer = tensor.op.input_tensors[0]
        if read_buffer.dtype == "int32":
            if cache_buffer.dtype == "float16" or \
                    cache_buffer.dtype == "float32":
                return False
        return True

    def _is_reduce_tensor(self, tensor):
        """
        :param tensor:
        :return:
        """
        return tensor.op.tag.find("reduce") != -1

    def _record_reduce_info(self, tensor):
        """
        :param tensor:
        :return:
        """
        if self._is_reduce_tensor(tensor):
            self._reduce_info["reduce_tensor"] = tensor
            tensor_op = tensor.op
            reduce_axis_var = []
            for i in tensor_op.reduce_axis:
                reduce_axis_var.append(i)
            data_axis_var = tensor_op.body[0].source[0].args
            for ax_item in reduce_axis_var:
                for index in range(0, len(data_axis_var), 1):
                    if data_axis_var[index].same_as(ax_item.var):
                        self._reduce_info["reduce_axis_index"].append(index)
                        self._reduce_info["reduce_axis_map"][index] = ax_item

            self._reduce_info["reduce_axis_index"].sort()
            tmp_reduce_axis_num = self._reduce_info["reduce_axis_index"]
            reduce_index_map = {}
            for i, ele in enumerate(tmp_reduce_axis_num):
                reduce_index_map[ele] = i

            self._reduce_info["reduce_index_map"] = reduce_index_map
            if tensor.op.input_tensors:
                shape_before_reduce_expr = tensor.op.input_tensors[0].shape
                shape_before_reduce = self._shape_to_list(
                    shape_before_reduce_expr)
                self._reduce_info["shape_before_reduce"] = shape_before_reduce
                self._reduce_info[
                    "shape_before_reduce_expr"] = shape_before_reduce_expr

            self._reduce_info["shape_after_reduce"] = tensor.shape
            self._reduce_info["dtype"] = tensor.dtype

            is_keep_dims = len(self._reduce_info["shape_before_reduce"]) == len(
                tensor.shape)
            self._reduce_info["keep_dims"] = is_keep_dims

    def _get_max_ub_count(self):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        if self._max_ub_count:
            return self._max_ub_count
        self._total_size = cceconf.get_soc_spec("UB_SIZE")
        self._total_size = self._total_size // 2  # div 2 for double buffer

        total_width = self._get_total_width()
        if not total_width:
            raise RuntimeError("Can not calculate with no compute")

        max_bound = total_width * 128
        max_ub_count = int(self._total_size // max_bound * 128)

        self._max_ub_count = max_ub_count

        return max_ub_count

    def _get_total_width(self):
        """
        caculate the max useable number based on op liveness
        return: max useable number
        """
        res = self._res_tensor

        def _post_dfs_order(op_node, op_graph, visited, post_order):
            if op_node in visited:
                return
            visited[op_node] = True
            post_order.append(op_node)
            if op_node in op_graph:
                for src in op_graph[op_node]:
                    _post_dfs_order(src, op_graph, visited, post_order)

        def _op_width(op_node):
            num_type = op_node.dtype
            if num_type.lower() not in DTYPE_BYTE_MAPPING.keys():
                raise RuntimeError("Can not calculate with no compute")
            tmp_width = 0
            if op_node.op.tag is not None:
                tag = op_node.op.tag
                # logic use 4 fp16 temp buffer
                if tag.find("logic") != -1:
                    tmp_width = 4 * DTYPE_BYTE_MAPPING["float16"]
                # cond use 3 fp16 temp buffer
                elif tag.find("cond") != -1:
                    tmp_width = 3 * DTYPE_BYTE_MAPPING["float16"]
                # vsel use 3 fp16 temp buffer
                elif tag.find("sel") != -1:
                    tmp_width = 3 * DTYPE_BYTE_MAPPING["float16"]
                # vcompare use 2 temp buffer
                elif tag.find("compare") != -1:
                    tmp_width = 2 * DTYPE_BYTE_MAPPING[num_type.lower()]
                # vcomsel use 3 temp buffer
                elif tag.find("cmpsel") != -1:
                    tmp_width = 3 * DTYPE_BYTE_MAPPING[num_type.lower()]

            return DTYPE_BYTE_MAPPING[num_type.lower()] + tmp_width

        op_graph = {}
        for op_node in self._origin_op:
            src_op = list(op_node['src_buffer'])
            src_op.reverse()
            op_graph[op_node['dst_buffer']] = src_op
        visited = {}
        post_order = []
        _post_dfs_order(res, op_graph, visited, post_order)
        lives = [res]
        live_width = _op_width(lives[0])
        max_width = live_width
        visited = {lives[0]: True}
        for op_node in post_order:
            if op_node in op_graph:
                for src in op_graph[op_node]:
                    if src in visited:
                        continue
                    lives.append(src)
                    live_width += _op_width(src)
                    visited[src] = True
                if live_width > max_width:
                    max_width = live_width
            lives.remove(op_node)
            live_width -= _op_width(op_node)
        # for tuple sum
        if self._is_tuple_reduce_output():
            max_width += _op_width(res)

        return max_width

    def __split_tensor_elewise_single(self, tmp_op, op_node):
        """
        :param tmp_op:
        :param op_node:
        :return:
        """
        if hasattr(op_node.body[0], 'b'):
            if isinstance(op_node.body[0].a, tvm.expr.Call):
                tmp_op["args"] = [op_node.body[0].b]
            else:
                tmp_op["args"] = [op_node.body[0].a]

    def __split_tensor_elewise_binary_compare(self, tmp_op, op_node):
        """
        :param tmp_op:
        :param op_node:
        :return:
        """
        if hasattr(op_node.body[0], 'condition'):
            tmp_op["args"] = [op_node.body[0].condition.b]
        if tmp_op["op"].find("lt") != -1:
            tmp_op["args"].append("lt")
        elif tmp_op["op"].find("gt") != -1:
            tmp_op["args"].append("gt")

    def __split_tensor_elewise_binary_scalar(self, tmp_op, op_node):
        """
        :param tmp_op:
        :param op_node:
        :return:
        """
        if hasattr(op_node.body[0], 'a'):
            if isinstance(op_node.body[0].a, tvm.expr.Call):
                if hasattr(op_node.body[0].b, 'a'):
                    if isinstance(op_node.body[0].b.a, tvm.expr.Call):
                        tmp_op["args"] = [op_node.body[0].b.b]
                    else:
                        tmp_op["args"] = [op_node.body[0].b.a]
            else:
                if hasattr(op_node.body[0].a, 'a'):
                    if isinstance(op_node.body[0].a.a, tvm.expr.Call):
                        tmp_op["args"] = [op_node.body[0].a.b]
                    else:
                        tmp_op["args"] = [op_node.body[0].a.a]

    def __split_tensor_broadcast(self, tmp_op, op_node):
        """
        :param tmp_op:
        :param op_node:
        :return:
        """
        if tmp_op["op"] == "broadcast_for_tensor":
            # broadcast not last axis
            if self._shape_to_list(tmp_op["src_buffer"][0].shape)[-1] != 1:
                tmp_op["effective_op"] = False
        else:
            tmp_op["args"] = [op_node.body[0]]

    def __split_tensor_reduce(self, tmp_op, op_node):
        """
        :param tmp_op:
        :param op_node:
        :return:
        """
        self._have_reduce = True
        tmp_op["reduce_axis"] = list(op_node.reduce_axis)
        reduce_axis_var = []
        for i in op_node.reduce_axis:
            reduce_axis_var.append(i.var)
        data_axis_var = op_node.body[0].source[0].args
        tmp_op["reduce_axis_num"] = []
        for axis in reduce_axis_var:
            axis_num = 0
            for i in data_axis_var:
                if i.same_as(axis):
                    tmp_op["reduce_axis_num"].append(axis_num)
                axis_num += 1

    def __split_tensor_others(self, tmp_op, op_node):
        """
        :param tmp_op:
        :param op_node:
        :return:
        """

        if tmp_op["op"].find("elewise_single_VS_cond") != -1 \
                or tmp_op["op"].find("elewise_binary_cmp") != -1 \
                or tmp_op["op"].find("elewise_binary_cmpsel") != -1 \
                or tmp_op["op"].find("elewise_binary_logic") != -1:
            str_list = op_node.tag.split("|")
            tmp_op["op"] = str_list[0]
            tmp_op["args"] = []
            for i in range(1, len(str_list)):
                tmp_op["args"].append(str_list[i])

        # split inputs sign and add into args for elewise_multiple op
        elif tmp_op["op"].find("elewise_multiple") != -1:
            str_list = op_node.tag.split("|")
            tmp_op["op"] = str_list[0]
            if len(str_list) >= 2:
                same_list_str = str_list[1].split(',')
                tmp_op["args"] = same_list_str

        if tmp_op["op"].find("|") != -1:
            str_list = op_node.tag.split("|")
            tmp_op["op"] = str_list[0]



    def __split_tensor(self, tensor):
        """
        Split the tensor and construct map

        Parameters:
        ----------
        None

        Returns
        -------
        Dict: construct map
        """
        tmp_op = {}
        op_node = tensor.op
        tmp_op["op"] = op_node.tag
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(op_node.input_tensors)
        tmp_op["args"] = []
        tmp_op["effective_op"] = True

        if tmp_op["op"].find("elewise_single") != -1:
            self.__split_tensor_elewise_single(tmp_op, op_node)
        if tmp_op["op"].find("elewise_binary_compare") != -1:
            self.__split_tensor_elewise_binary_compare(tmp_op, op_node)
        if tmp_op["op"].find("elewise_binary_scalar") != -1:
            self.__split_tensor_elewise_binary_scalar(tmp_op, op_node)
        elif tmp_op["op"].find("broadcast") != -1:
            self.__split_tensor_broadcast(tmp_op, op_node)
        elif tmp_op["op"].find("reduce") != -1:
            self.__split_tensor_reduce(tmp_op, op_node)

        self.__split_tensor_others(tmp_op, op_node)

        return tmp_op
