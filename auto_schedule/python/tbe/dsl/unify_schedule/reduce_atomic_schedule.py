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
reduce atomic schedule
"""
import abc
import copy

from te import platform as cceconf
from te import tvm
from tbe.dsl.base import operation
import te.platform.cce_params as cce
from te.platform.cce_conf import CceProductParams as Pver
from tbe.common.utils.errormgr import get_error_message
from .reduce_tilingcase import ReduceTilingCase
from .constants import INSN_MAPPING
from .constants import DTYPE_BYTE_MAPPING

CONST = "const"
BLOCK_SIZE_BYTE = 32


class _VectorSchedule(object):
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
            extra_space = self._emit_insn_map[stage].get("extra_space")
            if extra_space:
                self._schedule[stage].emit_insn(scope_iter_var, instruction,
                                                attrs=dict(storage_bound=[extra_space]))
            else:
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


class ReduceAtomicSchedule(_VectorSchedule):
    """
    class of cce elewise schedule

    Parameters
    ----------
    VectorSchedule: base class of reduce atomic schedule

    Returns
    -------
    ReduceAtomicSchedule_instance : instance of ReduceAtomicSchedule
    """

    def __init__(self):
        _VectorSchedule.__init__(self)
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

        self._reduce_tensors = []
        self._broadcast_tensors = []
        self._vector_dup_tensors = []  # broadcast scalar in ub
        self._tensor_dst_tensor_map = {}  # {tensor->dst_tensor(next_tensor)}

        self._tensor_scaler_operator = ["elewise_binary_mul",
                                        "elewise_binary_add"]

        # 0: reserved; 1: reduce all; 2: reduce nlast; 3: reduce last
        self._reduce_case = 0

        self._tiling_factor_vars = []

        self._spec_node_list = []
        self._total_size = 0
        self._is_muti_output = False
        self._have_reduce = False
        self._final_out_tensor_global = None
        self._final_out_tensor_global_emit_axis = 0
        self._final_out_tensor_ub_rf = None

        self._need_multi_core = True

        # reduce_axis_map: key:reduce_axis_index, value:reduce_axis_var
        # reduce_index_map: key:reduce_axis_index in original index,
        #                   value:reduce_axis_index in reduce axis
        self._reduce_info = {"reduce_tensor": None,
                             "reduce_axis_map": {},
                             "reduce_axis_index": [],
                             "reduce_index_map": [],
                             "shape_before_reduce": None,
                             "keep_dims": True,
                             "dtype": None}

        self._reduce_tiling_para = {
            "block_tiling": {"tiling_tensor": None, "axis": 0, "axis_var": None,
                             "factor": 1},
            "ub_tiling": [{"tiling_tensor": None, "axis": 0, "axis_var": None,
                           "factor": 1}]}

        self._reduce_tiling_result = {"block_tiling": {}, "ub_tiling": [{}]}

        self._storage_align_para = {}
        self._axis_offset = 0

    def do_schedule(self, outs, tiling_case, graph_info, reduce_info):
        self._res_tensor = outs[0]
        self._schedule = tvm.create_schedule([self._res_tensor.op])

        self._out_tensors = copy.copy(outs)

        self._tiling_case = tiling_case
        self.graph_info = graph_info
        self.reduce_info = reduce_info

        self._do_cache_read()
        self._do_cache_write()
        self._do_compute_inline()

        self._calculate_tiling()
        self._do_tiling()

        self._do_reorder()

        self._do_storage_bound()
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

        for i in range(0, len(outs)):
            outs.pop()
        for i in self._out_tensors:
            outs.append(i)

        return self._schedule

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

        self._calculate_cache_read()
        self._calculate_cache_write()
        self._calculate_compute_inline()

        return True

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

        # record tensor list and tensor->dst_tensor(next_tensor) map
        visited_list = []
        tensor_list = []

        visited_list.append(last_output_tensor)
        tensor_list.append(last_output_tensor)
        self.__gen_reversed_subgraph_list(last_output_tensor, tensor_list,
                                          visited_list)

        self._last_output_tensors = last_output_tensors

        # tensor classification
        self._tensor_classify(out_tensors, tensor_list)

        is_supported = self._check_pattern_supported()
        if not is_supported:
            return False

        self._res_tensor = self._last_output_tensors[0]
        self._record_reduce_info(self._res_tensor)

        reduce_tensor = self._reduce_tensors[0]

        tensor_list_before_reduce, tensor_list_after_reduce = \
            self._get_tensors_before_after_reduce(reduce_tensor,
                                                  self._res_tensor,
                                                  spec_node_list)

        self._tensor_list_before_reduce = tensor_list_before_reduce

        # record info in order to calculate ub tiling
        for tensor in reversed(tensor_list):
            tmp_op = self.__split_tensor(tensor)
            if tmp_op["effective_op"]:
                self._op.append(tmp_op)
            self._origin_op.append(tmp_op)

        is_supported = self._check_supported()
        if not is_supported:
            return False

        # calculate cache_write_exclude_tensors
        for i in self._broadcast_not_last_axis_tensors:
            self._cache_write_exclude_tensors.append(i)

        return True

    def _calculate_tiling_cases(self):
        """
        :return:
        """

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        self._tiling_case_list.clear()

        if self._is_reduce_all_axis(shape_before_reduce, reduce_axis_index):
            self._reduce_case = 1
            self._gen_tiling_case_reduce_all(shape_before_reduce,
                                             reduce_axis_index)

        elif self._is_reduce_not_last_axis(shape_before_reduce, reduce_axis_index):
            self._reduce_case = 2
            self._gen_tiling_case_not_last_axis(shape_before_reduce,
                                                reduce_axis_index)

        elif self._is_reduce_last_axis(shape_before_reduce, reduce_axis_index):
            self._reduce_case = 3
            self._gen_tiling_case_last_axis(shape_before_reduce,
                                            reduce_axis_index)

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
        for i in range(0, len(reordered_shape)):
            orignal_axis = reorder_to_orignal_axis_map[i]
            if orignal_axis in reduce_axis_index:
                block_split_axis = orignal_axis
                for j in range(0, len(reordered_shape)):
                    orignal_axis = reorder_to_orignal_axis_map[j]
                    if orignal_axis in reduce_axis_index and j < i:
                        continue
                    ub_split_axis = reorder_to_orignal_axis_map[j]
                    tiling_case = {"block_split_axis": block_split_axis,
                                   "block_factor": None,
                                   "ub_split_axis": ub_split_axis,
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
        for i in range(0, len(reordered_shape)):
            orignal_axis = reorder_to_orignal_axis_map[i]
            if orignal_axis in reduce_axis_index:
                block_split_axis = orignal_axis
                for j in range(0, len(reordered_shape)):
                    orignal_axis = reorder_to_orignal_axis_map[j]
                    if orignal_axis in reduce_axis_index and j < i:
                        continue
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
        for i in range(0, len(shape_before_reduce)):
            block_split_axis = i
            for j in range(i, len(shape_before_reduce)):
                ub_split_axis = j
                tiling_case = {"block_split_axis": block_split_axis,
                               "block_factor": None,
                               "ub_split_axis": ub_split_axis,
                               "ub_factor": None}
                self._tiling_case_list.append(tiling_case)

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
                    if tensor.op.tag == "unified_broadcast":
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
                                          visited_list)
        tensor_list_after_reduce = []
        if res_tensor != reduce_tensor:
            visited_list = []
            self.__gen_reversed_subgraph_list(res_tensor,
                                              tensor_list_after_reduce,
                                              visited_list)
        tensor_list_after_reduce.append(res_tensor)

        return tensor_list_before_reduce, tensor_list_after_reduce

    def _check_supported(self):
        """
        :return: Bool
        """
        is_supported = self._check_broadcast_supported()
        if not is_supported:
            return False
        is_supported = self._is_supported_atomic_add()
        if not is_supported:
            return False

        return True

    def _check_pattern_supported(self):
        """
        :return: Bool
        """
        if len(self._last_output_tensors) > 1:
            for tensor in self._last_output_tensors:
                if tensor.op.tag.find("tuple_reduce_sum") == -1:
                    return False
        else:
            if self._last_output_tensors[0].op.tag.find("reduce") == -1:
                return False
        for tensor in self._reduce_tensors:
            if tensor not in self._last_output_tensors:
                return False
        return True

    def _check_broadcast_supported(self):
        """
        :return: Bool
        """
        # broadcast axis must be a sub-set of reduce axis
        is_supported = self._check_broadcast_axis_support()
        if not is_supported:
            return False

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

    def _is_supported_atomic_add(self):
        """
        :return: Bool
        """
        # platform: cloud, ng1
        if not Pver().is_cloud_version() and not Pver().is_ng1_version():
            return False
        # type: fp32
        reduce_tensor = self._reduce_info["reduce_tensor"]
        if reduce_tensor is None:
            return False
        dtype = reduce_tensor.dtype
        if dtype != "float32":
            return False
        # op: sum
        tag = reduce_tensor.op.tag
        if tag.find("sum") == -1:
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

    def _support_tensor_scaler_operate(self, tensors):
        """
        Judge if tensor supports scalar instruction operations

        Parameters:
        ----------
        tensors :  input tensor

        Returns
        -------
        Bool: True or False
        """
        flag = True
        for tensor in tensors:
            tag = tensor.op.tag
            if tag.find("|") != -1:
                str_list = tag.split("|")
                tag = str_list[0]
            operator = tag
            if operator not in self._tensor_scaler_operator:
                flag = False

        return flag

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

    def __gen_reversed_subgraph_list(self, tensor, tensor_list, visited_list):
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
            if in_tensor in self._spec_node_list:
                continue

            self.__gen_reversed_subgraph_list(in_tensor, tensor_list,
                                              visited_list)

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

    @staticmethod
    def _is_reduce_all_axis(shape_before_reduce, reduce_axis_index):
        """
        :return:
        """
        for i, _ in enumerate(shape_before_reduce):
            if i not in reduce_axis_index:
                return False
        return True

    @staticmethod
    def _is_reduce_last_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        return reduce_axis_index[-1] == len(shape_before_reduce) - 1

    @staticmethod
    def _is_reduce_not_last_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        return reduce_axis_index[-1] != len(shape_before_reduce) - 1

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

    def _do_storage_bound(self):
        """
        :return:
        """
        def _get_tensor_space(_tensor):
            if _tensor in self.graph_info.tensors_after_reduce:
                _space = self.graph_info.tensor_ub_size_after_reduce
            elif _tensor in self.graph_info.tensors_before_reduce:
                _space = self.graph_info.tensor_ub_size_before_reduce
            else:
                raise RuntimeError("undefined tensor")
            return _space

        for tensor in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[tensor]
            self._schedule[read_buffer].set_storage_bound(_get_tensor_space(tensor))

        for tensor in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            self._schedule[write_buffer].set_storage_bound(_get_tensor_space(tensor))

        for tensor in self._mid_output_tensors:
            self._schedule[tensor].set_storage_bound(_get_tensor_space(tensor))

        # final output must be reduce_shape
        _bound = self.graph_info.tensor_ub_size_after_reduce
        self._schedule[self._final_out_tensor_ub_rf].set_storage_bound(_bound)
        self._schedule[self._final_out_tensor_global].set_storage_bound(_bound)

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

        if self._reduce_case == 2:
            align_axis = a1_start_index - 1
            if align_axis < 0:
                align_axis = a1_end_index

            tensor_list_before_reduce = self._tensor_list_before_reduce
            _construct_storage_align_para(tensor_list_before_reduce, align_axis,
                                          align_axis)

            is_keep_dims = self._reduce_info["keep_dims"]

            res_a1_start_index = a1_start_index
            if not is_keep_dims:
                res_a1_start_index = a1_start_index - len(reduce_axis_index)
            if res_a1_start_index == 0:
                return
            res_align_axis = res_a1_start_index - 1

            align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[
                self._final_out_tensor_ub_rf.dtype]
            para = {"align_axis_var": self._final_out_tensor_ub_rf.op.axis[
                res_align_axis + self._axis_offset],
                "align_factor": align_factor,
                "offset": 0
            }
            self._storage_align_para[self._final_out_tensor_ub_rf] = para

            para = {"align_axis_var": self._final_out_tensor_global.op.axis[
                res_align_axis],
                "align_factor": align_factor,
                "offset": 0
            }
            self._storage_align_para[self._final_out_tensor_global] = para

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
        if self._reduce_case == 2:
            a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if a1_end_index is None:
                return False
            if a1_start_index <= ub_split_axis <= a1_end_index:
                return False

        elif self._reduce_case == 3:
            r1_start_index, r1_end_index = self._find_last_reduce_axis(
                shape_before_reduce,
                reduce_axis_index)
            if r1_end_index is None:
                return False
            # for shape(a4,r4,a3,r3,a2,r2,a1,r1), if ub split r1, do not need storage_align
            if r1_start_index <= ub_split_axis <= r1_end_index:
                return False
        else:
            return False

        return True

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

    # 'pylint: disable=too-many-locals
    @staticmethod
    def _reorder_reduce_last_shape(shape_before_reduce,
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

        tiling_case: ReduceTilingCase = self._tiling_case

        block_split_axis = tiling_case.block_split_axis_index
        block_factor = tiling_case.block_factor
        ub_split_axis = tiling_case.ub_split_axis_index
        ub_factor = tiling_case.ub_factor

        if block_factor is None:

            block_inner = operation.var_inner("_block_factor", (1, None))
            self._tiling_factor_vars.append(block_inner)
        else:
            block_inner = block_factor

        if ub_factor is None:
            ub_inner = operation.var_inner("_ub_factor", (1, None))
            self._tiling_factor_vars.append(ub_inner)
        else:
            ub_inner = ub_factor

        res_tensor = self._res_tensor
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        if block_split_axis in reduce_axis_index:
            reduce_axis = self._reduce_info["reduce_axis_map"]
            axis_var = reduce_axis[block_split_axis]
            block_tiling_para = {"tiling_tensor": res_tensor,
                                 "axis": block_split_axis, "axis_var": axis_var,
                                 "factor": block_inner}
        else:
            block_tiling_para = {"tiling_tensor": res_tensor,
                                 "axis": block_split_axis, "axis_var": None,
                                 "factor": block_inner}
        # if ub tiling is performed along a certain reduce axis,
        # need to pass the reduce axis as the split itervar parameter
        if ub_split_axis in reduce_axis_index:
            reduce_axis = self._reduce_info["reduce_axis_map"]
            axis_var = reduce_axis[ub_split_axis]
            ub_tiling_para = [
                {"tiling_tensor": res_tensor, "axis": ub_split_axis,
                 "axis_var": axis_var, "factor": ub_inner}]
        else:
            ub_tiling_para = [
                {"tiling_tensor": res_tensor, "axis": ub_split_axis,
                 "axis_var": None, "factor": ub_inner}]

        self._reduce_tiling_para["block_tiling"] = block_tiling_para
        self._reduce_tiling_para["ub_tiling"] = ub_tiling_para

    def _do_tiling(self):
        """
        :return:
        """
        self._do_block_tiling()

        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_tiling_tensor = block_tiling_result["tiling_tensor"]
        block_split_axis = block_tiling_result["axis"]
        res_block_outer = block_tiling_result["outer_itervar"]

        self._atomic_additonal_schedule(block_tiling_tensor,
                                        block_split_axis, res_block_outer)

        self._do_ub_tiling()

    def _do_block_tiling(self):
        """
        :return:
        """
        block_tiling_para = self._reduce_tiling_para["block_tiling"]
        block_tiling_tensor = block_tiling_para["tiling_tensor"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner = block_tiling_para["factor"]

        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        if block_split_axis not in reduce_axis_index:
            dict_args = dict()
            dict_args["errCode"] = "E90003"
            dict_args["detailed_cause"] = "Atomic schedule block tiling can " \
                                          "only split reduce axis! " \
                                          "block_split_axis is [%s], " \
                                          "while reduce_axis is [%s]" \
                                          % (block_split_axis, reduce_axis_index)
            raise RuntimeError(dict_args, get_error_message(dict_args))

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
                    if self._reduce_case > 0:
                        shape_before_reduce = self._reduce_info["shape_before_reduce"]
                        reduce_axis_index = self._reduce_info["reduce_axis_index"]
                        is_keep_dim = self._reduce_info["keep_dims"]
                        none_reduce_index_map = self._find_none_reduce_axis_map(
                            shape_before_reduce, reduce_axis_index, is_keep_dim)
                        axis = none_reduce_index_map[ub_split_axis]
                        axis_var = ub_tiling_tensor.op.axis[axis + self._axis_offset]
                    else:
                        axis_var = ub_tiling_tensor.op.axis[ub_split_axis]
                if self._reduce_case > 0 and block_split_axis == ub_split_axis:
                    res_ub_outer, res_ub_inner = self._schedule[
                        ub_tiling_tensor].split(ub_tiling_tensor.op.reduce_axis[-1],
                                                factor=ub_split_inner)
                else:
                    res_ub_outer, res_ub_inner = self._schedule[
                        ub_tiling_tensor].split(axis_var, factor=ub_split_inner)

                ub_tiling_result = {"tiling_tensor": ub_tiling_tensor,
                                    "axis": ub_split_axis,
                                    "parent_itervar": axis_var,
                                    "outer_itervar": res_ub_outer,
                                    "inner_itervar": res_ub_inner}
            ub_tiling_result_list.append(ub_tiling_result)

        self._reduce_tiling_result["ub_tiling"] = ub_tiling_result_list

    def _atomic_additonal_schedule(self, block_tiling_tensor, block_split_axis, block_outer_var):
        """
        :param block_tiling_tensor:
        :param block_split_axis:
        :param block_outer_var:
        :return:
        """

        fused_list = []
        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_block_axis = reduce_index_map[block_split_axis]
        for i in range(0, reduce_block_axis):
            fused_list.append(block_tiling_tensor.op.reduce_axis[i])

        fused_list.append(block_outer_var)
        fused = self._schedule[block_tiling_tensor].fuse(*fused_list)

        factor_axis = 0
        final_out_tensor_ub_rf = self._schedule.rfactor(block_tiling_tensor,
                                                        fused,
                                                        factor_axis=factor_axis)
        if factor_axis == 0:
            self._axis_offset = 1
        else:
            self._axis_offset = 0

        if len(self._last_output_tensors) > 1:
            final_out_tensor_ub_rf = final_out_tensor_ub_rf[0]
        self._schedule[final_out_tensor_ub_rf].set_scope(cce.scope_ubuf)
        self._reduce_tiling_para["ub_tiling"][0]["tiling_tensor"] = \
            final_out_tensor_ub_rf

        final_out_tensor_global_list = self._schedule.cache_write(
            self._last_output_tensors, "")
        final_out_tensor_global = final_out_tensor_global_list[0]
        self._final_out_tensor_global = final_out_tensor_global
        self._final_out_tensor_ub_rf = final_out_tensor_ub_rf
        self.__replace_out_tensors(final_out_tensor_global_list)

    def _do_set_constraint(self):
        """
        :return:
        """
        if operation.get_context().get("_mode") == CONST:
            return
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]
        ub_tiling_para = ub_tiling_para_list[0]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        shape_before_reduce = self._reduce_info["shape_before_reduce_expr"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        max_ub_count = self.graph_info.tensor_ub_size_before_reduce

        if self._reduce_case == 2:
            reordered_shape, _, orignal_to_reorder_axis_map = \
                self._reorder_reduce_nlast_shape(shape_before_reduce,
                                                 reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            if isinstance(shape_in_ub, tvm.expr.Var):
                self._schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                if isinstance(shape_in_ub, tvm.expr.Var):
                    self._schedule.set_constraint(
                        reordered_shape[i] <= max_ub_count)

        else:
            reordered_shape, _, orignal_to_reorder_axis_map = \
                self._reorder_reduce_last_shape(shape_before_reduce,
                                                reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            if isinstance(shape_in_ub, tvm.expr.Var):
                self._schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis + 1, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                if isinstance(shape_in_ub, tvm.expr.Var):
                    self._schedule.set_constraint(
                        reordered_shape[i] <= max_ub_count)

        self._schedule.set_constraint(shape_in_ub <= max_ub_count)

    def _do_reorder(self):
        """
        :return:
        """

        final_out_tensor_global = self._final_out_tensor_global
        final_out_tensor_ub_rf = self._final_out_tensor_ub_rf

        if self._reduce_case == 1:
            self._reorder_atomic_reduce_all(final_out_tensor_ub_rf,
                                            final_out_tensor_global)
        if self._reduce_case == 2:
            self._reorder_reduce_not_last_axis_before_reduce()
            # for shape(r4,a4,r3,a3,r2,a2,r1,a1),
            # reorder ir (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
            # (rbo,a1,a2,..ak,r1,.rb-1,rbi,rb+1,,.rn)
            self._reorder_atomic_reduce_not_last_axis(final_out_tensor_ub_rf,
                                                      final_out_tensor_global)
        if self._reduce_case == 3:

            self._reorder_reduce_last_axis_before_reduce()
            # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
            # reorder ir (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
            # (rbo,a1,a2,..ak-1,r1,.rb-1,rbi,rb+1,,.,ak,rn)
            self._reorder_atomic_reduce_last_axis(final_out_tensor_ub_rf,
                                                  final_out_tensor_global)

    def _reorder_reduce_not_last_axis_before_reduce(self):

        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]

        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
            shape_before_reduce,
            reduce_axis_index)

        if a1_end_index is None:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "a1_end_index can not be none!"
            raise RuntimeError(dict_args, get_error_message(dict_args))

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

    def _reorder_reduce_last_axis_before_reduce(self):
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

    def __replace_out_tensors(self, final_out_tensor_global_list):
        """
        :param final_out_tensor_global_list:
        :return:
        """
        final_out_tensor_list_index = []
        for tensor in self._last_output_tensors:
            for i in range(0, len(self._out_tensors)):
                if tensor == self._out_tensors[i]:
                    final_out_tensor_list_index.append(i)
                    break
        for i, _ in enumerate(final_out_tensor_global_list):
            self._out_tensors[final_out_tensor_list_index[i]] = \
                final_out_tensor_global_list[i]

    def _reorder_atomic_reduce_all(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*(global_reordered_axis_list))

        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1 + self._axis_offset]]

        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_block_axis = reduce_index_map[block_split_axis]
        reduce_ub_axis = reduce_index_map[ub_split_axis]

        if block_split_axis != ub_split_axis:
            # rbi
            ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])
        for i in range(reduce_block_axis, reduce_ub_axis - 1):
            reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
            ub_rf_reordered_axis_list.append(reduce_axis)

        ub_rf_reordered_axis_list.append(res_ub_outer)
        ub_rf_reordered_axis_list.append(res_ub_inner)
        for i in range(reduce_ub_axis,
                       len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
            reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
            ub_rf_reordered_axis_list.append(reduce_axis)

        self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

    # 'pylint: disable=too-many-locals
    def _reorder_atomic_reduce_not_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # reorder (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak+1,ak,..a2,rk,.,rb-1,rbi,rb+1,..r2,r1,a1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..r2,r1,a1) if need fused
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        # rbo axis of out_tensor_global
        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*(global_reordered_axis_list))

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
            shape_before_reduce, reduce_axis_index)

        is_keep_dim = self._reduce_info["keep_dims"]

        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        none_reduce_index_map = self._find_none_reduce_axis_map(
            shape_before_reduce, reduce_axis_index, is_keep_dim)

        # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
        # reorder ir (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo_fused, ak,..a2,rbi,rb-1,..r2,r1,a1)
        # append rbo_fused
        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1 + self._axis_offset]]

        def __reorder_case_1(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self._schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])

            # add a1 outer, a1 may be continous
            for i in range(a1_start_index, ub_split_axis):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self._schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])
            ub_rf_reordered_axis_list.append(res_ub_outer)

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1 inner, a1 may be continous
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(ub_split_axis + 1, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self._schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])

            self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

        def __reorder_case_2(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self._schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            reduce_ub_axis = reduce_index_map[ub_split_axis]
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis,
                           len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1
            for i in range(a1_start_index, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self._schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])
            self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

        def __reorder_case_3(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, ub_split_axis - 1):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self._schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])
            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(ub_split_axis + 1, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self._schedule[out_tensor_ub_rf].op.axis[
                            none_reduce_index + self._axis_offset])

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1
            for i in range(a1_start_index, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(
                    self._schedule[out_tensor_ub_rf].op.axis[
                        none_reduce_index + self._axis_offset])

            self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

        # if ub split axis in(a1)
        reduce_block_axis = reduce_index_map[block_split_axis]
        if a1_start_index <= ub_split_axis <= a1_end_index:
            __reorder_case_1(ub_rf_reordered_axis_list)
            return

        # if ub split axis in (rbi,rb-1,..r2,r1)
        if ub_split_axis in reduce_axis_index:
            __reorder_case_2(ub_rf_reordered_axis_list)
            return

        # if ub split axis in (ak,..a2)
        if ub_split_axis not in reduce_axis_index:
            __reorder_case_3(ub_rf_reordered_axis_list)

    def _reorder_atomic_reduce_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
        # reorder (ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak,..a2,rk,.,rb-1,rbi,rb+1,..a1,r1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..a1,r1) if need fused
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        is_keep_dim = self._reduce_info["keep_dims"]

        # reorder ir (ak,..a2,a1,rbo) to (rbo,ak,..a2,a1)
        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*global_reordered_axis_list)

        none_reduce_index_map = self._find_none_reduce_axis_map(
            shape_before_reduce, reduce_axis_index, is_keep_dim)

        ub_rf_reordered_axis_list = []
        reduce_index_map = self._reduce_info["reduce_index_map"]

        # 'reorder (ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak,..a2,a1, rk,.,rb-1,rbi,rb+1,..r2,r1) or
        # (rbo_fused, ak,..a2,a1, rbi,rb+1,..r2,r1) if need fused

        # rbo
        ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.axis[-1 + self._axis_offset])

        # 'if ub split axis in (rbi,rb+1,..r2,r1)
        if ub_split_axis in reduce_axis_index:
            # add axis (ak,..a2,a1)
            for i in range(0, len(self._schedule[out_tensor_ub_rf].op.axis) - 1):
                ub_rf_reordered_axis_list.append(
                    self._schedule[out_tensor_ub_rf].op.axis[i + self._axis_offset])

            # 'append rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            reduce_block_axis = reduce_index_map[block_split_axis]
            reduce_ub_axis = reduce_index_map[ub_split_axis]
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis,
                           len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

        # 'if ub split axis in (ak,..a2,a1)
        else:
            # add axis (ak,..a2,a1)
            for i in range(0, ub_split_axis - 1):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index + self._axis_offset])

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)

            for i in range(ub_split_axis + 1, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(
                        self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index + self._axis_offset])

            # 'rbi
            ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

        self._schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

    @staticmethod
    def _find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index, keep_dims):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        none_reduce_index_map = {}
        if keep_dims:
            for i in range(0, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index_map[i] = i
        else:
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
            self._multi_core_fused_axis = \
                self._final_out_tensor_global.op.reduce_axis[0]
            self._multi_core_bind_tensor = self._final_out_tensor_global

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

        for ub_tiling_result in ub_tiling_result_list:
            if "tiling_tensor" not in ub_tiling_result.keys() or \
                    "outer_itervar" not in ub_tiling_result.keys():
                continue
            ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
            res_ub_outer = ub_tiling_result["outer_itervar"]
            if self._reduce_case == 2:
                ub_split_axis = ub_tiling_result["axis"]
                # for shape (r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, compute at r1
                # when a1 is continous,
                reduce_axis_index = self._reduce_info["reduce_axis_index"]
                shape_before_reduce = self._reduce_info["shape_before_reduce"]
                a1_start_index, a1_end_index = self._find_last_none_reduce_axis(
                    shape_before_reduce,
                    reduce_axis_index)
                if a1_end_index is None:
                    dict_args = dict()
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "a1_end_index can not be none!"
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                if a1_start_index <= ub_split_axis <= a1_end_index:
                    if len(ub_tiling_tensor.op.reduce_axis) > 1:
                        res_ub_outer = ub_tiling_tensor.op.reduce_axis[-2]
                    else:
                        res_ub_outer = ub_tiling_tensor.op.reduce_axis[-1]

            for i in self._cache_read_tensors_and_buffer_map:
                read_buffer = self._cache_read_tensors_and_buffer_map[i]
                para = {"parent": self._schedule[ub_tiling_tensor], "scope": res_ub_outer}
                self._compute_at_map[read_buffer] = para
            for i in self._cache_write_tensors_and_buffer_map:
                write_buffer = self._cache_write_tensors_and_buffer_map[i]
                para = {"parent": self._schedule[ub_tiling_tensor], "scope": res_ub_outer}
                self._compute_at_map[write_buffer] = para
            for i in self._mid_output_tensors:
                para = {"parent": self._schedule[ub_tiling_tensor], "scope": res_ub_outer}
                self._compute_at_map[i] = para

        para = {"parent": self._schedule[self._final_out_tensor_global],
                "scope": self._final_out_tensor_global.op.reduce_axis[0]}
        self._compute_at_map[self._final_out_tensor_ub_rf] = para

    def _calculate_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """

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
        ub_tiling_result = ub_tiling_result_list[0]
        ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        def get_insn(tensor_):
            tag = tensor_.op.tag
            if tensor_.op.tag.find("|") != -1:
                insn = tag.split("|")[0]
            else:
                insn = tag
            return INSN_MAPPING.get(insn, insn)

        ub_split_axis = ub_tiling_result["axis"]
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"scope": read_buffer.op.axis[ub_split_axis],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = get_insn(write_buffer)
            para = {"scope": write_buffer.op.axis[0],
                    "instruction": insn}
            self._emit_insn_map[write_buffer] = para

        for out_tensor in self._mid_output_tensors:
            para = {"scope": out_tensor.op.axis[0],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[out_tensor] = para

        # ub_tiling_tensor must be reduce_tensor
        res_tensor = self._res_tensor
        extra_space = self.graph_info.tensor_ub_size_before_reduce

        if self._reduce_case == 2:
            ub_split_axis = ub_tiling_result["axis"]
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
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "a1_end_index can not be none!"
                raise RuntimeError(dict_args, get_error_message(dict_args))
            if ub_split_axis < a1_start_index and \
                    ub_split_axis not in reduce_axis_index:
                res_ub_inner = ub_tiling_tensor.op.reduce_axis[-1]

            self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                     "instruction": 'vector_reduce_sum',
                                                     "extra_space": extra_space}
        elif self._reduce_case == 3:
            reduce_axis_index = self._reduce_info["reduce_axis_index"]
            ub_split_axis = ub_tiling_result["axis"]
            # ub cut ak (none reduce axis),
            if ub_split_axis not in reduce_axis_index:
                self._emit_insn_map[ub_tiling_tensor] = {
                    "scope": ub_tiling_tensor.op.reduce_axis[-1],
                    "instruction": 'vector_reduce_sum',
                    "extra_space": extra_space}
            else:
                self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                         "instruction": 'vector_reduce_sum',
                                                         "extra_space": extra_space}
        else:
            self._emit_insn_map[ub_tiling_tensor] = {"scope": res_ub_inner,
                                                     "instruction": 'vector_reduce_sum',
                                                     "extra_space": extra_space}

        self._emit_insn_map[self._final_out_tensor_global] = {
            "scope": self._final_out_tensor_global.op.axis[0],
            "instruction": 'dma_copy'}
        self._emit_insn_map[res_tensor] = {
            "scope": self._schedule[res_tensor].op.axis[0],
            "instruction": 'phony_insn'}

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
