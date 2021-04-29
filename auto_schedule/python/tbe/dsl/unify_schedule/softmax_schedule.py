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
softmax schedule
"""
from typing import Optional
from te import tvm
from .constants import Pattern, DTYPE_BYTE_MAPPING
from .constants import CompileInfo
from . import util
from .softmax_tilingcase import TilingStrategy

from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base import operation
from te import platform as cce
from .util import shape_to_list
from te.platform import cce_emitinsn_params
BLOCK_SIZE_BYTE = 32
EXTRA_UB_SIZE = 10240

def get_reduce_axis_num(reduce_tensor):
    """
    get reduce axis num
    """
    data_axis_var = reduce_tensor.op.body[0].source[0].args
    reduce_axis_var = []
    for i in reduce_tensor.op.reduce_axis:
        reduce_axis_var.append(i.var)

    axis_num = []
    for ax_var in reduce_axis_var:
        num = 0
        for i in data_axis_var:
            if i.same_as(ax_var):
                axis_num.append(num)
            num += 1

    return axis_num

def get_align_factor(dtype):
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

def get_bits_of(dtype):
    """
    calculate bits of dtype of TVM
    Parameters
    ----------
    dtype : string
        dtype of TVM

    Returns
    -------
    ret : int
        bit length of dtype.
    """
    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    return int(dtype[index:])

# the byte of dtype
DTYPE_BYTE_WIDTH_MAP = {"float16": 2,
                        "float32": 4,
                        "int32": 4,
                        "int16": 2,
                        "uint16": 2,
                        "int8": 1,
                        "uint8": 1,
                        "bool": 1}

@register_schedule(pattern=Pattern.SOFTMAX)
def schedule(outs, tiling_case):
    """
    :param outs:
    :param tiling_case:
    :return:
    """
    return SoftmaxSchedule(outs, tiling_case).do_schedule()

class SoftmaxSchedule:
    """
    SoftmaxSchedule
    """
    def __init__(self, outs, tiling_case):
        self._scope = cce.scope_ubuf
        self._core_dim = cce.get_soc_spec("CORE_NUM")
        if self._scope.lower().find('.ub') != -1:
            self._total_size = cce.get_soc_spec("UB_SIZE")
        else:
            raise RuntimeError("only support UB buffer now")

        self._ub_tiling_max_size = self._total_size
        self._max_ub_count = 0
        self._schedule = None
        self._spec_node_list = []
        self._spec_mid_list = []
        self._outs = outs
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._coexisting_quantity = 8
        self._tiling_result = {"block_tiling": {"axis": 0,
                                                "parent_itervar": None,
                                                "outer_itervar": None,
                                                "inner_itervar": None},
                               "ub_tiling": {"axis": 0,
                                             "parent_itervar": None,
                                             "outer_itervar": None,
                                             "inner_itervar": None}}

        self._cache_read_tensors_and_readers_map = {}
        self._cache_read_buffer_and_reader_map = {}
        self._cache_read_tensors_and_buffer_map = {}
        self._cache_write_tensors = []
        self._cache_write_tensors_and_input_map = {}
        self._cache_write_tensors_and_buffer_map = {}
        self._reduce_tensors = []
        self._broadcast_tensors = []
        self._after_reduce_bc_tensors = []
        self._compute_inline_buffer = []
        self._compute_at_map = {}
        self._compute_at_tensor_map = {}
        self._emit_insn_map = {}
        self._res = None
        self._is_last_reduce_axis = None
        self._is_32byte_align = None
        self._reduce_axis_num = None
        self._is_this_schedule_support = True
        self._origin_op = []
        self._double_buffer_tensors = []
        self._is_double_buffer = False
        self._is_multi_core = False
        self._is_block_ub_one_axis = False
        self._is_block_fuse = False
        self._is_reduce_last_axis_enhance_insn = False

    def _get_emit_insn_map(self):
        self._insn_map = {"elewise_single_cast": "vector_conv",
                          "elewise_single_round_d": "vector_conv_round",
                          "elewise_single_trunc": "vector_conv_trunc",
                          "elewise_single_VS_max": "vector_maxs",
                          "elewise_single_VS_min": "vector_mins",
                          "elewise_single_log": "vector_ln",
                          "elewise_single_exp": "vector_exp",
                          "elewise_single_relu": "vector_relu",
                          "elewise_single_lrelu": "vector_lrelu",
                          "elewise_single_abs": "vector_abs",
                          "elewise_single_not": "vector_not",
                          "elewise_single_sqrt": "vector_sqrt",
                          "elewise_single_rsqrt": "vector_rsqrt",
                          "elewise_binary_mul": "vector_mul",
                          "elewise_single_VS_mul": "vector_muls",
                          "elewise_binary_div": "vector_div",
                          "elewise_binary_add": "vector_add",
                          "elewise_single_VS_add": "vector_adds",
                          "elewise_binary_min": "vector_min",
                          "elewise_binary_max": "vector_max",
                          "elewise_binary_vcmpv_gt": "vector_gt",
                          "elewise_binary_vcmpv_ge": "vector_ge",
                          "elewise_binary_vcmpv_lt": "vector_lt",
                          "elewise_binary_vcmpv_le": "vector_le",
                          "elewise_binary_vcmpv_eq": "vector_eq",
                          "elewise_binary_vcmpv_ne": "vector_ne",
                          "elewise_binary_cmpsel_gt": "vector_select_gt",
                          "elewise_binary_cmpsel_ge": "vector_select_ge",
                          "elewise_binary_cmpsel_lt": "vector_select_lt",
                          "elewise_binary_cmpsel_le": "vector_select_le",
                          "elewise_binary_cmpsel_eq": "vector_select_eq",
                          "elewise_binary_cmpsel_ne": "vector_select_ne",
                          "elewise_binary_or": "vector_or",
                          "elewise_binary_and": "vector_and",
                          "elewise_binary_addrelu": "vector_addrelu",
                          "elewise_binary_subrelu": "vector_subrelu",
                          "elewise_multiple_mla": "vector_multiple",
                          "elewise_multiple_madd": "vector_multiple",
                          "elewise_multiple_maddrelu": "vector_multiple",
                          "elewise_multiple_sel": "vector_select_bool",
                          "elewise_binary_sub": "vector_sub",
                          "elewise_binary_phony": "elewise_binary_phony"}

    def _get_reg_emit_insn_map(self):
        self._reg_insn_map = {
            "broadcast_for_tensor": "unified_broadcast",
            "unified_broadcast": "unified_broadcast",
            "elewise_binary_scalar_axpy": "vector_multiple",
            "emit_insn_elewise_multiple_sel":
                "elewise_multiple_sel",
            "emit_insn_elewise_binary_cmp":
                "elewise_binary_cmp",
            "elewise_binary_cmpsel": "vector_cmpsel",
            "elewise_single_VS_cond": "elewise_single_VS_cond",
            "elewise_binary_logic": "elewise_binary_logic",
            "broadcast": "broadcast",
            "elewise_single_round": "vector_conv_rint",
            "elewise_single_rec": "vector_rec",
            "elewise_binary_sub": "elewise_binary_sub",
            "elewise_single_cast": "vector_conv",
            "elewise_single_floor": "vector_conv_floor",
            "elewise_single_ceil": "elewise_single_ceil",
            "elewise_binary_compare_lt":
                "elewise_binary_compare_lt",
            "elewise_binary_compare_gt":
                "elewise_binary_compare_gt",
            "elewise_single_VS_mul_with_reg_in_quant":
                "elewise_single_VS_mul_with_reg_in_quant",
            "elewise_single_VS_adds_with_reg":
                "elewise_single_VS_adds_with_reg",
            "elewise_single_VS_mul_with_reg_sqrt_in_quant":
                "elewise_single_VS_mul_with_reg_sqrt_in_quant",
            "elewise_single_VS_mul_with_reg":
                "elewise_single_VS_mul_with_reg",
            "elewise_single_VS_add_with_reg":
                "elewise_single_VS_add_with_reg",
            "elewise_single_diagonal":
                "elewise_single_diagonal",
            "read_select": "dma_copy",
            "write_select": "dma_copy"}

    # pylint: disable=arguments-differ
    def do_schedule(self):
        """
        auto_schedule for cce AI-CORE

        Parameters
        ----------
        outTensors : the out tvm.tensor

        sch : schedule, the computation schedule for the op

        Returns
        -------
        Bool, now is true

        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._res.op)
        self._schedule.tiling_key = self._tiling_case["key"]
        align_factor_32byte, _ = get_align_factor(self._res.dtype)
        cce_emitinsn_params.cceEmitParamsIns.clear_param()
        cce_emitinsn_params.cceEmitParamsIns.insert_param("broadcast_axis_offset", 1)

        if not self._is_last_reduce_axis:
            sch = self._do_schedule_nlst_axis()
        else:
            sch = self._do_schedule_last_noalign_storagealign()
        return sch


    def _construct_compute_graph(self):
        """
        record relate context imformations of operations

        outTensors only support form like: out_1->..out_2->..out_n

        """
        # 1. find the last out tensor
        if hasattr(self._outs, 'index'):
            if len(self._outs) > 1:
                self._is_this_schedule_support = False
                return False  # not mutile output
            self._res = self._outs[0]
        else:
            self._res = self._outs

        # 2. traverse tensors of subgraph by Depth-First-Search
        tensor_list = []
        visited_list = []
        self.__gen_reversed_subgraph_list(self._res, tensor_list, visited_list)

        self._cache_write_tensors = list(reversed(tensor_list))
        for tensor in self._cache_write_tensors:
            tmp_op = self.__get_tensor_op(tensor)
            self._origin_op.append(tmp_op)

        # 3. check if the subgraph matches this schedule template
        shape = shape_to_list(self._res.shape)
        for i in self._broadcast_tensors:
            if shape_to_list(i.shape) != shape:
                self._is_this_schedule_support = False
                return False  # broadcast must be as same as the shape of elewise res

        reduce_axis_num = get_reduce_axis_num(self._reduce_tensors[0])
        for i in self._reduce_tensors:
            if get_reduce_axis_num(i) != reduce_axis_num:
                self._is_this_schedule_support = False
                return False  # all reduce must have the same reduce_axis

        if len(reduce_axis_num) != 1:
            self._is_this_schedule_support = False
            return False  # must be only one reduce_axis

        self._is_last_reduce_axis = ((len(shape) - 1) in reduce_axis_num)
        self._reduce_axis_num = reduce_axis_num[0]

        align_factor_32byte, _ = \
            self._get_data_alignment(self._cache_write_tensors)
        self._is_32byte_align = (shape[-1] % align_factor_32byte == 0)

        #self._get_max_ub_count_and_try_double_buffer()

        return True


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
        search_tensor_list = list(tensor.op.input_tensors)
        search_tensor_list.append(tensor)
        for in_tensor in search_tensor_list:
            # Mark special nodes, handle them separately
            if self._is_reduce_ops(in_tensor):
                if in_tensor not in self._reduce_tensors:
                    self._reduce_tensors.append(in_tensor)
            elif self._is_broadcast_ops(in_tensor):
                if in_tensor not in self._broadcast_tensors:
                    self._broadcast_tensors.append(in_tensor)
            elif self._is_after_reduce_bc_ops(in_tensor):
                if in_tensor not in self._after_reduce_bc_tensors:
                    self._after_reduce_bc_tensors.append(in_tensor)

            if in_tensor in self._spec_node_list or \
                    isinstance(in_tensor.op, tvm.tensor.PlaceholderOp):
                if not in_tensor.same_as(tensor):
                    self._map_apend(self._cache_write_tensors_and_input_map, tensor, in_tensor)
                    self._map_apend(self._cache_read_tensors_and_readers_map, in_tensor, tensor)
                continue
            else:
                if not in_tensor.same_as(tensor):
                    self._map_apend(self._cache_write_tensors_and_input_map, tensor, in_tensor)

            if in_tensor in visited_list:
                continue

            visited_list.append(in_tensor)
            self.__gen_reversed_subgraph_list(in_tensor, tensor_list, visited_list)
            tensor_list.append(in_tensor)


    # pylint: disable=no-self-use
    def __get_tensor_op(self, tensor):
        """
        Split the tensor and construct map

        Parameters:
        ----------
        None

        Returns
        -------
        Dict: construct map
        """
        tmp_op = {"op": None,
                  "dst_buffer": [],
                  "src_buffer": [],
                  "args": [],
                  "effective_op": True}
        str_list = tensor.op.tag.split("|")
        tmp_op["op"] = str_list[0]
        tmp_op["dst_buffer"] = tensor
        tmp_op["src_buffer"] = list(tensor.op.input_tensors)

        return tmp_op


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


    # pylint: disable=no-self-use
    def _is_reduce_ops(self, tensor):
        """
        reduce_ops
        """
        if tensor.op.tag.find("reduce") != -1:
            return True
        return False


    # pylint: disable=no-self-use
    def _is_broadcast_ops(self, tensor):
        """
        broadcast
        """
        if tensor.op.tag.find("broadcast") != -1:  # broadcast_for_tensor
            return True
        return False


    def _is_after_reduce_bc_ops(self, tensor):
        """
        broadcast
        """
        if self._is_reduce_ops(tensor) or self._is_broadcast_ops(tensor):
            return False
        for in_tensor in list(tensor.op.input_tensors):
            if self._is_reduce_ops(in_tensor) or self._is_broadcast_ops(in_tensor):
                return True
        return False


    def _get_max_ub_count_and_try_double_buffer(self):
        shape = shape_to_list(self._res.shape)
        align_factor_32byte, _ = \
            self._get_data_alignment(self._cache_write_tensors)

        # calculate max_ub_count
        reduce_rows = 1 if self._is_last_reduce_axis else shape[self._reduce_axis_num]
        self._max_ub_count = self._get_max_ub_count(self._total_size,
                                                    align_factor_32byte,
                                                    reduce_rows)
        if (self._max_ub_count < align_factor_32byte) or (self._is_last_reduce_axis):
            self._is_this_schedule_support = False
            return False  # workspace

        # calculate max_ub_count: try double buffer
        self._is_double_buffer = False
        tmp_max_ub_count = self._get_max_ub_count(self._total_size // 2,
                                                  align_factor_32byte,
                                                  reduce_rows)
        if (tmp_max_ub_count < align_factor_32byte) or \
                (self._is_last_reduce_axis and tmp_max_ub_count < shape[self._reduce_axis_num]):
            self._is_double_buffer = False
        else:
            self._is_double_buffer = True
            self._total_size = self._total_size // 2
            self._max_ub_count = tmp_max_ub_count

        return True


    def _get_total_width(self):
        """
        caculate the max useable number based on op liveness
        return: max useable number.
            example: Three float16 nodes, return 2B*3 = 6B
                     Three float32 nodes, return 4B*3 = 12B
        """
        def _op_width(op_node):
            num_type = op_node.dtype
            if num_type.lower() not in DTYPE_BYTE_WIDTH_MAP.keys():
                raise RuntimeError("Can not calculate with no compute")

            tmp_width = 0
            if op_node.op.tag is not None:
                tag = op_node.op.tag
                # logic use 4 fp16 temp buffer
                if tag.find("logic") != -1:
                    tmp_width = 4 * DTYPE_BYTE_WIDTH_MAP["float16"]
                # cond use 3 fp16 temp buffer
                elif tag.find("cond") != -1:
                    tmp_width = 3 * DTYPE_BYTE_WIDTH_MAP["float16"]
                # vsel use 3 fp16 temp buffer
                elif tag.find("sel") != -1:
                    tmp_width = 3 * DTYPE_BYTE_WIDTH_MAP["float16"]
                # vcompare use 2 temp buffer
                elif tag.find("compare") != -1:
                    tmp_width = 2 * DTYPE_BYTE_WIDTH_MAP[num_type.lower()]
                # vcomsel use 3 temp buffer
                elif tag.find("cmpsel") != -1:
                    tmp_width = 3 * DTYPE_BYTE_WIDTH_MAP[num_type.lower()]

            return DTYPE_BYTE_WIDTH_MAP[num_type.lower()] + tmp_width

        def _post_dfs_order(op_node, op_graph, visited, post_order):
            if op_node in visited:
                return
            visited[op_node] = True
            post_order.append(op_node)
            if op_node in op_graph:
                for src in op_graph[op_node]:
                    _post_dfs_order(src, op_graph, visited, post_order)

        tensors_and_input_map = {}
        for op_node in self._origin_op:
            src_op = list(op_node['src_buffer'])
            src_op.reverse()
            tensors_and_input_map[op_node['dst_buffer']] = src_op

        visited = {}
        post_order = []
        _post_dfs_order(self._res, tensors_and_input_map, visited, post_order)

        lives = [self._res]
        visited = {lives[0]: True}
        live_width = _op_width(lives[0])
        max_width = live_width
        for op_node in post_order:
            if op_node in tensors_and_input_map:
                for src in tensors_and_input_map[op_node]:
                    if src in visited:
                        continue
                    lives.append(src)
                    live_width += _op_width(src)
                    visited[src] = True
                if live_width > max_width:
                    max_width = live_width
            lives.remove(op_node)
            live_width -= _op_width(op_node)
        return max_width


    def _get_max_ub_count(self, total_size, align_factor=32, reduce_rows=1):
        """
        caculate the max element num loaded in UB buffer
        :return: max element num loaded in UB buffer
        """
        total_width = self._get_total_width()
        if self._is_last_reduce_axis:
            total_width = total_width + DTYPE_BYTE_WIDTH_MAP["float32"]

        if not total_width:
            raise RuntimeError("Can not calculate with no compute")

        # example: Three float16 nodes, the number of reduce rows is 5
        # max_ub_count = 248KB // (2B*3*5) = 248KB // (2B*3) // 5
        max_ub_count = total_size // total_width // reduce_rows

        max_ub_count = max_ub_count // align_factor * align_factor

        return max_ub_count


    # pylint: disable=no-self-use
    def _get_data_alignment(self, tensor_list):
        """calculate the unified buffer data alignment
        1. enable multi-core needs to be larger than 32B,
        2. vector calculation should be 256B aligned as much as possible
        fp16: align_factor_32byte=16, align_factor_256byte=128
        fp32: align_factor_32byte=8,  align_factor_256byte=64

        therefore, using the smallest dtype to make a tiling
        """
        bits = 65535
        for i in tensor_list:
            bits_update = get_bits_of(i.dtype.lower())
            if bits > bits_update:
                bits = bits_update

        align_factor_32byte = 32*8//bits
        align_factor_256byte = 256*8//bits
        return align_factor_32byte, align_factor_256byte


    def _do_storage_align_last(self):
        if not self._is_last_reduce_axis:
            return

        at_axis = self._reduce_axis_num - 1  # axis: -2

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(read_buffer.dtype)
            self._schedule[read_buffer].storage_align(read_buffer.op.axis[at_axis], align_factor, 0)

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(write_buffer.dtype)
            if write_buffer not in self._compute_inline_buffer:
                self._schedule[write_buffer].storage_align(write_buffer.op.axis[at_axis],
                                                           align_factor, 0)


    def _do_storage_align_nlst(self):
        if self._is_last_reduce_axis:
            return

        at_axis = self._reduce_axis_num

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(read_buffer.dtype)
            self._schedule[read_buffer].storage_align(read_buffer.op.axis[at_axis], align_factor, 0)

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            align_factor, _ = get_align_factor(write_buffer.dtype)
            if write_buffer not in self._compute_inline_buffer:
                self._schedule[write_buffer].storage_align(write_buffer.op.axis[at_axis],
                                                           align_factor, 0)


    def _do_schedule_nlst_axis(self):
        self._do_cache_read()
        self._do_cache_write()
        self._calculate_compute_inline()
        self._do_compute_inline()

        self._do_tiling_last_noalign_storagealign()
        if not self._is_this_schedule_support:
            return None

        self._do_reorder()
        self._calculate_compute_at()
        self._do_compute_at()
        self._calculate_emit_insn_nlst()
        self._do_emit_insn()
        self._do_storage_align_nlst()
        self._do_double_buffer()
        sch = self._schedule

        return sch


    def _do_schedule_last_noalign_storagealign(self):
        self._do_cache_read()
        self._do_cache_write()
        self._calculate_compute_inline()
        self._do_compute_inline()
        self._do_storage_align_last()
        self._do_tiling_last_noalign_storagealign()
        self._calculate_compute_at()
        self._do_compute_at()
        self._calculate_emit_insn_last()
        self._do_emit_insn()
        self._do_double_buffer()
        sch = self._schedule

        return sch


    def _do_reorder(self):
        if self._is_last_reduce_axis:
            return
        u_idx = self._tiling_case["ub_tiling_axis"]
        ub_axis = self._tiling_result["ub_tiling"]["axis"]
        ub_outer = self._tiling_result["ub_tiling"]["outer_itervar"]
        ub_inner = self._tiling_result["ub_tiling"]["inner_itervar"]

        block_axis = self._tiling_result["block_tiling"]["axis"]
        block_inner = self._tiling_result["block_tiling"]["inner_itervar"]
        # 1. res tensor reorder
        # block_outer,block_inner,ub_outer,reduce_axis,ub_inner
        reorder_list = []
        if block_axis > self._reduce_axis_num:
            # (reduce~block], reorder multi_core first
            reorder_list.append(self._multi_core_fused_axis)
            reorder_list.append(block_inner)

            # [0~reduce)
            for i in range(0, self._reduce_axis_num):
                reorder_list.append(self._res.op.axis[i])
            # (reduce~block] ==split==> (self._multi_core_fused_axis, block_inner)
            # (block_axis~ub_axis)
            for i in range(block_axis+1, ub_axis):
                reorder_list.append(self._res.op.axis[i])
            # ub_split, reduce axis
            reorder_list.append(ub_outer)
            reorder_list.append(self._res.op.axis[self._reduce_axis_num])
            reorder_list.append(ub_inner)
        else:
            # [0~block]
            reorder_list.append(self._multi_core_fused_axis)
            reorder_list.append(block_inner)
            # (block_axis~reduce~ub_axis)
            for i in range(block_axis+1, ub_axis):
                if i != self._reduce_axis_num:
                    reorder_list.append(self._res.op.axis[i])
            # ub_split, reduce axis
            reorder_list.append(ub_outer)
            reorder_list.append(self._res.op.axis[self._reduce_axis_num])
            reorder_list.append(ub_inner)

        self._schedule[self._res].reorder(*reorder_list)

        # 2. reduce_tensor reorder
        #ub_factor = self._tiling_para["ub_tiling"]["factor"]
        for i in self._reduce_tensors:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            outer, inner = self._schedule[write_buffer].split(
                write_buffer.op.axis[u_idx],
                factor=self._ub_tiling_vars[u_idx])
            self._schedule[write_buffer].reorder(outer, write_buffer.op.reduce_axis[0], inner)


    def _do_tiling_last_noalign_storagealign(self):
        b_i = self._tiling_case["block_tiling_axis"]
        u_i = self._tiling_case["ub_tiling_axis"]
        self._block_tiling_vars[b_i] = operation.var("block_factor_" + str(b_i))
        self._ub_tiling_vars[u_i] = operation.var("ub_factor_" + str(u_i))
        self._is_block_ub_one_axis = True

        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]

        block_outer, block_inner = self._schedule[self._res].split(
            self._res.op.axis[b_idx], nparts=self._block_tiling_vars[b_idx])

        if self._is_block_ub_one_axis:
            block_inner, tobe_split = self._schedule[self._res].split(block_inner, nparts=1)
            ub_outer, ub_inner = self._schedule[self._res].split(tobe_split, factor=self._ub_tiling_vars[u_idx])
        else:
            ub_outer, ub_inner = self._schedule[self._res].split(self._res.op.axis[u_idx],
                                                                 factor=self._ub_tiling_vars[u_idx])

        self._multi_core_fused_axis = block_outer
        block = tvm.thread_axis("blockIdx.x")
        self._schedule[self._res].bind(self._multi_core_fused_axis, block)
        block_tiling_result = {"axis": b_idx,
                               "outer_itervar": block_outer,
                               "inner_itervar": block_inner}
        ub_tiling_result = {"axis": u_idx,
                            "outer_itervar": ub_outer,
                            "inner_itervar": ub_inner}
        self._tiling_result = {"block_tiling": block_tiling_result,
                               "ub_tiling": ub_tiling_result}


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
        # reduce_axis is not last axis, broadcast could compute_inline
        if not self._is_last_reduce_axis:
            for i in self._broadcast_tensors:
                write_buffer = self._cache_write_tensors_and_buffer_map[i]
                self._compute_inline_buffer.append(write_buffer)


    def _do_compute_inline(self):
        """
        compute inline operations
        """
        for i in self._cache_write_tensors[1:]:
            self._schedule[i].compute_inline()
        for i in self._compute_inline_buffer:
            self._schedule[i].compute_inline()


    def _calculate_compute_at(self):
        """
        Calculate the tensor that needs compute at
        """
        ub_outer = self._tiling_result["ub_tiling"]["outer_itervar"]

        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"parent": self._schedule[self._res], "scope": ub_outer}
            self._compute_at_map[read_buffer] = para

        # write_buffer compute_at, except compute_inline_buffer
        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            if write_buffer not in self._compute_inline_buffer:
                para = {"parent": self._schedule[self._res], "scope": ub_outer}
                self._compute_at_map[write_buffer] = para

    def _do_compute_at(self):
        for stage in self._compute_at_map:
            parent_stage = self._compute_at_map[stage]["parent"]
            scope_iter_var = self._compute_at_map[stage]["scope"]
            if scope_iter_var is not None:
                self._schedule[stage].compute_at(parent_stage, scope_iter_var)


    def _calculate_emit_insn_last(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        # Backend instruction map
        self._get_emit_insn_map()
        # Front-end instruction map
        self._get_reg_emit_insn_map()

        # 1. res dma_copy
        # 2. input dma_copy
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            self._emit_insn_map[read_buffer] = {"scope": read_buffer.op.axis[0],
                                                "instruction": 'dma_copy'}

        # 3. ub compute
        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._get_instruction(write_buffer)
            if not insn and i not in self._reduce_tensors:
                raise RuntimeError("get_instruction is None")

            # 3.0 compute_inline, (not last reduce axis broadcast)
            if write_buffer in self._compute_inline_buffer:
                continue
            # 3.1 reduce
            elif i in self._reduce_tensors:
                # vector_reduce_max/vector_reduce_min/vector_reduce_sum
                insn = write_buffer.op.tag.split("|")[0]
                if self._is_reduce_last_axis_enhance_insn:
                    insn = "reduce_last_axis_enhance_" + insn
                    insn_scope = write_buffer.op.axis[self._tiling_result["ub_tiling"]["axis"]]
                else:
                    insn = "vector_" + insn
                    insn_scope = write_buffer.op.axis[0]
                insn_para = {"scope": insn_scope, "instruction": insn}
            # 3.2 elewise using reduce result, compute_inline broadcast(not last axis)
            elif i in self._after_reduce_bc_tensors and (not self._is_last_reduce_axis):
                insn_para = {
                    "scope": write_buffer.op.axis[self._tiling_result["ub_tiling"]["axis"]],
                    "instruction": insn}
            # 3.3 last reduce axis broadcast
            elif i in self._broadcast_tensors and self._is_last_reduce_axis:
                insn_scope = write_buffer.op.axis[self._tiling_result["ub_tiling"]["axis"]]
                insn_para = {"scope": insn_scope, "instruction": "vector_broadcast"}
            # 3.4 elewise
            else:
                insn_para = {"scope": write_buffer.op.axis[0], "instruction": insn}

            self._emit_insn_map[write_buffer] = insn_para


    def _calculate_emit_insn_nlst(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        # Backend instruction map
        self._get_emit_insn_map()
        # Front-end instruction map
        self._get_reg_emit_insn_map()

        # 1. res dma_copy
        # 2. input dma_copy
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            self._emit_insn_map[read_buffer] = {"scope": read_buffer.op.axis[0],
                                                "instruction": 'dma_copy'}

        # 3. ub compute
        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = self._get_instruction(write_buffer)
            if not insn and i not in self._reduce_tensors:
                raise RuntimeError("get_instruction is None")

            # 3.0 compute_inline, (not last reduce axis broadcast)
            if write_buffer in self._compute_inline_buffer:
                continue
            # 3.1 reduce
            elif i in self._reduce_tensors:
                # vector_reduce_max/vector_reduce_min/vector_reduce_sum
                insn = write_buffer.op.tag.split("|")[0]
                insn = "vector_" + insn
                insn_para = {"scope": write_buffer.op.reduce_axis[0], "instruction": insn}
            # 3.2 elewise using reduce result, compute_inline broadcast(not last axis)
            elif i in self._after_reduce_bc_tensors and (not self._is_last_reduce_axis):
                insn_para = {"scope": write_buffer.op.axis[0],
                             "instruction": insn}
            # 3.3 last reduce axis broadcast
            elif i in self._broadcast_tensors and self._is_last_reduce_axis:
                insn_para = {"scope": write_buffer.op.axis[self._reduce_axis_num],
                             "instruction": insn}
            # 3.4 elewise
            else:
                insn_para = {"scope": write_buffer.op.axis[0], "instruction": insn}

            self._emit_insn_map[write_buffer] = insn_para


    def _do_emit_insn(self):
        softmax_compile_info = get_compile_info()
        self._coexisting_quantity = 4
        if softmax_compile_info["kernel_name"] in ["SoftmaxV2", "SoftmaxGrad"]:
            tensor_space = self._ub_tiling_max_size // self._coexisting_quantity
            tensor_space = tensor_space // 2
            is_log = False
        else:
            tensor_space = (self._ub_tiling_max_size - EXTRA_UB_SIZE) // self._coexisting_quantity
            is_log = True
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        for stage in self._emit_insn_map:
            scope_iter_var = self._emit_insn_map[stage]["scope"]
            instruction = self._emit_insn_map[stage]["instruction"]
            if instruction == "vector_reduce_sum" or instruction == "vector_reduce_max":
                if is_log:
                    self._schedule[stage].emit_insn(scope_iter_var, instruction, attrs=dict(storage_bound=[2048]))
                else:
                    self._schedule[stage].emit_insn(scope_iter_var, instruction, attrs=dict(storage_bound=[self._tensor_space]))
            else:
                self._schedule[stage].emit_insn(scope_iter_var, instruction)
            storage_bound = self._tensor_space // DTYPE_BYTE_MAPPING[stage.dtype]
            self._schedule[stage].set_storage_bound(storage_bound)

        # res dma_copy, if self._is_multi_core and >=32B:
        ub_inner = self._tiling_result["ub_tiling"]["inner_itervar"]
        # nlst pragma: ub_inner
        # last pragma: reduce_axis
        if not self._is_last_reduce_axis:
            ub_inner = self._res.op.axis[self._reduce_axis_num]
        self._schedule[self._res].emit_insn(ub_inner, 'dma_copy', {"no_overlap": 1})


    def _do_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        if self._is_double_buffer:
            for i in self._double_buffer_tensors:
                self._schedule[i].double_buffer()


    def _get_instruction(self, tensor):
        """
        Get the instruction map of tensor
        get insn for backend instruction map first (self._insn_map),
        then front-end instruction map (self._reg_insn_map)

        Parameters:
        ----------
        None

        Returns
        -------
        Instruction map string
        """
        str_list = tensor.op.tag.split("|")
        insn = self._insn_map.get(str_list[0])
        if insn and self._check_cast_support(tensor):
            return insn

        insn = self._reg_insn_map.get(str_list[0])
        return insn


    # pylint: disable=no-self-use
    def _check_cast_support(self, tensor):
        """
        Judge if tensor supports cast instruction operations,
        because of backend instruction bug of cast

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