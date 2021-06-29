# Copyright 2021 Huawei Technologies Co., Ltd
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
layer_norm_x_backprop_v2 schedule
"""

from tbe import dsl
from tbe import tvm
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base.operation import add_compile_info

from . import util
from .constants import DTYPE_BYTE_MAPPING
from .constants import INSN_MAPPING
from .constants import FAKE_NODE_TAG
from .constants import Pattern
from .layer_norm_x_backprop_v2_tilingcase import TilingStrategy

# block size in D architecture
BLOCK_SIZE_BYTE = 32

# temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192

@register_schedule(pattern=Pattern.LAYER_NORM_X_BACKPROP_V2)
def schedule(outs, tiling_case):
    """
    schedule for layer_norm_x_backprop_v2 dynamic shape
    """

    return LayerNormXBackpropScheduleV2(outs, tiling_case).do_schedule()

def _map_append(input_map, key, value):
    if input_map.get(key):
        if isinstance(value, list):
            for sub_v in value:
                if sub_v not in input_map[key]:
                    input_map[key].append(sub_v)
        else:
            if value not in input_map[key]:
                input_map[key].append(value)
    else:
        if isinstance(value, list):
            input_map[key] = value
        else:
            input_map[key] = [value]

def gen_reversed_subgraph_list(out_tensor, tensor_list_map, tensor_list_dst_tensor_map):
    """traverse tensors by Depth-First-Search
    Parameters
    ----------
    out_tensor : tensor
        traverse tensors from this tensor,
        traversing its input tensors recursively.
    tensor_list : list
    """
    if out_tensor is None:
        return
    stack = [out_tensor]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_list_map[in_tensor.name] = in_tensor
            _map_append(tensor_list_dst_tensor_map, in_tensor, cur_tensor)


class LayerNormXBackpropScheduleV2:
    """
    LayerNormXBackpropScheduleV2
    """

    def __init__(self, outs, tiling_case):
        self._out = None
        self._outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._tiling_key = self._tiling_case.get("key")
        self._open_double_buffer = self._tiling_case.get("is_need_db", False)

        self._scope = "local.UB"

        self._in_out_map = {}
        self._input_tensors = set()
        self._middle_tensors = set()
        self._out_tensors = set()

        self._broadcast_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._broadcast_axis_num = {}

        self._cache_read_tensors = set()
        self._cache_read_buffer_tensor_map = {}
        self._placeholder_tensor_map = {}
        
        self._cache_write_tensors = set()
        self._fake_middle_tensors = set()
        self._cache_write_buffer_tensor_map = {}
        self._cache_write_tensor_map = {}
        self._tensor_list_map = {}
        self._tensor_list_dst_tensor_map = {}
        self._input_tensor_dst_tensor_map = {}
        self._mid_tensor_dst_tensor_map = {}
        self._middle_out_cache_write_buffer_map = {}
        self._middle_out_cache_read_buffer_map = {}

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._ub_size = util.get_ub_size()
        self._correct_factor = 2 if self._open_double_buffer else 1
        self._tmp_ub_size = 0

        self._compute_at_map = {}

        self._block_bind_axis = None
        self._compute_at_axis = None
        self._compute_at_axis_idx = None
        self._emit_insn_axis = None
        self.sum_x_block_outer = None

        self._ir_axes = []
        self._inner_shape = []

        self._emit_insn_map = {}


    def do_schedule(self):
        """
        do schedule
        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case["key"]

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()
        self._do_compute_inline()

        self._calc_storage_bound()
        self._do_storage_bound()

        self._calc_tiling()
        self._do_tiling()

        self._calc_compute_at()
        self._do_compute_at()
        self._do_multi_core()

        self._calc_double_buffer()
        self._do_double_buffer()

        self._calc_emit_insn()
        self._do_emit_insn()
        self._add_compile_info()

        return self._schedule

    def _fake_node(self, _out_tensors):
        if len(self._outs) != 2:
            raise RuntimeError("real out tensors only have two")
        pd_x, res_for_gamma = _out_tensors
        if pd_x.dtype == "float16":
            pd_x_ub = dsl.cast_to(pd_x, res_for_gamma.dtype)
            res = dsl.vadd(pd_x_ub, res_for_gamma)
        else:
            res = dsl.vadd(pd_x, res_for_gamma)
        return res

    def _do_compute_inline(self):
        pass

    def _construct_compute_graph(self):
        def match_scalar_scene(tensor_):
            # condition:
            # 1. tensor --> tensor
            # 2. broadcast tensor is output
            # 3. next compute support scalar
            if len(tensor_.op.input_tensors) != 0 and \
                    util.get_tensor_size(tensor_.op.input_tensors[0]) != 1:
                return False
            if tensor_ in self._out_tensors:
                return False
            if all(util.support_scalar(tensor_o) for tensor_o in
                   self._in_out_map.get(tensor_)):
                return True
            return False

        def _no_broadcast(_src_shapes, _dst_shapes):
            _src_shapes = util.shape_to_list(_src_shapes)
            _dst_shapes = util.shape_to_list(_dst_shapes)
            broadcast_num = 0
            for x, y in zip(_src_shapes, _dst_shapes):
                if not expr_equal(x, y):
                    broadcast_num += 1
            return broadcast_num

        self._out_tensors = set(self._outs)
        visited_tensors = set()

        for out in self._out_tensors:
            self._dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors
        self._middle_out_tensors = self._middle_tensors.intersection(self._out_tensors)
        self._out = self._fake_node(list(self._out_tensors))
        self._dfs_sub_graph(self._out, visited_tensors)
        self._fake_middle_tensors = self._middle_tensors - self._pure_middle_tensors - self._out_tensors - self._middle_out_tensors

        for tensor_i in self._broadcast_tensors:
            if match_scalar_scene(tensor_i):
                self._absorbable_broadcast_tensors.add(tensor_i)

        for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
            if tensor_i.op.tag != "broadcast":
                src_shapes = tensor_i.op.input_tensors[0].shape[0:]
                dst_shapes = tensor_i.shape[0:]
                self._broadcast_axis_num[tensor_i] = _no_broadcast(src_shapes, dst_shapes)

        for tensor in self._tensor_list_dst_tensor_map:
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                self._input_tensor_dst_tensor_map[tensor] = self._tensor_list_dst_tensor_map[tensor]
            else:
                self._mid_tensor_dst_tensor_map[tensor] = self._tensor_list_dst_tensor_map[tensor]

    def _dfs_sub_graph(self, out, visited_tensors: set):
        for tensor_i in out.op.input_tensors:
            util.merge_value(self._in_out_map, tensor_i, out)
            self._dtypes.add(tensor_i.dtype)

            if util.is_placeholder(tensor_i):
                self._input_tensors.add(tensor_i)
            else:
                self._middle_tensors.add(tensor_i)

                if tensor_i.op.tag.find("unified_broadcast") != -1:
                    self._broadcast_tensors.add(tensor_i)

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self._dfs_sub_graph(tensor_i, visited_tensors)

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)
        self._cache_read_tensors.update(self._middle_out_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(
                tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._placeholder_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_read_buffer_map[tensor_i] = buffer_tensor

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._out_tensors)

    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[buffer_tensor] = tensor_i
            self._cache_write_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_write_buffer_map[tensor_i] = buffer_tensor

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._pure_middle_tensors:
            sch[tensor_i].set_scope(self._scope)

    def _calc_storage_bound(self):

        self._coexisting_quantity = 9
        if self._coexisting_quantity == 1:
            self._tmp_ub_size += BLOCK_SIZE_BYTE
        if len(self._broadcast_tensors) > 0:
            self._tmp_ub_size += BLOCK_SIZE_BYTE

    def _do_storage_bound(self):

        self._ub_size -= self._correct_factor * self._tmp_ub_size
        tensor_space = self._ub_size // self._coexisting_quantity
        if self._open_double_buffer:
            tensor_space = tensor_space // 2
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE
        
        sch = self._schedule
        tensors = self._middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_storage_bound(storage_bound)

    def _calc_tiling(self):
        funcs = {TilingStrategy.NONE_CUT: self._calc_tiling_none_cut}
        funcs[self._tiling_strategy]()

    def _calc_tiling_none_cut(self):
        pass

    def _do_tiling(self):
        funcs = {TilingStrategy.NONE_CUT: self._do_tiling_none_cut}
        funcs[self._tiling_strategy]()

    def _do_tiling_none_cut(self):
        res = self._out
        core_num = util.get_core_num()
        block_split_axis_index = 0

        block_split_axis = res.op.axis[block_split_axis_index]
        self.sum_x_block_outer, _ = self._schedule[res].split(
            block_split_axis, nparts=core_num
        )
        self._is_split_ub = True
        ub_factor = 1
        self._ub_split_axis_index = 1
        if self._is_split_ub:
            ub_outer, ub_inner = self._schedule[res].split(
                res.op.axis[self._ub_split_axis_index], factor=ub_factor
            )
        self._compute_at_axis = ub_outer
        self._emit_insn_axis = ub_inner

    def _calc_multi_core(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            self._block_bind_axis = self.sum_x_block_outer

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _calc_compute_at(self):
        for tensor_i in self._input_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._middle_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_read_buffer_tensor_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_write_buffer_tensor_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

    def _do_compute_at(self):
        sch = self._schedule
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1])

    def _calc_double_buffer(self):
        pass

    def _do_double_buffer(self):
        if self._open_double_buffer:
            sch = self._schedule

            tensors = self._middle_tensors \
                .union(self._cache_read_buffer_tensor_map.keys()) \
                .union(self._cache_write_buffer_tensor_map.keys())

            for tensor_i in tensors:
                sch[tensor_i].double_buffer()

    def _calc_emit_insn(self):
        def _get_emit_insn_map(tensor_):
            tag = tensor_.op.tag
            if tensor_.op.tag.find("|") != -1:
                insn = tag.split("|")[0]
            else:
                insn = tag
            return INSN_MAPPING.get(insn, insn)

        for source, target in self._cache_read_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], "dma_copy"]

        for tensor_i in self._pure_middle_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], _get_emit_insn_map(tensor_i)]

        for source, target in self._cache_write_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[2], _get_emit_insn_map(target)]

        for tensor_i in self._out_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], "dma_copy"]

        for tensor_i in self._fake_middle_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], "phony_insn"]

        if len(self._out_tensors) - len(self._middle_out_tensors) > 1:
            self._emit_insn_map[self._out] = [self._emit_insn_axis, "phony_insn"]
        else:
            self._emit_insn_map[self._out] = [self._emit_insn_axis, "dma_copy"]

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            sch[tensor_i].emit_insn(param[0], param[1])

    def _add_compile_info(self):
        add_compile_info("UB_SIZE", self._ub_size)
        add_compile_info("CORE_NUM", util.get_core_num())
        add_compile_info("MAX_DTYPE", self._max_dtype_bytes)
        add_compile_info("COEXISTING_QUANTITY", self._coexisting_quantity)
