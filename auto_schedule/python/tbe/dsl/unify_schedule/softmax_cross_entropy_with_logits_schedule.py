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
elewise schedule
"""
from typing import Optional

import tbe
from tbe import tvm
import te.lang.cce
from te import platform as cce

from .constants import CompileInfo
from .constants import DTYPE_BYTE_MAPPING
from .constants import FAKE_NODE_TAG
from .constants import INSN_MAPPING
from .constants import Pattern
from .constants import SUPPORT_SCALAR_INSNS
from .constants import TERNARY_INSNS

from . import util
from .softmax_cross_entropy_with_logits_tilingcase import TilingStrategy
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base import operation

# block size in D architecture
BLOCK_SIZE_BYTE = 32

CONST = "const"
VECTOR = "vector"
PHONY = "phony"


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


def gen_reversed_subgraph_list(out_tensor, tensor_list_map,
                               tensor_list_dst_tensor_map):
    """traverse tensors by Depth-First-Search

    Parameters
    ----------
    out_tensor : tensor
        traverse tensors from this tensor,
        traversing its input tensors recursively.

    tensor_list : list
        record tensors in the order of Depth-First-Search.

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


@register_schedule(pattern=Pattern.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS)
def schedule(outs, tiling_case):
    """
    :param outs:
    :param tiling_case:
    :return:
    """
    return SoftmaxCrossEntropyWithLogitsSchedule(outs, tiling_case).do_schedule()


# 'pylint: disable=R0902, R0903
class SoftmaxCrossEntropyWithLogitsSchedule:
    """
    SoftmaxCrossEntropyWithLogitsSchedule
    """

    def __init__(self, outs, tiling_case):
        self._outs = outs
        self._out = None  # type: Optional[tvm.tensor.Tensor]
        self._reduce_ext = None
        self._out_ub = None

        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._is_db = self._tiling_case.get("is_need_db", False)
        self._mode = operation.get_context().get("mode")

        self._scope = "local.UB"

        self._input_tensors = set()
        self._middle_tensors = set()
        self._pure_middle_tensors = set()
        self._middle_out_tensors = set()
        self._out_tensors = set()

        self._broadcast_tensors = set()
        self._reduce_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._compute_inline_broadcast = set()

        self._reduce_axis_index = -1
        self._reduce_axis_value = -1

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._tensor_space = None
        self._ub_size = util.get_ub_size()

        # static schedule var
        self._tensor_list_map = {}
        self._tensor_list_dst_tensor_map = {}

        self._input_tensor_dst_tensor_map = {}
        self._mid_tensor_dst_tensor_map = {}
        self._mid_out_tensor_list = []
        self._mid_out_buffer_tensor_list = {}

        self._input_tensor_buffer_tensor_map = {}
        self._mid_tensor_buffer_tensor_map = {}

        # input -> outputs mapping relations
        self._in_out_map = {}

        self._cache_read_tensors = set()
        self._cache_read_buffer_tensor_map = {}
        self._placeholder_tensor_map = {}
        self._middle_out_cache_read_buffer_map = {}

        self._cache_write_tensors = set()
        self._cache_write_buffer_tensor_map = {}
        self._cache_write_tensor_map = {}
        self._middle_out_cache_write_buffer_map = {}

        self._compute_inline_tensors = set()

        self._compute_at_map = {}

        # just for const tiling
        self._need_do_block = False
        self._block_dims = 1
        self._block_split_axis = -1
        self._block_factor = 1
        self._ub_split_axis = 0
        self._ub_factor = 1

        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._block_bind_axis = None
        self._compute_at_axis = None
        self._compute_at_axis_idx = None
        self._emit_insn_axis = None

        self._ir_axes = []
        self._inner_shape = []

        self._constraints = set()

        self._mem_reuse_map = {}
        self._data_reuse_map = {}

        self._emit_insn_map = {}

    def do_schedule(self):
        """
        :return:
        """
        self._get_reduce_axis_info()
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case["key"]

        self._do_cache_read()
        self._do_cache_write()

        self._calc_storage_bound()
        self._do_storage_bound()

        self._do_compute_inline()

        self._storage_align()

        self._calc_tiling()
        self._do_tiling()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_constraints()
        self._do_constraints()

        self._calc_emit_insn()
        self._do_emit_insn()
        self._add_compile_info()

        return self._schedule if self._check_tiling_case() else None

    def _get_reduce_axis_info(self):
        def get_reduce_axis_index(outs):
            tensor_0, tensor_1 = outs
            reduce_axis_index = -1
            reduce_axis_value = 1
            tensor_0_shape = util.shape_to_list(tensor_0.shape)
            tensor_1_shape = util.shape_to_list(tensor_1.shape)
            for i in range(len(tensor_0_shape)):
                if tensor_0_shape[i] != tensor_1_shape[i]:
                    reduce_axis_index = i
                    reduce_axis_value = tensor_0_shape[i] if tensor_0_shape[i] != 1 else tensor_1_shape[i]
            return reduce_axis_index, reduce_axis_value

        self._reduce_axis_index, self._reduce_axis_value = get_reduce_axis_index(self._outs)

    def _construct_compute_graph(self):
        def match_scalar_scene(tensor_):
            if len(tensor_.op.input_tensors) != 0 and \
                    util.get_tensor_size(tensor_.op.input_tensors[0]) != 1:
                return False
            if tensor_ in self._out_tensors:
                return False
            if all(util.support_scalar(tensor_o) for tensor_o in
                   self._in_out_map.get(tensor_)):
                return True
            return False

        self._out_tensors = set(self._outs)
        self._out = self._fake_node(self._out_tensors, self._reduce_axis_index)

        visited_tensors = set()
        for out in self._out_tensors:
            if util.is_broadcast(out):
                self._broadcast_tensors.add(out)
            if util.is_reduce_tensor(out):
                self._reduce_tensors.add(out)
            self.__dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)

        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors

        output_loss, output_backprop = self._out_tensors
        self.__dfs_sub_graph(self._out, visited_tensors)

        gen_reversed_subgraph_list(self._out, self._tensor_list_map, self._tensor_list_dst_tensor_map)

        self._mid_out_tensor_list = [output_loss, output_backprop]

        for tensor in self._tensor_list_dst_tensor_map:
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                self._input_tensor_dst_tensor_map[tensor] = \
                    self._tensor_list_dst_tensor_map[tensor]
            else:
                self._mid_tensor_dst_tensor_map[tensor] = \
                    self._tensor_list_dst_tensor_map[tensor]

    def _do_cache_read(self):
        for tensor in self._input_tensor_dst_tensor_map:
            tensor_ub = self._schedule.cache_read(tensor, cce.scope_ubuf,
                                                  self._input_tensor_dst_tensor_map[tensor])
            self._input_tensor_buffer_tensor_map[tensor] = tensor_ub
            self._cache_read_buffer_tensor_map[tensor_ub] = tensor

        for tensor in self._mid_out_tensor_list:
            tensor_ub = self._schedule.cache_read(tensor, cce.scope_ubuf,
                                                  self._mid_tensor_dst_tensor_map[tensor])
            self._mid_out_buffer_tensor_list[tensor] = tensor_ub
            self._cache_read_buffer_tensor_map[tensor_ub] = tensor

    def _do_cache_write(self):
        for tensor in self._mid_tensor_dst_tensor_map:
            tensor_ub = self._schedule.cache_write(tensor, cce.scope_ubuf)
            self._mid_tensor_buffer_tensor_map[tensor] = tensor_ub

            self._cache_write_buffer_tensor_map[tensor_ub] = tensor
            self._cache_write_tensor_map[tensor] = tensor_ub

        self._out_ub = self._schedule.cache_write(self._out, cce.scope_ubuf)

    def _calc_tiling(self):
        funcs = {
            TilingStrategy.ONE_CUT: self._calc_tiling_one_cut,
        }
        funcs[self._tiling_strategy]()

    def _calc_tiling_one_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._tiling_case["block_tiling_axis"]
        u_i = self._tiling_case["ub_tiling_axis"]
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_bound = self._tiling_case.get("ub_factor_bound")
        if u_bound is None:
            u_bound = (1, util.get_bound(shape[u_i])[1])
        self._block_tiling_vars[b_i] = operation.var("block_nparts_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = operation.var("ub_factor_" + str(u_i), u_bound)

    def _do_tiling(self):
        funcs = {
            TilingStrategy.ONE_CUT: self._do_tiling_one_cut,
        }
        funcs[self._tiling_strategy]()

    def _do_tiling_one_cut(self):
        sch = self._schedule
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]
        block_axes = []
        ub_axes = []
        inner_axes = []
        b_o, b_i = sch[res].split(res.op.axis[0],
                                  nparts=self._block_tiling_vars[0])
        block_axes.append([b_o, b_idx])
        u_o, u_i = sch[res].split(b_i, factor=self._ub_tiling_vars[u_idx])
        ub_axes.append([u_o, u_idx])
        inner_axes.append([u_i, u_idx])
        self._inner_shape.append([self._ub_tiling_vars[u_idx], u_idx])
        if len(res.op.axis) == 2:
            inner_axes.append([res.op.axis[1], 1])
            self._inner_shape.append([shape[1], 1])

        self._ir_axes = block_axes + ub_axes + inner_axes
        self._block_bind_axis = sch[res].fuse(*[x[0] for x in block_axes])
        self._compute_at_axis = ub_axes[-1][0]
        self._compute_at_axis_idx = ub_axes[-1][1]
        self._emit_insn_axis = inner_axes[0][0]

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor in self._mid_tensor_dst_tensor_map:
            if tensor not in self._mid_out_tensor_list:
                sch[tensor].compute_inline()

    def _storage_align(self):
        block_size_align = 8 if self._out.dtype == "float32" else 16

        # when reduce axis is 32B multiple, the op doesn't need storage align
        storage_align_flag = 1
        for i, (tensor_i, param) in enumerate(self._mid_tensor_buffer_tensor_map.items()):
            if tensor_i.op.name[0:6] != "reduce":
                shape_i = util.shape_to_list(tensor_i.shape)
                storage_align_flag = 0 if shape_i[1] % block_size_align == 0 else storage_align_flag

        if storage_align_flag:
            for i, (tensor_i, param) in enumerate(self._mid_tensor_buffer_tensor_map.items()):
                if param.op.name[0:6] != "reduce":
                    self._schedule[param].storage_align(param.op.axis[0], block_size_align, 0)

            for i, (tensor_i, param) in enumerate(self._cache_read_buffer_tensor_map.items()):
                if tensor_i.op.name[0:6] != "reduce":
                    self._schedule[tensor_i].storage_align(tensor_i.op.axis[0], block_size_align, 0)

    def _calc_multi_core(self):
        pass

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _calc_compute_at(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT or \
                (self._tiling_strategy == TilingStrategy.CONST and not self._need_do_block):
            self._compute_at_map.clear()
            return

        for tensor_i in self._input_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._middle_tensors - self._compute_inline_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_read_buffer_tensor_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_write_buffer_tensor_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

    def _do_compute_at(self):
        sch = self._schedule
        if not self._compute_at_map:
            return
        for tensor in self._input_tensor_dst_tensor_map:
            tensor_ub = self._input_tensor_buffer_tensor_map[tensor]
            sch[tensor_ub].compute_at(sch[self._out], self._compute_at_axis)

        for tensor in self._mid_out_tensor_list:
            tensor_ub = self._mid_out_buffer_tensor_list[tensor]
            sch[tensor].compute_at(sch[self._out], self._compute_at_axis)
            sch[tensor_ub].compute_at(sch[self._out], self._compute_at_axis)

        for tensor in self._mid_tensor_dst_tensor_map:
            tensor_ub = self._mid_tensor_buffer_tensor_map[tensor]
            sch[tensor_ub].compute_at(sch[self._out], self._compute_at_axis)

        sch[self._out_ub].compute_at(sch[self._out], self._compute_at_axis)

    def _calc_constraints(self):
        for tensor_i in self._broadcast_tensors:
            if tensor_i.op.tag == "unknown_broadcast":
                src_shapes = tensor_i.op.input_tensors[0].shape
                dst_shapes = tensor_i.shape
                for src_shape, dst_shape in zip(src_shapes, dst_shapes):
                    if src_shape != dst_shape:
                        self._constraints.add(src_shape <= dst_shape)
                operation.add_build_arg(
                    "constant_realize_extent_in_infer_bound", False)

    def _do_constraints(self):
        sch = self._schedule
        for cond in self._constraints:
            sch.set_constraint(cond)

    def _calc_emit_insn(self):
        def get_insn(tensor_):
            tag = tensor_.op.tag
            if tensor_.op.tag.find("|") != -1:
                insn = tag.split("|")[0]
            else:
                insn = tag
            return INSN_MAPPING.get(insn, insn)

        for source, target in self._cache_read_buffer_tensor_map.items():
            if target in self._middle_out_tensors:
                self._emit_insn_map[source] = [source.op.axis[0], "phony_insn"]
            else:
                self._emit_insn_map[source] = [source.op.axis[0], "dma_copy"]

        for tensor_i in (self._pure_middle_tensors - self._compute_inline_tensors):
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], get_insn(tensor_i)]
            if tensor_i in self._compute_inline_broadcast:
                self._emit_insn_map[tensor_i].append("phony_insn")

        for source, target in self._cache_write_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], get_insn(target)]
            if target in self._compute_inline_broadcast:
                self._emit_insn_map[source].append("phony_insn")

        if len(self._out_tensors) > 1:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], "dma_copy"]
            self._emit_insn_map[self._out] = [self._emit_insn_axis, "phony_insn"]
        else:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [self._emit_insn_axis, "dma_copy"]
        self._emit_insn_map[self._out_ub] = [self._out_ub.op.axis[0], "phony_insn"]

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            if param[1] == "unknown_broadcast":
                u_idx = self._tiling_case["ub_tiling_axis"]
                src_shapes = tensor_i.op.input_tensors[0].shape[u_idx:]
                tensor_bound = self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype]
                src_shape = tvm.expr.Call('handle', 'tvm_tuple', src_shapes,
                                          tvm.expr.Call.PureIntrinsic, None, 0)
                sch[tensor_i].emit_insn(param[0], param[1],
                                        attrs=dict(src_shape=src_shape, storage_bound=[tensor_bound]))
            else:
                if tensor_i.op.name in ["sub_7.local.UB", "reduce_2.local.UB"] and param[1][0:3] == 'dma':
                    sch[tensor_i].emit_insn(param[0], "phony_insn")
                elif tensor_i.op.name == "reduce_3.local.UB":
                    sch[tensor_i].emit_insn(param[0], "phony_insn")
                else:
                    if param[1][0:6] == VECTOR:
                        tensor_bound = self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype]
                        sch[tensor_i].emit_insn(param[0], param[1], attrs=dict(storage_bound=[tensor_bound]))
                    else:
                        if self._is_db and tensor_i in self._out_tensors:
                            sch[tensor_i].emit_insn(param[0], param[1], attrs=dict(no_overlap=0))
                        else:
                            sch[tensor_i].emit_insn(param[0], param[1])

    def _calc_double_buffer(self):
        pass

    def _do_double_buffer(self):
        pass

    def _calc_mem_reuse(self):
        ternary_reuse_map = {
            "elewise_binary_scalar_axpy": 1,
            "elewise_multiple_mla": 2,
            "elewise_multiple_madd": 1,
            "elewise_multiple_maddrelu": 1,
        }

        def __get_ub_tensor(_input_tensor, _output_tensor):
            if _input_tensor in self._placeholder_tensor_map:
                _input_tensor = self._placeholder_tensor_map[_input_tensor]
            if _output_tensor in self._cache_write_tensor_map:
                _output_tensor = self._cache_write_tensor_map[_output_tensor]
            return _input_tensor, _output_tensor

        # one of the input of the ternary instruction must be reused with the output, refer to "ternary_reuse_map"
        # consider "vmadd": A=A*A+B, output reuse the second A, input_tensors is 2, which need to be completed to 3
        for tensor_i in self._out_tensors | (self._pure_middle_tensors - self._compute_inline_tensors):
            insn = util.get_dsl_insn(tensor_i)
            args = ""
            if tensor_i.op.tag.find("|") != -1:
                args = tensor_i.op.tag.split("|")[1].split(",")
            if insn in TERNARY_INSNS:
                src_tensors = []
                index = 0
                first_same_tensor = None
                for i in args:
                    if i == "1":
                        if first_same_tensor not in src_tensors:
                            first_same_tensor = tensor_i.op.input_tensors[index]
                            index += 1
                        src_tensors.append(first_same_tensor)
                    else:
                        src_tensors.append(tensor_i.op.input_tensors[index])
                        index += 1
                reuse_index = ternary_reuse_map.get(insn)
                src_tensor = src_tensors[reuse_index]
                src_tensor, dst_tensor = __get_ub_tensor(src_tensor, tensor_i)
                util.merge_value(self._mem_reuse_map,
                                 src_tensor,
                                 dst_tensor)
        for tensor_i, write_buffer in \
                self._middle_out_cache_write_buffer_map.items():
            util.merge_value(self._mem_reuse_map,
                             self._middle_out_cache_read_buffer_map[tensor_i],
                             write_buffer)
        for tensor_i in self._compute_inline_broadcast:
            input_tensor = tensor_i.op.input_tensors[0]
            input_tensor, broadcast_tensor = __get_ub_tensor(input_tensor, tensor_i)
            util.merge_value(self._data_reuse_map,
                             input_tensor,
                             broadcast_tensor)

    def _do_mem_reuse(self):
        sch = self._schedule
        for _a, _b in self._mem_reuse_map.items():
            for b_i in _b:
                sch[_a].reused_by(b_i)
        for _a, _b in self._data_reuse_map.items():
            for b_i in _b:
                sch[_a].reused_by(b_i)
                sch[b_i].reused_by(reuse_data=True)

    def _calc_storage_bound(self):
        self._coexisting_quantity = 10
        tensor_space = self._ub_size // self._coexisting_quantity

        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_storage_bound(storage_bound)

    def _check_tiling_case(self):
        lower_bound = 1
        for item in self._inner_shape[::-1]:
            cur_bound = util.get_bound(item[0])[0]
            if cur_bound is None:
                return False
            lower_bound *= cur_bound
        if not self._tensor_space // self._max_dtype_bytes >= lower_bound:
            return False
        return True

    def _add_compile_info(self):
        cpt_compute = operation.get_context().get_current_compute()
        cpt_schedule = cpt_compute.get_current_schedule()
        if self._mode == CONST:
            # const shape: one compute, one schedule
            cpt_compute.add("const_block_dim", self._block_dims)
        else:
            cpt_schedule.add(CompileInfo.MAX_DTYPE, self._max_dtype_bytes)
            cpt_schedule.add(CompileInfo.COEXISTING_QUANTITY, self._coexisting_quantity)
            cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
            cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())

    def __dfs_sub_graph(self, out, visited_tensors: set):
        for tensor_i in out.op.input_tensors:
            util.merge_value(self._in_out_map, tensor_i, out)
            self._dtypes.add(tensor_i.dtype)

            if util.is_placeholder(tensor_i):
                self._input_tensors.add(tensor_i)
            else:
                self._middle_tensors.add(tensor_i)
                if util.is_broadcast(tensor_i):
                    self._broadcast_tensors.add(tensor_i)
                if util.is_reduce_tensor(tensor_i):
                    self._reduce_tensors.add(tensor_i)

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self.__dfs_sub_graph(tensor_i, visited_tensors)

    def _fake_node(self, tensors, reduce_axis_index):
        if len(self._outs) != 2:
            raise RuntimeError("real out tensors only have two")
        output_loss_reduce_out, output_backprop_sub_out = tensors
        self._reduce_ext = tbe.dsl.reduce_sum(output_backprop_sub_out, axis=reduce_axis_index, keepdims=True)
        res = tbe.dsl.vadd(self._reduce_ext, output_loss_reduce_out)
        return res
