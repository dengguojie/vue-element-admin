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
from functools import reduce
from math import ceil

from te import tvm

from . import Pattern, INSN_MAPPING, DTYPE_BYTE_MAPPING, FAKE_NODE_TAG, BROADCAST_INSNS
from . import CompileInfo
from . import util
from .elewise_tilingcase import TilingStrategy
from te.platform.operation import register_schedule
from te.platform.operation import add_compile_info
from te.platform.operation import get_compile_info
from te.platform import operation

# block size in D architecture
BLOCK_SIZE_BYTE = 32
MULTI_CORE_THRESHOLD = 1024

N_LAST_BROADCAST_THRESHOLD = 512

CONST = "const"
VECTOR = "vector"
SPECIAL_SCALAR = "special_scalar"

# vcmpsel constant
VSEL_INPUT_NUMBER = 3
VCMPSEL_INPUT_NUMBER = 4

@register_schedule(pattern=Pattern.ELEMWISE)
def schedule(outs, tiling_case):
    """
    :param outs:
    :param tiling_case:
    :return:
    """
    return ElewiseSchedule(outs, tiling_case).do_schedule()


# 'pylint: disable=R0902, R0903
class ElewiseSchedule:
    """
    ElewiseSchedule
    """

    def __init__(self, outs, tiling_case):
        self._out = None  # type: Optional[tvm.tensor.Tensor]
        self._outs = outs
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._mode = operation.get_context().get("mode")

        self._scope = "local.UB"

        self._input_tensors = set()
        self._middle_tensors = set()
        self._pure_middle_tensors = set()
        self._middle_out_tensors = set()
        self._out_tensors = set()

        self._broadcast_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._compute_inline_broadcast = set()

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._tensor_space = None
        self._ub_size = util.get_ub_size()

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

        self._db_tensors = set()

        # just for const tiling
        self._need_do_block = False
        self._block_dims = 1
        self._block_split_axis = -1
        self._block_factor = 1
        self._ub_split_axis = -1
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
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case["key"]

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._calc_storage_bound()
        self._do_storage_bound()

        self._set_scope()

        self._calc_tiling()
        self._do_tiling()

        self._calc_compute_inline()
        self._do_compute_inline()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_double_buffer()
        self._do_double_buffer()

        self._calc_mem_reuse()
        self._do_mem_reuse()

        self._calc_constraints()
        self._do_constraints()

        self._calc_emit_insn()
        self._do_emit_insn()
        self._add_compile_info()

        return self._schedule if self._check_tiling_case() else None

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

        self._out_tensors = set(self._outs)

        visited_tensors = set()
        for out in self._out_tensors:
            if util.is_broadcast(out):
                self._broadcast_tensors.add(out)
            self.__dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors
        self._middle_out_tensors = self._middle_tensors.intersection(
            self._out_tensors)

        pure_out_tensors = list(self._out_tensors - self._middle_out_tensors)
        if len(pure_out_tensors) > 1:
            self._out = _fake_node(pure_out_tensors)
        else:
            self._out = pure_out_tensors[0]

        for tensor_i in self._broadcast_tensors:
            if match_scalar_scene(tensor_i):
                self._absorbable_broadcast_tensors.add(tensor_i)

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)
        self._cache_read_tensors.update(self._middle_out_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(
                tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._placeholder_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._input_tensors:
                self._db_tensors.add(buffer_tensor)
            elif tensor_i in self._middle_out_tensors:
                self._middle_out_cache_read_buffer_map[tensor_i] = \
                    buffer_tensor

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._out_tensors)

    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[buffer_tensor] = tensor_i
            self._cache_write_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_write_buffer_map[tensor_i] = \
                    buffer_tensor

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._pure_middle_tensors:
            sch[tensor_i].set_scope(self._scope)

    def _calc_tiling(self):
        funcs = {TilingStrategy.ALL_CUT: self._calc_tiling_all_cut,
                 TilingStrategy.NONE_CUT: self._calc_tiling_none_cut,
                 TilingStrategy.ONE_CUT: self._calc_tiling_one_cut,
                 TilingStrategy.STATIC: self._calc_tiling_static,
                 TilingStrategy.CONST: self._calc_tiling_const,
                 }
        funcs[self._tiling_strategy]()

    def _calc_tiling_all_cut(self):
        res = self._out
        for _i, _x in enumerate(res.shape):
            bound = (1, util.get_bound(_x)[1])
            self._block_tiling_vars[_i] = operation.var("block_factor_" + str(_i), bound)
            self._ub_tiling_vars[_i] = operation.var("ub_factor_" + str(_i), bound)

    def _calc_tiling_none_cut(self):
        pass

    def _calc_tiling_one_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._tiling_case["block_tiling_axis"]
        u_i = self._tiling_case["ub_tiling_axis"]
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_bound = self._tiling_case.get("ub_factor_bound")
        if u_bound is None:
            u_bound = (1, util.get_bound(shape[u_i])[1])
        self._block_tiling_vars[b_i] = operation.var("block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = operation.var("ub_factor_" + str(u_i), u_bound)

    def _calc_tiling_static(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._tiling_case["block_tiling_axis"]
        u_i = self._tiling_case["ub_tiling_axis"]
        b_bound = (1, util.get_bound(shape[b_i])[1])
        self._block_tiling_vars[b_i] = operation.var("block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = self._tiling_case["ub_tiling_factor"]

    def _calc_tiling_const(self):
        def _get_type_size(dtype):
            type_size = 2
            if dtype == "int8" or dtype == "uint8":
                type_size = 1
            elif dtype == "float32" or dtype == "int32" or dtype == "uint32":
                type_size = 4
            elif dtype == "int64" or dtype == "uint64":
                type_size = 8
            elif dtype == "bool":
                type_size = 1
            return type_size

        def _calc_block_tiling():
            rank = len(shape)
            block_dims = 1
            if rank == 1:
                self._block_split_axis = 0
                type_size = _get_type_size(res.dtype)
                ele_in_block = BLOCK_SIZE_BYTE // type_size
                block_factor = ceil(output_size / util.get_core_num())
                self._block_factor = ceil((block_factor / ele_in_block)
                                          * ele_in_block)
                block_dims = ceil(shape[0] / self._block_factor)
                shape[0] = self._block_factor
            else:
                cur_core = util.get_core_num()
                for index, _shape in enumerate(shape):
                    if _shape > cur_core:
                        self._block_split_axis = index
                        self._block_factor = ceil(_shape / cur_core)
                        block_dims *= ceil(shape[index] / self._block_factor)
                        shape[index] = self._block_factor
                        break
                    else:
                        cur_core //= _shape
                        block_dims *= shape[index]
                        shape[index] = 1
            self._block_dims = block_dims

        def _find_bottoms_broadcast_axis():
            broadcast_axis = -(1 << (32-1))
            for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
                if tensor_i.op.input_tensors:
                    input = util.shape_to_list(tensor_i.op.input_tensors[0].shape)
                    output = util.shape_to_list(tensor_i.shape)
                    for axis in range(min(len(input), len(output))):
                        if input[-(axis + 1)] == 1 and output[-(axis + 1)] != 1:
                            cur_broadcast_axis = -(axis + 1)
                            if cur_broadcast_axis > broadcast_axis:
                                broadcast_axis = cur_broadcast_axis
                                break
            return broadcast_axis

        def _calc_ub_tiling():
            ub_limit = ((self._ub_size // self._coexisting_quantity
                         // BLOCK_SIZE_BYTE) * BLOCK_SIZE_BYTE) // \
                self._max_dtype_bytes
            type_size = _get_type_size(res.dtype)
            ele_in_block = BLOCK_SIZE_BYTE // type_size
            if len(shape) == 1:
                self._ub_split_axis = 0
                ub_factor = shape[0]
                ub_for_num = ceil(ub_factor / ub_limit)
                adjust_factor = ceil(ub_factor / ub_for_num)
                align_factor = ceil(adjust_factor / ele_in_block)
                self._ub_factor = align_factor * ele_in_block
                if self._ub_factor > ub_limit:
                    self._ub_factor = adjust_factor // ele_in_block * ele_in_block
            else:
                broadcast_axis = _find_bottoms_broadcast_axis()
                shape_len = len(shape) - 1
                under_broadcast_shape = 1
                broadcast_axis = len(shape) + broadcast_axis
                for index in range(shape_len, self._block_split_axis - 1, -1):
                    if broadcast_axis == index and broadcast_axis != shape_len:
                        is_miass_align = under_broadcast_shape % ele_in_block != 0
                        is_cut_under_b = (under_broadcast_shape > N_LAST_BROADCAST_THRESHOLD or
                                          (under_broadcast_shape > (ele_in_block * 2) and shape[index] < ele_in_block))\
                                         and is_miass_align
                        if is_cut_under_b:
                            self._ub_split_axis = index + 1
                            self._ub_factor = shape[index + 1]
                            break
                        if shape[index] > (ele_in_block * 3) and is_miass_align:
                            self._ub_split_axis = index
                            self._ub_factor = min(shape[index], ub_limit)
                            break
                    if shape[index] >= ub_limit:
                        self._ub_split_axis = index
                        self._ub_factor = ub_limit
                        break
                    else:
                        ub_limit //= shape[index]
                        under_broadcast_shape *= shape[index]
                if self._ub_split_axis < 0:
                    self._ub_split_axis = self._block_split_axis
                    self._ub_factor = shape[self._ub_split_axis]

        res = self._out
        shape = util.shape_to_list(res.shape)
        output_size = reduce(lambda x, y: x * y, shape)
        block_dims = 1
        if output_size > MULTI_CORE_THRESHOLD:
            self._need_do_block = True
            _calc_block_tiling()
            _calc_ub_tiling()
        else:
            self._ub_split_axis = 0
            self._ub_factor = shape[0]

    def _do_tiling(self):
        funcs = {TilingStrategy.ALL_CUT: self._do_tiling_all_cut,
                 TilingStrategy.NONE_CUT: self._do_tiling_none_cut,
                 TilingStrategy.ONE_CUT: self._do_tiling_one_cut,
                 TilingStrategy.STATIC: self._do_tiling_static,
                 TilingStrategy.CONST: self._do_tiling_const,
                 }
        funcs[self._tiling_strategy]()

    def _do_tiling_all_cut(self):
        sch = self._schedule
        res = self._out
        block_axes = []
        ub_axes = []
        inner_axes = []
        for _i, _x in enumerate(res.op.axis):
            x_o, x_i = sch[res].split(_x, factor=self._block_tiling_vars[_i])
            x_io, x_ii = sch[res].split(x_i, factor=self._ub_tiling_vars[_i])
            block_axes.append([x_o, _i])
            ub_axes.append([x_io, _i])
            inner_axes.append([x_ii, _i])
            self._inner_shape.append([self._ub_tiling_vars[_i], _i])
        self._ir_axes = block_axes + ub_axes + inner_axes
        ordered_axes = [x[0] for x in self._ir_axes]
        sch[res].reorder(*ordered_axes)
        self._block_bind_axis = sch[res].fuse(*[x[0] for x in block_axes])
        self._compute_at_axis = ub_axes[-1][0]
        self._compute_at_axis_idx = ub_axes[-1][1]
        self._emit_insn_axis = inner_axes[0][0]

    def _do_tiling_none_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        for _i, _x in enumerate(res.op.axis):
            self._ir_axes.append([_x, _i])
            self._inner_shape.append([shape[_i], _i])
        self._emit_insn_axis = res.op.axis[0]

    def _do_tiling_one_cut(self):
        sch = self._schedule
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]
        block_axes = []
        ub_axes = []
        inner_axes = []
        for i in range(b_idx):
            block_axes.append([res.op.axis[i], i])
        b_o, b_i = sch[res].split(res.op.axis[b_idx],
                                  factor=self._block_tiling_vars[b_idx])
        block_axes.append([b_o, b_idx])
        if b_idx == u_idx:
            u_o, u_i = sch[res].split(b_i, factor=self._ub_tiling_vars[u_idx])
            ub_axes.append([u_o, u_idx])
            inner_axes.append([u_i, u_idx])
            self._inner_shape.append([self._ub_tiling_vars[u_idx], u_idx])
        else:
            ub_axes.append([b_i, b_idx])
            for i in range(b_idx + 1, u_idx):
                ub_axes.append([res.op.axis[i], i])
            u_o, u_i = sch[res].split(res.op.axis[u_idx],
                                      factor=self._ub_tiling_vars[u_idx])
            ub_axes.append([u_o, u_idx])
            inner_axes.append([u_i, u_idx])
            self._inner_shape.append([self._ub_tiling_vars[u_idx], u_idx])
        for i in range(u_idx + 1, len(res.op.axis)):
            inner_axes.append([res.op.axis[i], i])
            self._inner_shape.append([shape[i], i])

        self._ir_axes = block_axes + ub_axes + inner_axes
        self._block_bind_axis = sch[res].fuse(*[x[0] for x in block_axes])
        self._compute_at_axis = ub_axes[-1][0]
        self._compute_at_axis_idx = ub_axes[-1][1]
        self._emit_insn_axis = inner_axes[0][0]

    def _do_tiling_static(self):
        self._do_tiling_one_cut()

    def _do_tiling_const(self):
        sch = self._schedule
        res = self._out
        block_axes = []
        if self._need_do_block:
            for i in range(self._block_split_axis):
                block_axes.append([res.op.axis[i], i])
            b_o, b_i = sch[res].split(res.op.axis[self._block_split_axis],
                                      factor=self._block_factor)
            block_axes.append([b_o, self._block_split_axis])
            self._block_bind_axis = sch[res].fuse(*[x[0] for x in block_axes])

        if self._block_split_axis == self._ub_split_axis:
            u_o, u_i = sch[res].split(b_i, factor=self._ub_factor)
        else:
            u_o, u_i = sch[res].split(res.op.axis[self._ub_split_axis],
                                      factor=self._ub_factor)
        self._compute_at_axis = u_o
        self._emit_insn_axis = u_i

    def _calc_compute_inline(self):
        def _no_broadcast(_src_shapes, _dst_shapes):
            _src_shapes = util.shape_to_list(_src_shapes)
            _dst_shapes = util.shape_to_list(_dst_shapes)
            is_no_broadcast = True
            for x, y in zip(_src_shapes, _dst_shapes):
                if x != y:
                    is_no_broadcast = False
                    break
            return is_no_broadcast

        self._compute_inline_tensors = \
            self._absorbable_broadcast_tensors.copy()
        if self._tiling_strategy == TilingStrategy.ONE_CUT:
            ub_idx = self._tiling_case["ub_tiling_axis"]
            for tensor_i in self._broadcast_tensors:
                if tensor_i.op.tag != "broadcast":
                    src_shapes = tensor_i.op.input_tensors[0].shape[ub_idx:]
                    dst_shapes = tensor_i.shape[ub_idx:]
                    if _no_broadcast(src_shapes, dst_shapes):
                        self._compute_inline_broadcast.add(tensor_i)
        if self._tiling_strategy == TilingStrategy.CONST:
            ub_idx = self._ub_split_axis
            for tensor_i in self._broadcast_tensors:
                if tensor_i.op.tag != "broadcast":
                    src_shapes = tensor_i.op.input_tensors[0].shape[ub_idx:]
                    dst_shapes = tensor_i.shape[ub_idx:]
                    if _no_broadcast(src_shapes, dst_shapes):
                        self._compute_inline_tensors.add(tensor_i)

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()

    def _calc_multi_core(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            self._block_bind_axis = None

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _calc_compute_at(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
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
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1])

    def _calc_constraints(self):
        for tensor_i in self._broadcast_tensors:
            if tensor_i.op.tag == "unknown_broadcast":
                src_shapes = tensor_i.op.input_tensors[0].shape
                dst_shapes = tensor_i.shape
                for src_shape, dst_shape in zip(src_shapes, dst_shapes):
                    if src_shape != dst_shape:
                        self._constraints.add(src_shape <= dst_shape)
                # add build args: constant_realize_extent_in_infer_bound
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

        for tensor_i in (self._pure_middle_tensors -
                         self._compute_inline_tensors):
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0],
                                             get_insn(tensor_i)]
            if tensor_i in self._compute_inline_broadcast:
                self._emit_insn_map[tensor_i].append("phony_insn")

        for source, target in self._cache_write_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], get_insn(target)]
            if target in self._compute_inline_broadcast:
                self._emit_insn_map[source].append("phony_insn")

        for tensor_i in self._out_tensors:
            self._emit_insn_map[tensor_i] = [self._emit_insn_axis, "dma_copy"]

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            if len(param) > 2:
                sch[tensor_i].emit_insn(param[0], param[2])
            if param[1] == "unknown_broadcast":
                if self._tiling_strategy == TilingStrategy.NONE_CUT:
                    u_idx = 0
                else:
                    u_idx = self._tiling_case["ub_tiling_axis"]
                src_shapes = tensor_i.op.input_tensors[0].shape[u_idx:]
                tensor_bound = self._tensor_space // \
                    DTYPE_BYTE_MAPPING[tensor_i.dtype]
                src_shape = tvm.expr.Call('handle', 'tvm_tuple',
                                          src_shapes,
                                          tvm.expr.Call.PureIntrinsic,
                                          None, 0)
                sch[tensor_i].emit_insn(param[0], param[1],
                                        attrs=dict(src_shape=src_shape,
                                                   storage_bound=[
                                                       tensor_bound]))
            else:
                if param[1][0:6] == VECTOR:
                    tensor_bound = self._tensor_space // \
                        DTYPE_BYTE_MAPPING[tensor_i.dtype]
                    sch[tensor_i].emit_insn(param[0], param[1],
                                            attrs=dict(storage_bound=[
                                                tensor_bound]))
                else:
                    sch[tensor_i].emit_insn(param[0], param[1])

    def _calc_double_buffer(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            self._db_tensors.clear()

    def _do_double_buffer(self):
        pass

    def _calc_mem_reuse(self):
        for tensor_i, write_buffer in \
                self._middle_out_cache_write_buffer_map.items():
            util.merge_value(self._mem_reuse_map,
                             write_buffer,
                             self._middle_out_cache_read_buffer_map[tensor_i])
        for tensor_i in self._compute_inline_broadcast:
            input_tensor = tensor_i.op.input_tensors[0]
            if util.is_placeholder(input_tensor):
                input_tensor = self._placeholder_tensor_map[input_tensor]
            broadcast_tensor = tensor_i
            if broadcast_tensor in self._cache_write_tensor_map:
                broadcast_tensor = \
                    self._cache_write_tensor_map[broadcast_tensor]
            util.merge_value(self._data_reuse_map,
                             broadcast_tensor,
                             input_tensor)

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
        def _r_coexisting(_tensor):
            if _tensor in dependent_map:
                return len(dependent_map)
            _need_space = []
            for _tensor_i in _tensor.op.input_tensors:
                _need_space.append(_r_coexisting(_tensor_i))
            _current_space = len(dependent_map) + 1
            if (util.need_temp_space(_tensor) and _tensor not in self._compute_inline_broadcast) \
                    or _need_external_space(_tensor):
                _current_space += 1
            if util.is_vsel_insn(_tensor):
                _current_space += VSEL_INPUT_NUMBER - len(_tensor.op.input_tensors)
            if util.is_vcmpsel_insn(_tensor):
                _current_space += VCMPSEL_INPUT_NUMBER - len(_tensor.op.input_tensors)
            _need_space.append(_current_space)
            _refresh_dependent(_tensor)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = self._in_out_map[_tensor].copy()
            return max(_need_space)

        def _refresh_dependent(_tensor):
            for _tensor_i in _tensor.op.input_tensors:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        def _need_external_space(_tensor):
            # pass memory reuse exists error, avoid it in schedule

            exist_absorbable_broadcast = any([x in self._absorbable_broadcast_tensors
                                              for x in _tensor.op.input_tensors])
            if not exist_absorbable_broadcast:
                return False

            op_tag = util.get_dsl_insn(_tensor)
            if op_tag in ("elewise_binary_sub", "elewise_binary_div"):
                return True

            if op_tag in ("elewise_binary_add", "elewise_binary_mul") and _tensor.dtype == "int32":
                return True

        coexisting_quantities = []
        dependent_map = {}
        for tensor_i in self._out.op.input_tensors:
            coexisting_quantities.append(_r_coexisting(tensor_i))
        if not self._out.op.tag == FAKE_NODE_TAG:
            current_space = len(dependent_map) + 1

            if util.is_vsel_insn(self._out):
                current_space += VSEL_INPUT_NUMBER - len(self._out.op.input_tensors)
            if util.is_vcmpsel_insn(self._out):
                current_space += VCMPSEL_INPUT_NUMBER - len(self._out.op.input_tensors)

            if util.need_temp_space(self._out) or _need_external_space(self._out):
                current_space += 1
            coexisting_quantities.append(current_space)

        self._coexisting_quantity = max(coexisting_quantities)
        if self._coexisting_quantity == 1:
            self._ub_size = self._ub_size - BLOCK_SIZE_BYTE

        tensor_space = self._ub_size // self._coexisting_quantity
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())
        for tensor_i in tensors:
            storage_bound = self._tensor_space // \
                DTYPE_BYTE_MAPPING[tensor_i.dtype]
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
        if operation.get_context().get("mode") == SPECIAL_SCALAR:
            if self._broadcast_tensors == self._absorbable_broadcast_tensors:
                operation.get_context().add("support_absorbable_broadcast", True)
            else:
                return False
        return True

    def _add_compile_info(self):
        def _update_compile_info(key, value, keep_max):
            cur_value = get_compile_info().get(key)
            if cur_value is None:
                add_compile_info(key, value)
            else:
                if keep_max:
                    add_compile_info(key, max(value, cur_value))
                else:
                    add_compile_info(key, min(value, cur_value))

        if self._mode == CONST:
            const_shapes = operation.get_context().get("const_shapes")
            const_shape = operation.get_context().\
                get_current_compute().get("const_shape")
            if const_shapes is None:
                const_shapes = [const_shape]
            else:
                const_shapes.append(const_shape)
            operation.get_context().add("const_shapes", const_shapes)
            block_dims = operation.get_context().get("const_block_dims")
            if block_dims is None:
                block_dims = [self._block_dims]
            else:
                block_dims.append(self._block_dims)
            operation.get_context().add("const_block_dims", block_dims)
        else:
            _update_compile_info(CompileInfo.MAX_DTYPE, self._max_dtype_bytes,
                                 True)
            _update_compile_info(CompileInfo.COEXISTING_QUANTITY,
                                 self._coexisting_quantity, True)
            _update_compile_info(CompileInfo.UB_SIZE, self._ub_size, False)
            add_compile_info(CompileInfo.CORE_NUM, util.get_core_num())

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

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self.__dfs_sub_graph(tensor_i, visited_tensors)


def _fake_node(tensors):
    dtype = tensors[0].dtype
    shape = util.shape_to_list(tensors[0].shape)
    for tensor_i in tensors:
        for i, (_a, _b) in enumerate(zip(shape, tensor_i.shape)):
            if util.equals_one(_a):
                shape[i] = _b

    def _fake_compute(*indices):
        res_ = tvm.const(1, dtype)
        for tensor in tensors:
            cur_indices = []
            for idx, dim in enumerate(tensor.shape):
                if util.equals_one(dim):
                    cur_indices.append(0)
                else:
                    cur_indices.append(indices[idx])
            res_ *= tvm.expr.Cast(dtype, tensor(*cur_indices))
        return res_

    with tvm.tag_scope(FAKE_NODE_TAG):
        res = tvm.compute(shape, _fake_compute, name="fake_node")

    return res
