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

from te import tvm

from . import Pattern, INSN_MAPPING, DTYPE_BYTE_MAPPING, FAKE_NODE_TAG, SUPPORT_SCALAR_INSNS, TERNARY_INSNS
from . import CompileInfo
from . import util
from .elewise_tilingcase import TilingStrategy
from te.lang.base.operation import register_schedule
from te.lang.base.operation import get_compile_info
from te.lang.base import operation
from te.lang.base import op_tiling

# block size in D architecture
BLOCK_SIZE_BYTE = 32
MULTI_CORE_THRESHOLD = 1024

N_LAST_BROADCAST_THRESHOLD = 512

#temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192

CONST = "const"
VECTOR = "vector"

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
        self._is_db = self._tiling_case.get("is_need_db", False)
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

            if tensor_i in self._middle_out_tensors:
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
        res = self._out
        shape = util.shape_to_list(res.shape)
        base_info = {"000": [self._ub_size, self._max_dtype_bytes, self._coexisting_quantity, util.get_core_num()]}
        const_compile_info = {
            CompileInfo.ONLY_CONST_TILING: True,
            CompileInfo.BASE_INFO: base_info,
        }
        const_compile_info = {**const_compile_info, **get_compile_info()}
        inputs = []
        for _input in self._input_tensors:
            inputs.append({"shape": util.shape_to_list(_input.shape), "dtype": _input.dtype})
        outputs = [{"shape": shape, "dtype": res.dtype}]
        if len(inputs) == 0:
            inputs = outputs.copy
        op_type = operation.get_context().get_op_type()
        run_info = op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)
        tiling_format = {
            "need_multi_core": "int",
            "block_axis": "int",
            "block_factor": "int",
            "ub_axis": "int",
            "ub_factor": "int"}
        tiling_data = op_tiling.decode(run_info['tiling_data'], tiling_format)
        self._block_dims = run_info["block_dim"]
        self._need_do_block = True if tiling_data["need_multi_core"] > 0 else False
        if self._need_do_block:
            self._block_split_axis = tiling_data["block_axis"]
            self._block_factor = tiling_data["block_factor"]
            self._ub_split_axis = tiling_data["ub_axis"]
            self._ub_factor = tiling_data["ub_factor"]

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
        else:
            self._emit_insn_axis = res.op.axis[0]

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

        for tensor_i in (self._pure_middle_tensors - self._compute_inline_tensors):
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], get_insn(tensor_i)]
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
                tensor_bound = self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype]
                src_shape = tvm.expr.Call('handle', 'tvm_tuple', src_shapes,
                                          tvm.expr.Call.PureIntrinsic, None, 0)
                sch[tensor_i].emit_insn(param[0], param[1],
                                        attrs=dict(src_shape=src_shape, storage_bound=[tensor_bound]))
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
        if self._is_db:
            sch = self._schedule
            
            tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

            for tensor_i in tensors:
                sch[tensor_i].double_buffer()

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
        def _r_coexisting(_tensor):
            if _tensor in dependent_map:
                return len(dependent_map)
            if util.is_vtranspose_broadcast(_tensor):
                self._ub_size = self._ub_size - VTRANSPOSE_TEMP_SPACE
            _need_space = []
            for _tensor_i in _tensor.op.input_tensors:
                _need_space.append(_r_coexisting(_tensor_i))
            # one of the input of the ternary instruction must be reused with the output
            if util.get_dsl_insn(tensor_i) in TERNARY_INSNS:
                _current_space = len(dependent_map)
            else:
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
            support_vector_scalar_insns = ("elewise_binary_add", "elewise_binary_mul")
            if op_tag in set(SUPPORT_SCALAR_INSNS) - set(support_vector_scalar_insns):
                return True

            if util.is_v100() and op_tag in support_vector_scalar_insns and _tensor.dtype == "int32":
                return True

        coexisting_quantities = []
        dependent_map = {}
        for tensor_i in self._out.op.input_tensors:
            coexisting_quantities.append(_r_coexisting(tensor_i))
        if not self._out.op.tag == FAKE_NODE_TAG:
            if util.get_dsl_insn(self._out) in TERNARY_INSNS:
                current_space = len(dependent_map)
            else:
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
            self._ub_size -= BLOCK_SIZE_BYTE
            if self._is_db:
                self._ub_size -= BLOCK_SIZE_BYTE

        tensor_space = self._ub_size // self._coexisting_quantity
        if self._is_db:
            tensor_space = tensor_space // 2
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        for tensor_i in tensors:
            storage_bound = self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype]
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
