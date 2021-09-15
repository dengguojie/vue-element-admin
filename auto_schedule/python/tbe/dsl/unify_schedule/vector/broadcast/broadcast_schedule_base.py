# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
base broadcast schedule
"""
import copy
from typing import Optional
from functools import reduce

from tbe import tvm
from tbe.common.utils import op_tiling
from tbe.dsl.base import operation
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.operation import get_compile_info
from tbe.common.platform import SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec

from ... import util
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import INSN_MAPPING
from ...constants import TERNARY_INSNS
from .broadcast_tilingcase import TilingStrategy
from .broadcast_tilingcase import BroadcastTilingCase

# block size in D architecture
BLOCK_SIZE_BYTE = 32
ONE_DIM_ALIGN = 128

# temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192
MAX_EXTEND_NODE_NUM = 2

CONST = "const"
BROADCAST = "broadcast"
VECTOR_BROADCAST = "vector_broadcast"
UNKNOWN_BROADCAST = "unknown_broadcast"
DMA_COPY = "dma_copy"
PHONY_INSN = "phony_insn"


# 'pylint: disable=R0902, R0903
class BaseBroadcastSchedule:
    """
    BaseBroadcastSchedule
    """

    class ComputeAt:
        """
        BaseBroadcast ComputeAt
        """
        def __init__(self):
            self._compute_at_axis = None

        @property
        def compute_at_axis(self):
            """
            return compute_at_axis
            """
            return self._compute_at_axis

        @compute_at_axis.setter
        def compute_at_axis(self, axis):
            self._compute_at_axis = axis

    class EmitInsn:
        """
        BaseBroadcast EmitInsn Bean
        """
        def __init__(self):
            self._emit_insn_axis = None

        @property
        def emit_insn_axis(self):
            """
            return emit_insn_axis
            """
            return self._emit_insn_axis

        @emit_insn_axis.setter
        def emit_insn_axis(self, axis):
            self._emit_insn_axis = axis

    def __init__(self, outs, tiling_case):
        self._out: Optional[tvm.tensor.Tensor] = None
        self._outs = outs
        self._schedule = None
        self._tiling_case: Optional[BroadcastTilingCase] = tiling_case
        self._tiling_strategy = self._tiling_case.tiling_strategy
        self._enable_db = self._tiling_case.enable_db
        self._is_one_dim = self._tiling_case.is_one_dim
        self._mode = operation.get_context().get_current_compute().get("_mode")

        self._scope = "local.UB"

        self._input_tensors = set()
        self._middle_tensors = set()
        self._pure_middle_tensors = set()
        self._middle_out_tensors = set()
        self._out_tensors = set()

        self._broadcast_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._broadcast_axis_num = {}

        self._broadcast_store_predicate = set()
        self._store_predicate_common_tensors = set()
        self._all_pre_node_broadcast = set()

        self._dtypes = set()
        self._outs_dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._tensor_space = None
        self._ub_size = util.get_ub_size()
        self._correct_factor = 2 if self._enable_db else 1
        self._tmp_ub_size = 0
        self._min_storage_bound = -1

        # input -> outputs mapping relations
        self._in_out_map = {}

        self._cache_read_tensors = set()
        self._cache_read_buffer_tensor_map = {}
        self._input_tensor_map = {}
        self._middle_out_cache_read_buffer_map = {}

        self._cache_write_tensors = set()
        self._cache_write_buffer_tensor_map = {}
        self._out_tensor_map = {}
        self._middle_out_cache_write_buffer_map = {}

        self._compute_inline_tensors = set()

        self._compute_at_map = {}

        self._need_do_block = False
        self._block_dims = 1
        self._block_split_axis = -1 if self._tiling_case.block_split_axis is None \
            else self._tiling_case.block_split_axis
        self._block_factor = 1
        self._ub_split_axis = 0 if self._tiling_case.ub_split_axis is None else self._tiling_case.ub_split_axis
        self._ub_factor = 1

        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._block_bind_axis = None
        self._compute_at_axis = self.ComputeAt()
        self._compute_at_axis_idx = None
        self._emit_insn_axis = self.EmitInsn()

        self._inner_shape = []

        self._constraints = set()

        self._mem_reuse_map = {}
        self._data_reuse_map = {}

        self._emit_insn_map = {}

        self._compute_align_map = {}
        self._remove_pad_map = {}

    def do_schedule(self):
        """
        :return:
        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case.tiling_key

        self._calc_remove_pad()

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        if self._tiling_strategy == TilingStrategy.CONST:
            self._get_const_storage_bound()
            self._calc_tiling()
            self._calc_store_predicate()
        else:
            self._calc_store_predicate()
            self._calc_storage_bound()
            self._calc_tiling()
        self._calc_compute_inline()
        self._calc_multi_core()
        self._calc_ub_align()
        self._calc_compute_at()
        self._calc_double_buffer()
        self._calc_mem_reuse()
        self._calc_constraints()
        self._calc_emit_insn()

        self._do_tiling()
        self._do_storage_bound()
        self._do_compute_inline()
        self._do_multi_core()
        self._do_ub_align()
        self._do_remove_pad()
        self._do_compute_at()
        self._do_store_predicate()
        self._do_double_buffer()
        self._do_mem_reuse()
        self._do_constraints()
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
            if util.is_broadcast(out):
                self._broadcast_tensors.add(out)
            self.__dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)
            self._outs_dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors
        self._middle_out_tensors = self._middle_tensors.intersection(
            self._out_tensors)

        pure_out_tensors = list(self._out_tensors - self._middle_out_tensors)
        if len(pure_out_tensors) > 1:
            self._out = _fake_node(pure_out_tensors)
            self.__dfs_sub_graph(self._out, visited_tensors)
        else:
            self._out = pure_out_tensors[0]

        for tensor_i in self._broadcast_tensors:
            if match_scalar_scene(tensor_i):
                self._absorbable_broadcast_tensors.add(tensor_i)

        ub_idx = self._ub_split_axis
        for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
            if util.get_dsl_insn(tensor_i) != BROADCAST:
                src_shapes = tensor_i.op.input_tensors[0].shape[ub_idx:]
                dst_shapes = tensor_i.shape[ub_idx:]
                self._broadcast_axis_num[tensor_i] = _no_broadcast(src_shapes, dst_shapes)

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)
        self._cache_read_tensors.update(self._middle_out_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._input_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_read_buffer_map[tensor_i] = buffer_tensor

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._out_tensors)

    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[buffer_tensor] = tensor_i
            self._out_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_write_buffer_map[tensor_i] = \
                    buffer_tensor

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._pure_middle_tensors:
            sch[tensor_i].set_scope(self._scope)

    def _get_const_storage_bound(self):
        res = self._out
        output_shape = util.shape_to_list(res.shape)
        const_tmp_ub_size = 0
        const_coexisting_quantity = 1
        for i in range(len(output_shape) - 1, -1, -1):
            self._ub_split_axis = i
            self._tmp_ub_size = 0
            self._coexisting_quantity = 1
            self._need_do_block = True
            self._calc_store_predicate()
            self._need_do_block = False
            self._calc_storage_bound()
            const_tmp_ub_size = max(const_tmp_ub_size, self._tmp_ub_size)
            const_coexisting_quantity = max(const_coexisting_quantity, self._coexisting_quantity)
            self._broadcast_store_predicate.clear()
            self._all_pre_node_broadcast.clear()
            self._store_predicate_common_tensors.clear()

        self._tmp_ub_size = const_tmp_ub_size
        self._coexisting_quantity = const_coexisting_quantity
        self._ub_split_axis = 0

    def _calc_store_predicate(self):
        def _dfs_cur_tensor(tensor_i):
            for _tensor in tensor_i.op.input_tensors:
                all_pre_node.add(_tensor)
                _dfs_cur_tensor(_tensor)

        def _is_only_calc_one(tensor_i):
            if len(tensor_i.op.input_tensors) != 1:
                return False
            if self._tiling_strategy == TilingStrategy.ONE_CUT or \
                    (self._tiling_strategy == TilingStrategy.CONST and self._need_do_block):
                u_i = self._ub_split_axis
                src_shape = tensor_i.op.input_tensors[0].shape
                dst_shape = tensor_i.shape
                return not expr_equal(src_shape[u_i], dst_shape[u_i])
            return False

        def _has_ternary_insns():
            for tensor_i in self._out_tensors | self._pure_middle_tensors:
                insn = util.get_dsl_insn(tensor_i)
                if insn in TERNARY_INSNS:
                    return True
            return False

        if _has_ternary_insns():
            return

        for tensor_i in self._broadcast_tensors:
            if not _is_only_calc_one(tensor_i):
                continue
            cur_tensor = tensor_i
            pre_tensor = tensor_i
            all_pre_node = set()
            if tensor_i in self._absorbable_broadcast_tensors:
                cur_tensor = tensor_i.op.input_tensors[0]
                pre_tensor = cur_tensor
            elif tensor_i in self._remove_pad_map:
                cur_tensor = self._remove_pad_map[tensor_i]
                all_pre_node.add(tensor_i)
            self._broadcast_store_predicate.add(cur_tensor)
            _dfs_cur_tensor(pre_tensor)
            self._all_pre_node_broadcast.update(all_pre_node)

        disable_store_perdicate = False
        for tensor_i in self._all_pre_node_broadcast:
            common_tensor = self._in_out_map[tensor_i] - \
                            (self._all_pre_node_broadcast | self._broadcast_store_predicate)
            if len(common_tensor) > 0:
                # common in multi output
                if tensor_i in self._out_tensors:
                    disable_store_perdicate = True
                    break
                self._store_predicate_common_tensors.add(tensor_i)
        extend_node_num = len(self._broadcast_store_predicate) + len(self._store_predicate_common_tensors)
        if disable_store_perdicate or extend_node_num > MAX_EXTEND_NODE_NUM:
            self._broadcast_store_predicate.clear()
            self._all_pre_node_broadcast.clear()
            self._store_predicate_common_tensors.clear()

    def _calc_storage_bound(self):
        pass

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
            self._block_tiling_vars[_i] = operation.var_inner("_block_factor_" + str(_i), bound)
            self._ub_tiling_vars[_i] = operation.var_inner("_ub_factor_" + str(_i), bound)

    def _calc_tiling_none_cut(self):
        pass

    def _calc_tiling_one_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis
        u_i = self._ub_split_axis
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_bound = self._tiling_case.ub_factor_bound
        if u_bound is None:
            u_bound = (1, util.get_bound(shape[u_i])[1])
        self._block_tiling_vars[b_i] = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = operation.var_inner("_ub_factor_" + str(u_i), u_bound)
        self._block_factor = self._block_tiling_vars[b_i]
        self._ub_factor = self._ub_tiling_vars[u_i]

    def _calc_tiling_static(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis
        u_i = self._ub_split_axis
        b_bound = (1, util.get_bound(shape[b_i])[1])
        self._block_tiling_vars[b_i] = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = self._tiling_case.ub_factor_bound
        self._block_factor = self._block_tiling_vars[b_i]
        self._ub_factor = self._ub_tiling_vars[u_i]

    def _calc_tiling_const(self):
        def _get_const_tiling():
            input_shapes = []
            inputs = []
            max_dim_length = len(output_shape)
            for _input in self._input_tensors:
                input_shape = util.shape_to_list(_input.shape)
                input_shapes.append([1] * (max_dim_length - len(input_shape)) + input_shape)
                inputs.append({"shape": input_shape, "dtype": _input.dtype})
            outputs = [{"shape": output_shape, "dtype": res.dtype}]
            if len(inputs) == 0:
                inputs = copy.deepcopy(outputs)
                max_dim_length = 0

            input_shapes = list(map(list, zip(*input_shapes)))
            broadcast_axis = [False] * max_dim_length
            for i in range(max_dim_length - 1, -1, -1):
                if any(input_shapes[i][0] != s for s in input_shapes[i]):
                    broadcast_axis[i] = True

            max_available_ub = ((((self._ub_size - self._tmp_ub_size) // self._coexisting_quantity) //
                                 BLOCK_SIZE_BYTE) * BLOCK_SIZE_BYTE) // self._max_dtype_bytes
            max_available_ub_db = ((((self._ub_size - 2 * self._tmp_ub_size) // 2 // self._coexisting_quantity) //
                                    BLOCK_SIZE_BYTE) * BLOCK_SIZE_BYTE) // self._max_dtype_bytes
            base_info = {"000": [util.get_core_num(), self._max_dtype_bytes, max_available_ub, max_available_ub_db]}

            const_compile_info = {
                CompileInfo.FLAG_INFO: [True],
                CompileInfo.BASE_INFO: base_info,
                CompileInfo.SOC_VERSION: get_soc_spec(SOC_VERSION),
                CompileInfo.BROADCAST_AXIS: broadcast_axis,
                CompileInfo.OUTS_UINT1: "uint1" in self._outs_dtypes,
            }
            const_compile_info.update(get_compile_info())

            op_type = "AutoTiling"
            return op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)

        res = self._out
        output_shape = util.shape_to_list(res.shape)
        if output_shape == [0]:
            self._block_dims = 1
            self._need_do_block = False
            return

        run_info = _get_const_tiling()

        tiling_format = {
            "need_multi_core": "int",
            "block_axis": "int",
            "block_factor": "int",
            "ub_axis": "int",
            "ub_factor": "int",
            "is_need_db": "int"}
        tiling_data = op_tiling.decode(run_info['tiling_data'], tiling_format)
        self._block_dims = run_info["block_dim"]
        self._need_do_block = True if tiling_data["need_multi_core"] > 0 else False
        if self._need_do_block:
            self._block_split_axis = tiling_data["block_axis"]
            self._block_factor = tiling_data["block_factor"]
            self._ub_split_axis = tiling_data["ub_axis"]
            self._ub_factor = tiling_data["ub_factor"]

        self._enable_db = True if tiling_data.get("is_need_db", 0) == 1 else False

    def _calc_compute_inline(self):
        def no_broadcast(_src_shapes, _dst_shapes):
            _src_shapes = util.shape_to_list(_src_shapes)
            _dst_shapes = util.shape_to_list(_dst_shapes)
            for x, y in zip(_src_shapes, _dst_shapes):
                if not expr_equal(x, y):
                    return False
            return True

        def update_store_predicate(_tensor):
            if _tensor in self._broadcast_store_predicate:
                self._broadcast_store_predicate.remove(_tensor)
                self._all_pre_node_broadcast.remove(_tensor.op.input_tensors[0])
                self._broadcast_store_predicate.add(_tensor.op.input_tensors[0])

        self._compute_inline_tensors = self._absorbable_broadcast_tensors.copy()
        if self._tiling_strategy == TilingStrategy.CONST:
            ub_idx = self._ub_split_axis
            for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
                if util.get_dsl_insn(tensor_i) != BROADCAST:
                    src_shapes = tensor_i.op.input_tensors[0].shape[ub_idx:]
                    dst_shapes = tensor_i.shape[ub_idx:]
                    if no_broadcast(src_shapes, dst_shapes):
                        update_store_predicate(tensor_i)
                        self._compute_inline_tensors.add(self._get_ub_tensor(tensor_i))
        else:
            for tensor_i in self._broadcast_axis_num:
                if self._broadcast_axis_num[tensor_i] == 0:
                    self._compute_inline_tensors.add(self._get_ub_tensor(tensor_i))

        self.__calc_tensor_space()

        for tensor_i in self._broadcast_tensors - self._compute_inline_tensors:
            need_update_var_range = False
            if util.get_dsl_insn(tensor_i) != BROADCAST:
                ub_under_shapes = util.shape_to_list(tensor_i.op.input_tensors[0].shape[self._ub_split_axis + 1:])
                ub_under_size = reduce(lambda x, y: x * y, ub_under_shapes or [1])
                if (self._min_storage_bound // ub_under_size) == 1 and isinstance(self._ub_factor, tvm.expr.Var):
                    self._compute_inline_tensors.add(self._get_ub_tensor(tensor_i))
                    update_store_predicate(tensor_i)
                    need_update_var_range = True
            if need_update_var_range:
                self._constraints.add(tvm.expr.EQ(self._ub_factor, 1))
                out_ub_shapes = util.shape_to_list(self._out.shape[self._ub_split_axis + 1:])
                for in_var, out_var in zip(ub_under_shapes, out_ub_shapes):
                    if isinstance(out_var, tvm.expr.Var):
                        self._constraints.add(tvm.expr.EQ(out_var, in_var))

    def _calc_multi_core(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            self._block_bind_axis = None

    def _calc_remove_pad(self):
        pass

    def _calc_ub_align(self):
        pass

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
            if tensor_i not in self._compute_inline_tensors:
                self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

    def _calc_double_buffer(self):
        pass

    def _calc_mem_reuse(self):
        ternary_reuse_map = {
            "elewise_binary_scalar_axpy": 1,
            "elewise_multiple_mla": 2,
            "elewise_multiple_madd": 1,
            "elewise_multiple_maddrelu": 1,
        }

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
                src_tensor = self._get_ub_tensor(src_tensor)
                dst_tensor = self._get_ub_tensor(tensor_i)
                util.merge_value(self._mem_reuse_map,
                                 src_tensor,
                                 dst_tensor)
        for tensor_i, write_buffer in self._middle_out_cache_write_buffer_map.items():
            util.merge_value(self._mem_reuse_map,
                             self._middle_out_cache_read_buffer_map[tensor_i],
                             write_buffer)

    def _calc_constraints(self):
        def add_condition(condition):
            if isinstance(condition, tvm.expr.Expr):
                self._constraints.add(condition)

        for tensor_i in self._broadcast_tensors:
            if util.is_unknown_broadcast(tensor_i):
                src_shapes = util.shape_to_list(tensor_i.op.input_tensors[0].shape)
                dst_shapes = util.shape_to_list(tensor_i.shape)
                for src_shape, dst_shape in zip(src_shapes, dst_shapes):
                    if not expr_equal(src_shape, dst_shape):
                        add_condition(src_shape <= dst_shape)

        shapes = util.shape_to_list(self._out.shape)
        ub_shapes = shapes[self._ub_split_axis + 1:]
        for shape in ub_shapes:
            add_condition(shape <= self._min_storage_bound)
        if self._tiling_strategy != TilingStrategy.NONE_CUT:
            block_shapes = shapes[:self._block_split_axis]
            core_num = util.get_core_num()
            for shape in block_shapes:
                add_condition(shape <= core_num)
            ub_shapes.insert(0, self._ub_factor)
            shape_size = reduce(lambda x, y: x * y, ub_shapes)
            add_condition(shape_size <= self._min_storage_bound)
            block_size = reduce(lambda x, y: x * y, block_shapes or [1])
            add_condition(block_size <= core_num)
            add_condition(self._ub_factor <= self._min_storage_bound)

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
                self._emit_insn_map[source] = [source.op.axis[0], PHONY_INSN]
            else:
                self._emit_insn_map[source] = [source.op.axis[0], DMA_COPY]

        for tensor_i in self._pure_middle_tensors - self._compute_inline_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], get_insn(tensor_i)]

        for source, target in self._cache_write_buffer_tensor_map.items() - self._compute_inline_tensors:
            self._emit_insn_map[source] = [source.op.axis[0], get_insn(target)]

        if len(self._out_tensors) > 1:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], DMA_COPY]
            if len(self._out_tensors) - len(self._middle_out_tensors) > 1:
                self._emit_insn_map[self._out] = [self._emit_insn_axis, PHONY_INSN]
            else:
                self._emit_insn_map[self._out] = [self._emit_insn_axis, DMA_COPY]
        else:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [self._emit_insn_axis, DMA_COPY]

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
        ir_axes = block_axes + ub_axes + inner_axes
        ordered_axes = [x[0] for x in ir_axes]
        sch[res].reorder(*ordered_axes)
        self._block_bind_axis = sch[res].fuse(*[x[0] for x in block_axes])
        self._compute_at_axis.compute_at_axis = ub_axes[-1][0]
        self._compute_at_axis_idx = ub_axes[-1][1]
        self._emit_insn_axis.emit_insn_axis = inner_axes[0][0]

    def _do_tiling_none_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        for _i, _x in enumerate(res.op.axis):
            self._inner_shape.append([shape[_i], _i])
        self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_tiling_one_cut(self):
        sch = self._schedule
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_idx = self._block_split_axis
        u_idx = self._ub_split_axis
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

        self._block_bind_axis = sch[res].fuse(*[x[0] for x in block_axes])
        self._compute_at_axis.compute_at_axis = ub_axes[-1][0]
        self._compute_at_axis_idx = ub_axes[-1][1]
        self._emit_insn_axis.emit_insn_axis = inner_axes[0][0]

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
            if self._block_split_axis >= 0 and self._block_split_axis == self._ub_split_axis:
                u_o, u_i = sch[res].split(b_i, factor=self._ub_factor)
            else:
                u_o, u_i = sch[res].split(res.op.axis[self._ub_split_axis],
                                          factor=self._ub_factor)
            self._compute_at_axis.compute_at_axis = u_o
            self._emit_insn_axis.emit_insn_axis = u_i
        else:
            self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_storage_bound(storage_bound)

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _do_ub_align(self):
        pass

    def _do_remove_pad(self):
        pass

    def _do_compute_at(self):
        sch = self._schedule
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1].compute_at_axis)

    def _do_store_predicate(self):
        sch = self._schedule
        cond_limit = 1
        if self._enable_db:
            # pass double buffer error, avoid it in schedule
            # double buffer IR:
            # ' for i:a
            # '   if i * 2 < cond_limit:
            # '     do()
            # '   if i * 2 + 1 < cond_limit:
            # '     do()
            cond_limit = 2
        u_idx = self._ub_split_axis
        for tensor_i in self._broadcast_store_predicate:
            input_tensors = tensor_i.op.input_tensors
            is_pad_broadcast = len(input_tensors) > 0 and input_tensors[0] in self._remove_pad_map \
                               and util.is_broadcast(input_tensors[0])
            if util.is_broadcast(tensor_i):
                ub_split_src = tensor_i.op.input_tensors[0].shape[u_idx]
            elif is_pad_broadcast:
                ub_split_src = tensor_i.op.input_tensors[0].op.input_tensors[0].shape[u_idx]
            else:
                ub_split_src = tensor_i.shape[u_idx]
            cond = tvm.any(self._compute_at_axis.compute_at_axis < cond_limit, ub_split_src != 1)
            sch[self._get_ub_tensor(tensor_i)].set_store_predicate(cond)
            sch[self._get_ub_tensor(tensor_i)].mem_unique()
        for tensor_i in self._all_pre_node_broadcast:
            if util.is_broadcast(tensor_i) and not util.is_scalar_broadcast(tensor_i):
                ub_split_src = tensor_i.op.input_tensors[0].shape[u_idx]
            else:
                ub_split_src = tensor_i.shape[u_idx]
            cond = tvm.any(self._compute_at_axis.compute_at_axis < cond_limit, ub_split_src != 1)
            if tensor_i in self._input_tensor_map:
                sch[self._input_tensor_map[tensor_i]].set_store_predicate(cond)
            else:
                sch[tensor_i].set_store_predicate(cond)
        for tensor_i in self._store_predicate_common_tensors:
            if tensor_i in self._input_tensor_map:
                sch[self._input_tensor_map[tensor_i]].mem_unique()
            else:
                sch[tensor_i].mem_unique()

    def _do_double_buffer(self):
        if self._enable_db:
            sch = self._schedule

            tensors = self._pure_middle_tensors \
                .union(self._cache_read_buffer_tensor_map.keys()) \
                .union(self._cache_write_buffer_tensor_map.keys())

            for tensor_i in tensors:
                sch[tensor_i].double_buffer()

    def _do_mem_reuse(self):
        sch = self._schedule
        for _a, _b in self._mem_reuse_map.items():
            for b_i in _b:
                sch[_a].reused_by(b_i)

    def _do_constraints(self):
        sch = self._schedule
        for cond in self._constraints:
            sch.set_constraint(cond)

    def _do_emit_insn(self):
        def get_ori_tensor(tensor_i):
            if tensor_i in self._cache_read_buffer_tensor_map:
                return self._cache_read_buffer_tensor_map[tensor_i]
            if tensor_i in self._cache_write_buffer_tensor_map:
                return self._cache_write_buffer_tensor_map[tensor_i]
            return tensor_i

        def enable_dynamic_optimize(_tensor_i):
            def is_original():
                tiling_key = self._schedule.tiling_key
                base = 10000000
                return (tiling_key // base % 10) == 0

            def is_last_align():
                is_align = False
                element_in_block = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[_tensor_i.dtype]
                for _src_shape in reversed(_src_shapes):
                    if isinstance(_src_shape, tvm.expr.Var) or expr_equal(_src_shape, 1):
                        break
                    if _src_shape % element_in_block == 0:
                        is_align = True
                        break
                return is_align

            def is_more_var_shape():
                var_count = 0
                tvm_expr = (tvm.expr.Expr, tvm.expr.Var)
                for _src_shape in _src_shapes:
                    if isinstance(_src_shape, tvm_expr):
                        var_count += 1
                for _dst_shape in _dst_shapes:
                    if isinstance(_dst_shape, tvm_expr):
                        var_count += 1
                if not isinstance(_dst_shapes[0], tvm_expr) and isinstance(self._ub_factor, tvm_expr):
                    var_count += 1
                var_num_threshold = 0.6
                return var_count > ((len(_src_shapes) + len(_dst_shapes)) * var_num_threshold)

            _u_idx = self._ub_split_axis
            _src_shapes = util.shape_to_list(_tensor_i.op.input_tensors[0].shape[_u_idx:])
            _dst_shapes = util.shape_to_list(_tensor_i.shape[_u_idx:])
            return not util.is_v220() and is_original() and not is_last_align() and is_more_var_shape()

        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            emit_insn_axis = param[0]
            if isinstance(emit_insn_axis, self.EmitInsn):
                emit_insn_axis = emit_insn_axis.emit_insn_axis
            if len(param) > 2:
                sch[tensor_i].emit_insn(emit_insn_axis, param[2])
            compile_broadcast_no_inline = (param[1] == VECTOR_BROADCAST and
                                           self._broadcast_axis_num.get(get_ori_tensor(tensor_i), 0) > 1)
            attrs = {}
            tensor_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            if param[1] == UNKNOWN_BROADCAST:
                u_idx = self._ub_split_axis
                src_shapes = tensor_i.op.input_tensors[0].shape[u_idx:]
                is_all_const = all(isinstance(s, int) for s in util.shape_to_list(src_shapes))
                if is_all_const:
                    attrs = {"storage_bound":[tensor_bound]}
                    param[1] = VECTOR_BROADCAST
                else:
                    attrs = {"storage_bound":[tensor_bound],
                                 "dynamic_fuse":False,
                                 "dynamic_split":False}
                    if enable_dynamic_optimize(tensor_i):
                        attrs["dynamic_fuse"] = True
                        attrs["dynamic_split"] = True
            elif compile_broadcast_no_inline:
                attrs = {"storage_bound":[tensor_bound]}
            elif tensor_i in self._out_tensors and self._is_one_dim:
                attrs = {"no_overlap":0}
            if param[1] in (VECTOR_BROADCAST, UNKNOWN_BROADCAST) and tensor_i in self._compute_align_map:
                attrs["last_src_valid_element"] = self._compute_align_map[tensor_i][1]
            sch[tensor_i].emit_insn(emit_insn_axis, param[1], attrs)

    def _add_compile_info(self):
        cpt_compute = operation.get_context().get_current_compute()
        cpt_schedule = cpt_compute.get_current_schedule()
        if self._mode == CONST:
            # const shape: one compute, one schedule
            cpt_compute.add("_const_block_dim", self._block_dims)
        else:
            cpt_schedule.add(CompileInfo.MAX_DTYPE, self._max_dtype_bytes)
            cpt_schedule.add(CompileInfo.COEXISTING_QUANTITY, self._coexisting_quantity)
            cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
            cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())
            cpt_schedule.add("_tiling_key", self._schedule.tiling_key)

        outs_contains_uint1 = "uint1" in self._outs_dtypes or \
                              operation.get_compile_info().get(CompileInfo.OUTS_UINT1, False)
        operation.add_compile_info_inner(CompileInfo.OUTS_UINT1, outs_contains_uint1)

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

    def __calc_tensor_space(self):
        # minus tmp size
        self._correct_factor = 2 if self._enable_db else 1
        self._ub_size -= self._correct_factor * self._tmp_ub_size
        tensor_space = self._ub_size // self._coexisting_quantity
        if self._enable_db:
            tensor_space = self._ub_size // 2 // self._coexisting_quantity
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        # adjust storage bound by tiling handle one dim (128 align)
        if self._is_one_dim and self._tensor_space > ONE_DIM_ALIGN:
            self._tensor_space = self._tensor_space // ONE_DIM_ALIGN * ONE_DIM_ALIGN

        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        min_storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[self._out.dtype])
        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            if storage_bound < min_storage_bound:
                min_storage_bound = storage_bound
        self._min_storage_bound = min_storage_bound

    def _get_ub_tensor(self, tensor_i):
        if tensor_i in self._input_tensor_map:
            return self._input_tensor_map[tensor_i]
        if tensor_i in self._out_tensor_map:
            return self._out_tensor_map[tensor_i]
        return tensor_i

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
    dim_length = max(len(t.shape) for t in tensors)
    shape = [1] * dim_length
    for tensor_i in tensors:
        if DTYPE_BYTE_MAPPING[tensor_i.dtype] > DTYPE_BYTE_MAPPING[dtype]:
            dtype = tensor_i.dtype
        shape_i = util.shape_to_list(tensor_i.shape)
        diff = dim_length - len(shape_i)
        shape_i = [1] * diff + shape_i
        for j in range(diff, dim_length):
            if util.equals_one(shape[j]):
                shape[j] = shape_i[j]
            elif not expr_equal(shape[j], shape_i[j]) and not util.equals_one(shape_i[j]):
                shape[j] = tvm.max(shape_i[j], shape[j])

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
