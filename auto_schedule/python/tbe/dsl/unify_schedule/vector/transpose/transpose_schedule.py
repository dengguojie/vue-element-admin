# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
transpose schedule
"""
from abc import ABC
from typing import Optional
from typing import List

from tbe import tvm
from tbe.dsl.base import operation
from tbe.common.utils import op_tiling
from tbe.dsl.base.operation import get_compile_info
from tbe.tvm.schedule import IterVar

from tbe.dsl.unify_schedule import util
from tbe.dsl.unify_schedule.schedule import Schedule
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.constants import CompileInfo
from tbe.dsl.unify_schedule.constants import TransposePattern
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from .transpose_tilingcase import TilingStrategy
from .transpose_tilingcase import TransposeTilingCase

DEFAULT = "default"
DMA_COPY = "dma_copy"

BLOCK_SIZE_BYTE = 32
ALIGN_THRESHOLD = 128

TYPE_IN_BLOCK = {
    1: 32,
    2: 16,
    4: 8,
    8: 4,
}


# 'pylint: disable=R0902, R0903
class TransposeSchedule(Schedule, ABC):
    """
    TransposeSchedule
    """

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TRANSPOSE]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [TransposePattern.T_0]

    class ComputeAt:
        """
        TransposeSchedule ComputeAt
        """

        def __init__(self):
            self._compute_at_axis = None

        @property
        def compute_at_axis(self):
            """
            :return: compute_at_axis
            """
            return self._compute_at_axis

        @compute_at_axis.setter
        def compute_at_axis(self, axis):
            """
            set compute_at_axis
            :param axis:
            :return:
            """
            self._compute_at_axis = axis

    class EmitInsn:
        """
        TransposeSchedule EmitInsn Bean
        """

        def __init__(self):
            self._emit_insn_axis = None

        @property
        def emit_insn_axis(self):
            """
            :return: emit_insn_axis
            """
            return self._emit_insn_axis

        @emit_insn_axis.setter
        def emit_insn_axis(self, axis):
            """
            :param axis:
            :return: emit_insn_axis
            """
            self._emit_insn_axis = axis

    class Util:
        @staticmethod
        def is_store_align(strategy: TilingStrategy):
            return strategy in [TilingStrategy.STORE_ALIGN_SINGLE, TilingStrategy.STORE_ALIGN_NONE_CUT]

        @staticmethod
        def is_store_align_single(strategy: TilingStrategy):
            return strategy == TilingStrategy.STORE_ALIGN_SINGLE

        @staticmethod
        def is_all_read_align(strategy: TilingStrategy):
            return strategy in [TilingStrategy.READ_ALIGN, TilingStrategy.READ_ALIGN_NONE_CUT]

        @staticmethod
        def is_const(strategy: TilingStrategy):
            return strategy == TilingStrategy.CONST

        @staticmethod
        def is_general(strategy: TilingStrategy):
            return strategy == TilingStrategy.GENERAL

        @staticmethod
        def is_read_align(strategy: TilingStrategy):
            return strategy == TilingStrategy.READ_ALIGN

        @staticmethod
        def is_all_none_cut(strategy: TilingStrategy):
            return strategy in [TilingStrategy.NONE_CUT,
                                TilingStrategy.STORE_ALIGN_NONE_CUT, TilingStrategy.READ_ALIGN_NONE_CUT]

        @staticmethod
        def is_none_cut(strategy: TilingStrategy):
            return strategy == TilingStrategy.NONE_CUT

        @staticmethod
        def is_single_cut(strategy: TilingStrategy):
            return strategy in [TilingStrategy.NONE_CUT, TilingStrategy.STORE_ALIGN_SINGLE,
                                TilingStrategy.STORE_ALIGN_NONE_CUT, TilingStrategy.READ_ALIGN_NONE_CUT]

    def __init__(self, outs: List[tvm.tensor.Tensor], tiling_case):
        self._out: tvm.tensor.Tensor = outs[0]
        self._schedule: Optional[tvm.schedule] = None
        self._tiling_case: Optional[TransposeTilingCase] = tiling_case
        self._tiling_strategy: TilingStrategy = self._tiling_case.tiling_strategy
        self._enable_db: bool = self._tiling_case.enable_db
        self._max_dtype_bytes: Optional[float] = None
        self._block_split_axis_index: int = self._tiling_case.block_split_axis
        self._low_ub_split_axis_index: int = self._tiling_case.low_ub_split_axis
        self._high_ub_split_axis_index: int = self._tiling_case.high_ub_split_axis

        self._base_order = []
        self._permute = []
        self._ori_permute = []

        self._none_cut = self.Util.is_all_none_cut(self._tiling_strategy)
        self._const_none_cut = False

        self._scope = "local.UB"

        self._in_out_map = {}

        self._input_tensors = set()
        self._out_tensors = set()
        self._middle_tensors = set()

        self._cache_read_tensors = set()
        self._cache_read_buffer_tensors_map = {}

        self._cache_write_tensors = set()
        self._cache_write_buffer_tensors_map = {}

        self._reorder_out_index = []
        self._reorder_in_index = []
        self._last_transpose = False
        self._is_const_align = False
        self._is_const_store_align = False
        self._block_dims = 0

        self._coexisting_quantity = 1

        self._need_do_block = False
        self._block_factor = 1
        self._low_ub_factor = 1
        self._high_ub_factor = 1
        self._block_fuse_axis = []
        self._ub_split_out_axis = []
        self._ub_split_in_axis = []
        self._block_split_out_axis: Optional[IterVar] = None
        self._block_split_in_axis: Optional[IterVar] = None
        self._ub_split_same_axis = False

        self._input_align_axis = -1
        self._output_align_axis = -1

        self._compute_inline_tensors = set()

        self._compute_at_map = {}
        self._compute_at_axis = self.ComputeAt()

        self._emit_insn_axis = self.EmitInsn()
        self._emit_insn_map = {}

        self._ub_size = util.get_ub_size()
        self._tensor_space: Optional[int] = None

    def do_schedule(self):
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case.tiling_key

        if self.Util.is_general(self._tiling_strategy) or self.Util.is_read_align(self._tiling_strategy):
            if not self._check_tiling_case():
                return None

        self._calc_tiling()

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_real_reorder()
        self._calc_cache_write()
        self._do_cache_write_reorder()

        self._set_scope()

        self._calc_storage_bound()
        self._calc_reorder()
        self._calc_compute_inline()
        self._calc_multi_core()
        self._calc_compute_at()
        self._calc_storage_align()
        self._calc_double_buffer()
        self._calc_constraints()
        self._calc_emit_insn()

        self._do_tiling()
        self._do_storage_align()
        self._do_reorder()
        self._do_storage_bound()
        self._do_multi_core()
        self._do_compute_at()
        self._do_compute_inline()
        self._do_double_buffer()
        self._do_constraints()
        self._do_emit_insn()

        self._add_compile_info()

        return self._schedule if self._check_tiling_factor() else None

    def _construct_compute_graph(self):
        def _dfs_sub_graph(out):
            for tensor_i in out.op.input_tensors:
                util.merge_value(self._in_out_map, tensor_i, out)
                dtypes.add(out.dtype)
                if util.is_placeholder(tensor_i):
                    self._input_tensors.add(tensor_i)
                else:
                    self._middle_tensors.add(tensor_i)
                _dfs_sub_graph(tensor_i)

        dtypes = set()
        dtypes.add(self._out.dtype)
        _dfs_sub_graph(self._out)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in dtypes]
        self._max_dtype_bytes = max(byte_len)
        self._out_tensors.add(self._out)

        transpose_index = [int(i) for i in self._out.op.attrs["permute"]]
        self._ori_permute = transpose_index
        self._permute = sorted(range(len(transpose_index)), key=transpose_index.__getitem__)
        self._base_order = [i for i in range(len(self._permute))]

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensors_map[buffer_tensor] = tensor_i

    def _calc_real_reorder(self):
        def calc_last_cross_index():
            cur_input_reorder_index = [i for i in range(dim_len)]
            cur_consequent_output_by_transpose = []
            for i in self._permute:
                if i not in consequent_dst:
                    cur_consequent_output_by_transpose.append(i)
            cur_consequent_output_by_transpose.extend(consequent_dst)
            cur_output_reorder_index = cur_consequent_output_by_transpose[:]
            return cur_input_reorder_index, cur_output_reorder_index

        def calc_last_transpose_index():
            # base by input
            dst_in_ub_by_input = self._ori_permute[len(self._ori_permute) - len(consequent_dst):]
            cur_input_reorder_index = dst_in_ub_by_input[:]
            for i in range(dim_len):
                if i not in dst_in_ub_by_input:
                    cur_input_reorder_index.append(i)
            cur_consequent_output_by_transpose = cur_input_reorder_index[len(dst_in_ub_by_input):]
            cur_consequent_output_by_transpose.extend(dst_in_ub_by_input)
            cur_output_reorder_index = [self._ori_permute.index(i) for i in cur_consequent_output_by_transpose]
            return cur_input_reorder_index, cur_output_reorder_index

        dim_len = len(self._ori_permute)
        is_no_reorder_case = (self._none_cut and self._tiling_strategy != TilingStrategy.READ_ALIGN_NONE_CUT) \
            or self.Util.is_store_align_single(self._tiling_strategy) or self._const_none_cut
        if is_no_reorder_case:
            self._reorder_in_index = [i for i in range(dim_len)]
            self._reorder_out_index = [i for i in range(dim_len)]
            return

        # base by output
        is_last_transpose = self._ori_permute[dim_len - 1] != dim_len - 1
        src_in_ub_by_out = self._permute[self._ori_permute[self._low_ub_split_axis_index]:]
        consequent_dst = self._base_order[self._high_ub_split_axis_index:]
        last_has_cross = is_last_transpose and any([i in consequent_dst for i in src_in_ub_by_out])
        if last_has_cross:
            input_reorder_index, output_reorder_index = calc_last_cross_index()
        elif is_last_transpose:
            input_reorder_index, output_reorder_index = calc_last_transpose_index()
            self._last_transpose = True
        else:
            input_reorder_index = [i for i in range(dim_len)]
            consequent_output_by_transpose = []
            for i in self._permute:
                if i not in consequent_dst:
                    consequent_output_by_transpose.append(i)
            consequent_output_by_transpose.extend(consequent_dst)
            output_reorder_index = consequent_output_by_transpose[:]

        self._reorder_in_index = input_reorder_index
        self._reorder_out_index = output_reorder_index

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._out_tensors)

    def _do_cache_write_reorder(self):
        sch = self._schedule
        input_buffer = None
        for buffer in self._cache_read_buffer_tensors_map:
            src_reorder_axis = [buffer.op.axis[i] for i in self._reorder_in_index]
            sch[buffer].reorder(*src_reorder_axis)
            buffer_tensor = self._schedule.cache_write(buffer, self._scope)
            input_buffer = buffer
        self._cache_read_buffer_tensors_map[buffer_tensor] = input_buffer
        for tensor_i in self._cache_write_tensors:
            dst_reorder_axis = [tensor_i.op.axis[i] for i in self._reorder_out_index]
            sch[tensor_i].reorder(*dst_reorder_axis)
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensors_map[buffer_tensor] = tensor_i

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._middle_tensors:
            sch[tensor_i].set_scope(self._scope)

    def _calc_storage_bound(self):
        dim_len = len(self._ori_permute)
        is_cut_last = self._ori_permute[dim_len - 1] == dim_len - 1 \
            and self.Util.is_general(self._tiling_strategy) \
            and self._high_ub_split_axis_index == self._low_ub_split_axis_index \
            and self._high_ub_split_axis_index == len(self._permute) - 1
        if self.Util.is_store_align(self._tiling_strategy) or self._is_const_store_align:
            self._coexisting_quantity = 1
            self._ub_size -= 32
        elif self._last_transpose or self._is_const_align or is_cut_last:
            self._coexisting_quantity = 2
            if self._last_transpose and self._max_dtype_bytes > 2:
                self._coexisting_quantity = 4
        elif self.Util.is_all_read_align(self._tiling_strategy):
            self._coexisting_quantity = 2
        else:
            factor = 32 if self._max_dtype_bytes == 1 else 16
            self._coexisting_quantity = factor * 2 + 2

    def _calc_reorder(self):
        pass

    def _calc_tiling(self):
        funcs = {TilingStrategy.GENERAL: self._calc_tiling_general,
                 TilingStrategy.READ_ALIGN: self._calc_tiling_general,
                 TilingStrategy.NONE_CUT: self._calc_tiling_none_cut,
                 TilingStrategy.STORE_ALIGN_NONE_CUT: self._calc_tiling_none_cut,
                 TilingStrategy.READ_ALIGN_NONE_CUT: self._calc_tiling_none_cut,
                 TilingStrategy.CONST: self._calc_tiling_const,
                 TilingStrategy.STORE_ALIGN_SINGLE: self._calc_tiling_one}
        funcs[self._tiling_strategy]()

    def _calc_tiling_one(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis_index
        u_h_i = self._high_ub_split_axis_index
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_h_bound = (1, util.get_bound(shape[u_h_i])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._high_ub_factor = operation.var_inner("_ub_factor_" + str(u_h_i), u_h_bound)

    def _calc_tiling_general(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._block_split_axis_index
        u_l_i = self._low_ub_split_axis_index
        u_h_i = self._high_ub_split_axis_index
        b_bound = (1, util.get_bound(shape[b_i])[1])
        u_l_bound = (1, util.get_bound(shape[u_l_i])[1])
        u_h_bound = (1, util.get_bound(shape[u_h_i])[1])
        self._block_factor = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_split_same_axis = u_l_i == u_h_i
        if self._ub_split_same_axis:
            ub_factor = operation.var_inner("_ub_factor_" + str(u_l_i), u_l_bound)
            self._low_ub_factor = ub_factor
            self._high_ub_factor = ub_factor
        else:
            self._low_ub_factor = operation.var_inner("_ub_factor_" + str(u_l_i), u_l_bound)
            self._high_ub_factor = operation.var_inner("_ub_factor_" + str(u_h_i), u_h_bound)

    def _calc_tiling_none_cut(self):
        pass

    def _calc_tiling_const(self):
        inputs = []
        for _input in self._input_tensors:
            input_shape = util.shape_to_list(_input.shape)
            inputs.append({"shape": input_shape, "dtype": _input.dtype})
        res = self._out
        output_shape = util.shape_to_list(res.shape)
        outputs = [{"shape": output_shape, "dtype": res.dtype}]
        const_compile_info = get_compile_info()
        new_compile_info = {
            CompileInfo.CORE_NUM: util.get_core_num(),
            CompileInfo.UB_SIZE: util.get_ub_size(),
            "_permute": self._permute,
            "_ori_permute": self._ori_permute,
            "_mergeable": [0 for _ in self._ori_permute],
            "_only_const_tiling": True,
            "_is_const": False
        }
        const_compile_info.update(new_compile_info)

        op_type = "AutoTiling"
        run_info = op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)

        dim_len = len(self._ori_permute) - 1
        is_nlast_transpose = self._ori_permute[dim_len] == dim_len
        ele_in_block = TYPE_IN_BLOCK[DTYPE_BYTE_MAPPING[self._out.dtype]]
        self._is_const_align = is_nlast_transpose and (
            output_shape[dim_len] > ALIGN_THRESHOLD or output_shape[dim_len] %
            ele_in_block == 0)
        self._is_const_store_align = is_nlast_transpose and output_shape[dim_len] > ALIGN_THRESHOLD \
            and output_shape[dim_len] % ele_in_block != 0 or len(
            self._ori_permute) == 1
        if self._is_const_store_align:
            tiling_format = {
                "need_multi_core": "int",
                "block_axis": "int",
                "block_factor": "int",
                "high_ub_split_axis": "int",
                "high_ub_factor": "int"
            }
            tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
            self._block_dims = run_info["block_dim"]
            self._need_do_block = True if tiling_data["need_multi_core"] > 0 else False
            if self._need_do_block:
                self._block_split_axis_index = tiling_data["block_axis"]
                self._block_factor = tiling_data["block_factor"]
                self._high_ub_split_axis_index = tiling_data["high_ub_split_axis"]
                self._high_ub_factor = tiling_data["high_ub_factor"]
                self._ub_split_same_axis = self._low_ub_split_axis_index == self._high_ub_split_axis_index
        else:
            tiling_format = {
                "need_multi_core": "int",
                "block_axis": "int",
                "block_factor": "int",
                "low_ub_split_axis": "int",
                "low_ub_factor": "int",
                "high_ub_split_axis": "int",
                "high_ub_factor": "int"
            }
            tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
            self._block_dims = run_info["block_dim"]
            self._need_do_block = True if tiling_data["need_multi_core"] > 0 else False
            if self._need_do_block:
                self._block_split_axis_index = tiling_data["block_axis"]
                self._block_factor = tiling_data["block_factor"]
                self._low_ub_split_axis_index = tiling_data["low_ub_split_axis"]
                self._low_ub_factor = tiling_data["low_ub_factor"]
                self._high_ub_split_axis_index = tiling_data["high_ub_split_axis"]
                self._high_ub_factor = tiling_data["high_ub_factor"]
                self._ub_split_same_axis = self._low_ub_split_axis_index == self._high_ub_split_axis_index
        self._const_none_cut = not self._need_do_block

    def _calc_compute_inline(self):
        for buffer, source in self._cache_read_buffer_tensors_map.items():
            if util.is_placeholder(source):
                self._compute_inline_tensors.add(buffer)

        ub_in_index = [self._permute[i] for i in self._reorder_in_index]
        ub_out_index = [self._base_order[i] for i in self._reorder_out_index]
        src_in_ub_by_out = ub_in_index[self._high_ub_split_axis_index:]
        dst_in_ub = ub_out_index[self._high_ub_split_axis_index:]
        dim_len = len(self._ori_permute)
        nlast_transpose = self._ori_permute[dim_len - 1] == dim_len - 1
        const_only_one = self.Util.is_const(self._tiling_strategy) and nlast_transpose and \
            len(src_in_ub_by_out) == len(dst_in_ub) and len(dst_in_ub) == 2 and \
            self._high_ub_factor == 1 and self._low_ub_factor == 1

        # no transpose under the split
        is_ub_no_conv = self.Util.is_store_align(self._tiling_strategy) or self._is_const_store_align
        if src_in_ub_by_out == dst_in_ub or const_only_one or is_ub_no_conv:
            for buffer, source in self._cache_write_buffer_tensors_map.items():
                self._compute_inline_tensors.add(buffer)

    def _calc_multi_core(self):
        pass

    def _calc_compute_at(self):
        if self._none_cut or self._const_none_cut:
            return

        for tensor_i in self._input_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._middle_tensors - self._compute_inline_tensors:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_read_buffer_tensors_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_write_buffer_tensors_map:
            self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

    def _calc_storage_align(self):
        is_no_align = self.Util.is_none_cut(self._tiling_strategy) \
            or (self._const_none_cut and not self._is_const_align) \
            or self.Util.is_all_read_align(self._tiling_strategy)
        if is_no_align:
            return

        is_last_align = self.Util.is_store_align(self._tiling_strategy) \
            or self._is_const_align or self._is_const_store_align
        if is_last_align:
            input_split_ub = self._ori_permute[self._low_ub_split_axis_index]
            dim_len = len(self._ori_permute)
            need_align = dim_len - input_split_ub >= 2 and dim_len - self._high_ub_split_axis_index >= 2
            if need_align:
                self._input_align_axis = -2
                self._output_align_axis = -2
                return

        # base by input
        output_in_ub = self._ori_permute[self._high_ub_split_axis_index:]
        input_split_ub = self._ori_permute[self._low_ub_split_axis_index]
        out_in_input_split = []
        for axis in output_in_ub:
            if axis < input_split_ub:
                out_in_input_split.append(axis)
        ori_input_index = self._reorder_in_index
        input_align_axis = -1
        for index, value in enumerate(ori_input_index):
            if value in out_in_input_split:
                input_align_axis = index

        # base by output
        input_in_ub = self._permute[self._ori_permute[self._low_ub_split_axis_index]:]
        output_split_ub = self._high_ub_split_axis_index
        out_in_output_split = []
        for axis in input_in_ub:
            if axis < output_split_ub:
                out_in_output_split.append(axis)
        ori_output_index = self._reorder_out_index
        output_align_axis = -1
        for index, value in enumerate(ori_output_index):
            if value in out_in_output_split:
                output_align_axis = index

        self._input_align_axis = input_align_axis
        self._output_align_axis = output_align_axis

    def _calc_double_buffer(self):
        pass

    def _calc_constraints(self):
        pass

    def _calc_emit_insn(self):
        def calc_ub_permute():
            src_split_index = self._ori_permute[self._low_ub_split_axis_index]
            consequent_src = src_order[src_split_index:]
            consequent_dst = dst_order[self._high_ub_split_axis_index:]
            ub_src_order = [o for o in src_order if o in consequent_dst and o not in consequent_src]
            ub_src_order.extend(consequent_src)
            ub_dst_order = [o for o in dst_order if o in consequent_src and o not in consequent_dst]
            ub_dst_order.extend(consequent_dst)
            src_shape = util.shape_to_list(self._out.op.input_tensors[0].shape)
            new_ub_src_order = []
            for o in ub_src_order:
                if src_split_index == o:
                    if self._low_ub_factor != 1:
                        new_ub_src_order.append(o)
                elif src_shape[o] != 1:
                    new_ub_src_order.append(o)
            dst_split_index = self._ori_permute[self._high_ub_split_axis_index]
            new_ub_dst_order = []
            for o in ub_dst_order:
                if dst_split_index == o:
                    if self._high_ub_factor != 1:
                        new_ub_dst_order.append(o)
                elif src_shape[o] != 1:
                    new_ub_dst_order.append(o)
            remove_low_factor_one = self._low_ub_factor == 1 and src_split_index != dst_split_index
            if remove_low_factor_one:
                new_ub_dst_order.remove(src_split_index)
            remove_high_factor_one = self._high_ub_factor == 1 and src_split_index != dst_split_index
            if remove_high_factor_one:
                new_ub_src_order.remove(dst_split_index)
            return new_ub_src_order, new_ub_dst_order

        for source, target in self._cache_read_buffer_tensors_map.items():
            if source not in self._compute_inline_tensors:
                self._emit_insn_map[source] = [source.op.axis[0], DMA_COPY]

        for tensor_i in (self._middle_tensors - self._compute_inline_tensors):
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], tensor_i.op.tag]

        for source, target in (self._cache_write_buffer_tensors_map.items() - self._compute_inline_tensors):
            src_order = [self._base_order[i] for i in self._reorder_in_index]
            dst_order = [self._ori_permute[i] for i in self._reorder_out_index]
            need_ub_permute = self._tiling_case.tiling_strategy == TilingStrategy.GENERAL \
                or (self._tiling_case.tiling_strategy == TilingStrategy.CONST
                    and self._need_do_block and not self._is_const_align)
            if need_ub_permute:
                new_ub_src_order, new_ub_dst_order = calc_ub_permute()
            else:
                new_ub_src_order = src_order
                new_ub_dst_order = dst_order
            ub_permute = [new_ub_src_order.index(o) for o in new_ub_dst_order]
            if self._is_const_align or self.Util.is_all_read_align(self._tiling_strategy):
                insn_name = "dma_copy"
                self._emit_insn_map[source] = [source.op.axis[0], insn_name]
            else:
                ub_permute = tvm.expr.Call(
                    "handle",
                    "tvm_tuple",
                    ub_permute,
                    tvm.expr.Call.PureIntrinsic,
                    None,
                    0)
                attrs = {"src_in_dst_order": ub_permute}
                self._emit_insn_map[source] = [source.op.axis[0], "vector_transpose", attrs]

        for tensor_i in self._out_tensors:
            attrs = {"no_overlap": 2}
            if self.Util.is_all_read_align(self._tiling_strategy):
                attrs = {"no_overlap": 0}
            self._emit_insn_map[tensor_i] = [self._emit_insn_axis, DMA_COPY, attrs]

    def _do_tiling(self):
        funcs = {TilingStrategy.GENERAL: self._do_tiling_general,
                 TilingStrategy.READ_ALIGN: self._do_tiling_general,
                 TilingStrategy.NONE_CUT: self._do_tiling_none_cut,
                 TilingStrategy.STORE_ALIGN_NONE_CUT: self._do_tiling_none_cut,
                 TilingStrategy.READ_ALIGN_NONE_CUT: self._do_tiling_none_cut,
                 TilingStrategy.CONST: self._do_tiling_const,
                 TilingStrategy.STORE_ALIGN_SINGLE: self._do_tiling_one}
        funcs[self._tiling_strategy]()

    def _do_tiling_one(self):
        sch = self._schedule
        res = self._out

        u_o, u_i = sch[res].split(res.op.axis[self._high_ub_split_axis_index], factor=self._high_ub_factor)
        self._emit_insn_axis.emit_insn_axis = u_i
        self._compute_at_axis.compute_at_axis = u_o
        if self._block_split_axis_index == self._high_ub_split_axis_index:
            b_o, b_i = sch[res].split(u_o, factor=self._block_factor)
            self._compute_at_axis.compute_at_axis = b_i
        else:
            b_o, b_i = sch[res].split(res.op.axis[self._block_split_axis_index], factor=self._block_factor)

        self._block_split_out_axis = b_o
        self._block_split_in_axis = b_i

    def _do_tiling_general(self):
        sch = self._schedule
        res = self._out

        if self._ub_split_same_axis:
            u_o, u_i = sch[res].split(res.op.axis[self._low_ub_split_axis_index], factor=self._low_ub_factor)
            self._ub_split_out_axis = [u_o, u_o]
            self._ub_split_in_axis = [u_i, u_i]
            self._emit_insn_axis.emit_insn_axis = u_i
            if self._block_split_axis_index == self._low_ub_split_axis_index:
                b_o, b_i = sch[res].split(u_o, factor=self._block_factor)
                self._compute_at_axis.compute_at_axis = b_i
                self._ub_split_out_axis = [b_i, b_i]
            else:
                b_o, b_i = sch[res].split(res.op.axis[self._block_split_axis_index], factor=self._block_factor)
                self._compute_at_axis.compute_at_axis = u_o
        else:
            low_u_o, low_u_i = sch[res].split(res.op.axis[self._low_ub_split_axis_index], factor=self._low_ub_factor)
            u_o, u_i = sch[res].split(res.op.axis[self._high_ub_split_axis_index], factor=self._high_ub_factor)
            self._ub_split_out_axis = [low_u_o, u_o]
            self._ub_split_in_axis = [low_u_i, u_i]
            self._emit_insn_axis.emit_insn_axis = low_u_i
            self._compute_at_axis.compute_at_axis = u_o
            if self._block_split_axis_index == self._low_ub_split_axis_index:
                b_o, b_i = sch[res].split(low_u_o, factor=self._block_factor)
                self._ub_split_out_axis = [b_i, u_o]
            elif self._block_split_axis_index == self._high_ub_split_axis_index:
                b_o, b_i = sch[res].split(u_o, factor=self._block_factor)
                self._compute_at_axis.compute_at_axis = b_i
                self._ub_split_out_axis = [low_u_o, b_i]
            else:
                b_o, b_i = sch[res].split(res.op.axis[self._block_split_axis_index], factor=self._block_factor)

        self._block_split_out_axis = b_o
        self._block_split_in_axis = b_i

    def _do_tiling_none_cut(self):
        res = self._out
        self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_tiling_const(self):
        if self._need_do_block:
            if self._is_const_store_align:
                self._do_tiling_one()
            else:
                self._do_tiling_general()
        else:
            res = self._out
            self._emit_insn_axis.emit_insn_axis = res.op.axis[0]

    def _do_reorder(self):
        def do_block_reorder():
            if self._block_split_axis_index == self._low_ub_split_axis_index \
                    or self._block_split_axis_index == self._high_ub_split_axis_index:
                for i in range(self._high_ub_split_axis_index):
                    if i != self._low_ub_split_axis_index and i not in low_ub_inner:
                        block_axis.append(i)
            else:
                for i in range(self._block_split_axis_index):
                    if i not in low_ub_inner and i != self._low_ub_split_axis_index:
                        block_axis.append(i)

            for i in block_axis:
                reorder_axis.append(res.op.axis[i])

        def do_ub_outer_reorder():
            if self._block_split_axis_index == self._high_ub_split_axis_index and not self._ub_split_same_axis:
                reorder_axis.append(self._ub_split_out_axis[0])
            reorder_axis.append(self._block_split_out_axis)
            self._block_fuse_axis = reorder_axis[:]

            for i in range(self._high_ub_split_axis_index):
                if i == self._block_split_axis_index:
                    reorder_axis.append(self._block_split_in_axis)
                elif i == self._low_ub_split_axis_index:
                    if self._block_split_axis_index != self._high_ub_split_axis_index:
                        reorder_axis.append(self._ub_split_out_axis[0])
                elif i not in block_axis and i not in low_ub_inner:
                    reorder_axis.append(res.op.axis[i])
            reorder_axis.append(self._ub_split_out_axis[1])

        def do_ub_inner_reorder():
            input_in_ub = []
            for i in range(self._high_ub_split_axis_index):
                if i in low_ub_inner:
                    input_in_ub.append(res.op.axis[i])
                if i == self._low_ub_split_axis_index:
                    input_in_ub.append(self._ub_split_in_axis[0])

            if len(input_in_ub) != 0:
                self._emit_insn_axis.emit_insn_axis = input_in_ub[0]

            reorder_axis.extend(input_in_ub)
            reorder_axis.append(self._ub_split_in_axis[1])

            for i in range(self._high_ub_split_axis_index + 1, len(res.shape)):
                reorder_axis.append(res.op.axis[i])

        if self._none_cut or self._const_none_cut:
            return

        if self.Util.is_store_align_single(self._tiling_strategy) or self._is_const_store_align:
            return

        sch = self._schedule
        res = self._out
        reorder_axis = []
        # base by output
        low_ub_inner = self._permute[self._ori_permute[self._low_ub_split_axis_index] + 1:]
        block_axis = []
        do_block_reorder()

        do_ub_outer_reorder()

        do_ub_inner_reorder()

        sch[res].reorder(*reorder_axis)

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._middle_tensors.union(
            self._cache_read_buffer_tensors_map.keys()).union(
            self._cache_write_buffer_tensors_map.keys())

        tensor_space = self._ub_size // self._coexisting_quantity
        if self._enable_db:
            tensor_space = self._ub_size // 2 // self._coexisting_quantity
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_buffer_size(storage_bound)

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()

    def _do_multi_core(self):
        if self._none_cut or self._const_none_cut:
            return

        sch = self._schedule
        res = self._out
        if self.Util.is_store_align_single(self._tiling_strategy) or self._is_const_store_align:
            for i in range(self._block_split_axis_index):
                self._block_fuse_axis.append(res.op.axis[i])
            self._block_fuse_axis.append(self._block_split_out_axis)

        block_bind_axis = sch[res].fuse(*[x for x in self._block_fuse_axis])
        block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(block_bind_axis, block)

    def _do_compute_at(self):
        sch = self._schedule
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1].compute_at_axis)

    def _do_storage_align(self):
        sch = self._schedule
        res = self._out
        align_factor = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[res.dtype]
        offset = 0

        if self._input_align_axis != -1:
            for source, target in self._cache_read_buffer_tensors_map.items():
                if source not in self._compute_inline_tensors:
                    sch[source].storage_align(source.op.axis[self._input_align_axis], align_factor, offset)

        if self._output_align_axis != -1:
            if self._last_transpose and align_factor < 16:
                align_factor *= 16
            for source, target in self._cache_write_buffer_tensors_map.items():
                sch[source].storage_align(source.op.axis[self._output_align_axis], align_factor, offset)

    def _do_double_buffer(self):
        if self._enable_db:
            sch = self._schedule

            tensors = self._middle_tensors.union(
                self._cache_read_buffer_tensors_map.keys()).union(
                self._cache_write_buffer_tensors_map.keys())

            for tensor_i in tensors:
                sch[tensor_i].double_buffer()

    def _do_constraints(self):
        pass

    def _do_emit_insn(self):
        sch = self._schedule

        for tensor_i, param in self._emit_insn_map.items():
            emit_insn_axis = param[0]
            if isinstance(emit_insn_axis, self.EmitInsn):
                emit_insn_axis = emit_insn_axis.emit_insn_axis
            if len(param) > 2:
                sch[tensor_i].emit_insn(emit_insn_axis, param[1], param[2])
            else:
                sch[tensor_i].emit_insn(emit_insn_axis, param[1])

    def _add_compile_info(self):
        if CompileInfo.CORE_NUM in get_compile_info() and self._tiling_strategy != TilingStrategy.CONST:
            return

        operation.add_compile_info_inner(CompileInfo.CORE_NUM, util.get_core_num())
        operation.add_compile_info_inner(CompileInfo.UB_SIZE, util.get_ub_size())
        operation.add_compile_info_inner("_ori_permute", self._ori_permute)
        operation.add_compile_info_inner("_permute", self._permute)
        shape = util.shape_to_list(self._out.shape)
        shape_is_var = []
        for index, value in enumerate(shape):
            if isinstance(shape[self._permute[index]], int):
                shape_is_var.append(False)
            else:
                shape_is_var.append(True)
        operation.add_compile_info_inner("_transpose_vars", shape_is_var)
        operation.add_compile_info_inner("_only_const_tiling", False)
        if self.Util.is_const(self._tiling_strategy):
            operation.add_compile_info_inner("_is_const", True)
            operation.add_compile_info_inner("_const_dims", self._block_dims)
        else:
            operation.add_compile_info_inner("_is_const", False)

    def _check_tiling_case(self):
        # base by output
        low_input_right = self._permute[self._ori_permute[self._low_ub_split_axis_index] + 1:]
        high_output_right = self._base_order[self._high_ub_split_axis_index + 1:]
        cond0 = self._high_ub_split_axis_index in low_input_right or self._block_split_axis_index in low_input_right
        cond1 = self._low_ub_split_axis_index in high_output_right
        if cond0 or cond1:
            return False
        return True

    def _check_tiling_factor(self):
        def calc_low_bound(inner_shape):
            low_bound = 1
            for item in inner_shape[::-1]:
                cur_bound = util.get_bound(item)[0]
                if cur_bound is None:
                    return False
                low_bound *= cur_bound
            if low_bound == 1 and not self.Util.is_read_align(self._tiling_strategy):
                align_factor = BLOCK_SIZE_BYTE // self._max_dtype_bytes
                if self._last_transpose and align_factor < 16:
                    align_factor *= 16
                low_bound = align_factor
            return low_bound

        if self.Util.is_const(self._tiling_strategy):
            return True
        shape = util.shape_to_list(self._out.shape)
        input_low_bound = 1
        low_ub_inner = []
        if not self.Util.is_single_cut(self._tiling_strategy):
            low_ub_inner = self._permute[self._ori_permute[self._low_ub_split_axis_index] + 1:]
            inner_input_shape = [shape[i] for i in low_ub_inner]
            inner_input_shape.append(self._low_ub_factor)
            input_low_bound = calc_low_bound(inner_input_shape)
        if self.Util.is_all_none_cut(self._tiling_strategy):
            high_ub_inner = list(set(self._base_order[self._high_ub_split_axis_index:]) - set(low_ub_inner))
            inner_output_shape = [shape[i] for i in high_ub_inner]
        else:
            high_ub_inner = list(set(self._base_order[self._high_ub_split_axis_index + 1:]) - set(low_ub_inner))
            inner_output_shape = [shape[i] for i in high_ub_inner]
            if not self._ub_split_same_axis:
                inner_output_shape.append(self._high_ub_factor)
        output_low_bound = 1
        if len(inner_output_shape) != 0:
            output_low_bound = calc_low_bound(inner_output_shape)
        if not self._tensor_space // self._max_dtype_bytes >= (input_low_bound * output_low_bound):
            return False
        return True
