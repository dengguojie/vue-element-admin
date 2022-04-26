#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
import copy
from copy import deepcopy
from typing import Optional

from tbe import tvm
from tbe.common.utils import op_tiling
from tbe.common.platform import intrinsic_check_support
from tbe.dsl.base import operation
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.operation import get_compile_info

from ... import util
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import ElewisePattern
from ...constants import FAKE_NODE_TAG
from ...constants import INSN_MAPPING
from ...constants import Pattern
from ...constants import SUPPORT_SCALAR_INSNS
from ...constants import DST_SRC_NO_REUSE_SET
from ...constants import TERNARY_INSNS
from ...schedule import Schedule
from .elewise_tilingcase import TilingStrategy

DEFAULT = "default"

# block size in D architecture
BLOCK_SIZE_BYTE = 32

# tiling factor align
ONE_DIM_ALIGN = 128
SPECIAL_FACTOR_ALIGN = 256
# Cast Special Nodes
SPECIAL_CAST_DEPENDENT = 2

CONST = "const"
VECTOR = "vector"

# vcmpsel constant
VCMP_INPUT_NUMBER = 2
VSEL_INPUT_NUMBER = 3
VCMPSEL_INPUT_NUMBER = 4

# TYPE DOUNDS
TYPE_DOUNDS = {
    1: (1, 32767),
    2: (1, 32767),
    4: (1, 16383),
    8: (1, 8191),
}


# 'pylint: disable=R0902, R0903
class ElewiseSchedule(Schedule):
    """
    ElewiseSchedule
    """

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.ELEMWISE]

    @classmethod
    def get_supported_sub_pattern(cls):
        return [ElewisePattern.E_0]

    def __init__(self, outs, tiling_case):
        self._out = None  # type: Optional[tvm.tensor.Tensor]
        self._outs = outs
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._is_db = self._tiling_case.get("is_need_db", False)
        self._is_one_dim = self._tiling_case.get("is_one_dim", False)
        self._mode = operation.get_context().get_current_compute().get("_mode")

        self._scope = "local.UB"

        self._input_tensors = set()
        self._middle_tensors = set()
        self._pure_middle_tensors = set()
        self._middle_out_tensors = set()
        self._out_tensors = set()

        self._dtypes = set()
        self._outs_dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._tensor_space = None
        self._ub_size = util.get_ub_size()
        self._correct_factor = 2 if self._is_db else 1
        self._tmp_ub_size = 0

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

        self._compute_at_map = {}

        self._copy_out_tensors = set()
        self._remove_out_tensors = set()

        # just for const tiling
        self._need_do_block = False
        self._block_dims = 1
        self._block_split_axis = -1
        self._block_factor = 1
        self._ub_split_axis = 0
        self._ub_factor = 1
        self._ub_factor_align = ONE_DIM_ALIGN

        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._block_bind_axis = None
        self._compute_at_axis = None
        self._compute_at_axis_idx = None
        self._emit_insn_axis = None

        self._ir_axes = []
        self._inner_shape = []

        self._mem_reuse_map = {}

        self._emit_insn_map = {}

    def do_schedule(self):
        """
        :return:
        """
        self._construct_compute_graph()

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case.get("key")

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._calc_storage_bound()

        self._calc_tiling()
        self._do_tiling()

        self._do_storage_bound()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_double_buffer()
        self._do_double_buffer()

        self._calc_mem_reuse()
        self._do_mem_reuse()

        self._calc_emit_insn()
        self._do_emit_insn()
        self._add_compile_info()

        return self._schedule if self._check_tiling_case() else None

    def _construct_compute_graph(self):
        def _pre_handle_placeholder(tensors):
            for out in tensors:
                if util.is_placeholder(out):
                    self._copy_out_tensors.add(_copy_node(out))
                    self._remove_out_tensors.add(out)
            self._out_tensors.update(self._copy_out_tensors)
            self._out_tensors = self._out_tensors - self._remove_out_tensors

        self._out_tensors = set(self._outs)
        _pre_handle_placeholder(self._out_tensors)

        visited_tensors = set()
        for out in self._out_tensors:
            self.__dfs_sub_graph(out, visited_tensors)
            self._dtypes.add(out.dtype)
            self._outs_dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING.get(dtype) for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)
        # out uint1 dtype needs ub_factor align to 256
        if "uint1" in self._outs_dtypes:
            self._ub_factor_align = max(self._ub_factor_align, SPECIAL_FACTOR_ALIGN)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors
        self._middle_out_tensors = self._middle_tensors.intersection(self._out_tensors)

        pure_out_tensors = list(self._out_tensors - self._middle_out_tensors)
        if len(pure_out_tensors) > 1:
            self._out = _fake_node(pure_out_tensors)
            self.__dfs_sub_graph(self._out, visited_tensors)
        else:
            self._out = pure_out_tensors[0]

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)
        self._cache_read_tensors.update(self._middle_out_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(tensor_i, self._scope, self._in_out_map.get(tensor_i))
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._placeholder_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._middle_out_tensors:
                self._middle_out_cache_read_buffer_map[tensor_i] = buffer_tensor

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._out_tensors - self._copy_out_tensors)

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

    def _calc_tiling(self):
        funcs = {TilingStrategy.ONE_CUT: self._calc_tiling_one_cut,
                 TilingStrategy.STATIC: self._calc_tiling_static,
                 TilingStrategy.CONST: self._calc_tiling_const,
                 }
        funcs[self._tiling_strategy]()

    def _calc_tiling_one_cut(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._tiling_case.get("block_tiling_axis")
        u_i = self._tiling_case.get("ub_tiling_axis")
        b_bound = (1, util.get_bound(shape[b_i])[1])
        if self._is_one_dim:
            u_bound = TYPE_DOUNDS.get(self._max_dtype_bytes)
        else:
            u_bound = self._tiling_case.get("ub_factor_bound")
        if u_bound is None:
            u_bound = (1, util.get_bound(shape[u_i])[1])
        self._block_tiling_vars[b_i] = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = operation.var_inner("_ub_factor_" + str(u_i), u_bound)

    def _calc_tiling_static(self):
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_i = self._tiling_case.get("block_tiling_axis")
        u_i = self._tiling_case.get("ub_tiling_axis")
        b_bound = (1, util.get_bound(shape[b_i])[1])
        self._block_tiling_vars[b_i] = operation.var_inner("_block_factor_" + str(b_i), b_bound)
        self._ub_tiling_vars[u_i] = self._tiling_case.get("ub_tiling_factor")

    def _calc_tiling_const(self):
        res = self._out
        output_shape = util.shape_to_list(res.shape)
        if output_shape == [0]:
            self._block_dims = 1
            self._need_do_block = False
            return

        max_dim_length = len(output_shape)
        inputs = []
        for _input in self._input_tensors:
            input_shape = util.shape_to_list(_input.shape)
            inputs.append({"shape": input_shape, "dtype": _input.dtype})
        outputs = [{"shape": output_shape, "dtype": res.dtype}]
        if len(inputs) == 0:
            max_dim_length = 0
            inputs = copy.deepcopy(outputs)

        # pure eletwise delete double tmp size
        if len(output_shape) == 1:
            self._is_one_dim = True

        max_available_ub = ((((self._ub_size - self._tmp_ub_size) // self._coexisting_quantity) // BLOCK_SIZE_BYTE) *
                            BLOCK_SIZE_BYTE) // self._max_dtype_bytes
        max_available_ub_db = ((((self._ub_size - 2 * self._tmp_ub_size) // 2 // self._coexisting_quantity)
                                // BLOCK_SIZE_BYTE) * BLOCK_SIZE_BYTE) // self._max_dtype_bytes
        base_info = {"000": [util.get_core_num(), self._max_dtype_bytes, max_available_ub, max_available_ub_db]}

        const_compile_info = {
            CompileInfo.FLAG_INFO: [True],
            CompileInfo.BASE_INFO: base_info,
            CompileInfo.UB_FACTOR_ALIGN: self._ub_factor_align
        }
        const_compile_info.update(get_compile_info())

        op_type = "AutoTiling"
        run_info = op_tiling.do_op_tiling(op_type, const_compile_info, inputs, outputs)
        tiling_format = {
            "need_multi_core": "int",
            "block_axis": "int",
            "block_factor": "int",
            "ub_axis": "int",
            "ub_factor": "int",
            "is_need_db": "int"}
        tiling_data = op_tiling.decode(run_info.get('tiling_data'), tiling_format)
        self._block_dims = run_info.get("block_dim")
        self._need_do_block = True if tiling_data.get("need_multi_core") > 0 else False
        if self._need_do_block:
            self._block_split_axis = tiling_data.get("block_axis")
            self._block_factor = tiling_data.get("block_factor")
            self._ub_split_axis = tiling_data.get("ub_axis")
            self._ub_factor = tiling_data.get("ub_factor")

        self._is_db = True if tiling_data.get("is_need_db", 0) == 1 else False

    def _do_tiling(self):
        funcs = {TilingStrategy.ONE_CUT: self._do_tiling_one_cut,
                 TilingStrategy.STATIC: self._do_tiling_static,
                 TilingStrategy.CONST: self._do_tiling_const,
                 }
        funcs[self._tiling_strategy]()

    def _do_tiling_one_cut(self):
        sch = self._schedule
        res = self._out
        shape = util.shape_to_list(res.shape)
        b_idx = self._tiling_case.get("block_tiling_axis")
        u_idx = self._tiling_case.get("ub_tiling_axis")
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

    def _calc_multi_core(self):
        pass

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _calc_compute_at(self):
        if self._tiling_strategy == TilingStrategy.CONST and not self._need_do_block:
            self._compute_at_map.clear()
            return

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

        for tensor_i in self._pure_middle_tensors:
            self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], get_insn(tensor_i)]

        for source, target in self._cache_write_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], get_insn(target)]

        if len(self._out_tensors) > 1:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [tensor_i.op.axis[0], "dma_copy"]
            if len(self._out_tensors) - len(self._middle_out_tensors) > 1:
                self._emit_insn_map[self._out] = [self._emit_insn_axis, "phony_insn"]
            else:
                self._emit_insn_map[self._out] = [self._emit_insn_axis, "dma_copy"]
        else:
            for tensor_i in self._out_tensors:
                self._emit_insn_map[tensor_i] = [self._emit_insn_axis, "dma_copy"]

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            if len(param) > 2:
                sch[tensor_i].emit_insn(param[0], param[2])

            if tensor_i in self._out_tensors and self._is_one_dim:
                sch[tensor_i].emit_insn(param[0], param[1], attrs={"no_overlap": 0})
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
        for tensor_i in self._out_tensors | self._pure_middle_tensors:
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

    def _do_mem_reuse(self):
        sch = self._schedule
        for _a, _b in self._mem_reuse_map.items():
            for b_i in _b:
                sch[_a].reused_by(b_i)

    def _calc_storage_bound(self):
        def _correct_ub_size_by_cmp_sel(_tensor):
            if util.is_vcmp_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VCMP_INPUT_NUMBER - len(_tensor.op.input_tensors))
            if util.is_vsel_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))
                if (util.is_v200() or util.is_v220()) and (VSEL_INPUT_NUMBER == len(_tensor.op.input_tensors)):
                    self._tmp_ub_size += BLOCK_SIZE_BYTE
            if util.is_vcmpsel_insn(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE * (VCMPSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))

        def _dst_can_not_reuse_src(_tensor):
            # get tensor insn
            _tensor_insn = util.get_dsl_insn(_tensor)

            # place hold tensor need one in ub
            is_place_hold = util.is_placeholder(_tensor)

            # check by insn if dst can reuse src
            dst_no_reuse_src_insn = _tensor_insn in DST_SRC_NO_REUSE_SET

            # check by dependent_map if dst can reuse src
            _tensor_inputs = _tensor.op.input_tensors
            dependent_keys = dependent_map.keys()
            # if inputs(src) are all in dependent after refresh dependent, dst can not reuse inputs
            dst_no_src_dependent = set(_tensor_inputs).issubset(set(dependent_keys))

            return is_place_hold or dst_no_reuse_src_insn or dst_no_src_dependent

        def _cast_complex_instructions(_tensor):
            """
            Check if cast impl by complex instructions
            """
            if not intrinsic_check_support("Intrinsic_vconv", "s322s64") and \
                util.get_dsl_insn(_tensor) == "elewise_single_cast" and \
                    _tensor.op.input_tensors:
                src_dtype = _tensor.op.input_tensors[0].dtype
                cur_dtype = _tensor.dtype
                return (src_dtype == "int64" and cur_dtype == "int32") or \
                    (src_dtype == "int32" and cur_dtype == "int64")
            return False

        def _calc_current_coexist_node(_tensor):
            # one of the input of the ternary instruction must be reused with the output
            _current_coexist_node = len(dependent_map)

            _refresh_dependent(_tensor)

            # check if cast tensor impl by complex instructions
            if _cast_complex_instructions(_tensor):
                _current_coexist_node += SPECIAL_CAST_DEPENDENT
                if _tensor.dtype == "int64":
                    self._ub_factor_align = max(SPECIAL_FACTOR_ALIGN, self._ub_factor_align)

            # check if all src be used later
            if _dst_can_not_reuse_src(_tensor):
                _current_coexist_node += 1

            # correct ub size in broadcast absorb
            if util.need_temp_space(_tensor) or _need_external_space(_tensor):
                self._tmp_ub_size += BLOCK_SIZE_BYTE

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)

            return _current_coexist_node

        def _r_coexisting(_tensor):
            if _tensor in dependent_map:
                return len(dependent_map)
            _need_coexist_node = []
            for _tensor_i in _tensor.op.input_tensors:
                _need_coexist_node.append(_r_coexisting(_tensor_i))

            _current_coexist_node = _calc_current_coexist_node(_tensor)

            _need_coexist_node.append(_current_coexist_node)

            if _tensor not in dependent_map:
                dependent_map[_tensor] = self._in_out_map[_tensor].copy()
            return max(_need_coexist_node)

        def _refresh_dependent(_tensor):
            for _tensor_i in _tensor.op.input_tensors:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        def _need_external_space(_tensor):
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
            # last node cal current node
            current_coexist_node = _calc_current_coexist_node(self._out)
            coexisting_quantities.append(current_coexist_node)

        self._coexisting_quantity = max(coexisting_quantities)

        if self._coexisting_quantity == 1:
            self._tmp_ub_size += BLOCK_SIZE_BYTE

        # in order to improve performance, add one node
        if len(self._input_tensors) >= 2:
            self._coexisting_quantity += 1

    def _do_storage_bound(self):
        self._correct_factor = 2 if self._is_db else 1
        # delete tmp size
        self._ub_size -= self._correct_factor * self._tmp_ub_size
        tensor_space = self._ub_size // self._coexisting_quantity
        if self._is_db:
            tensor_space = tensor_space // 2
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

        # adjust storage bound by tiling handle one dime (128 align)
        if self._is_one_dim and self._tensor_space > ONE_DIM_ALIGN:
            self._tensor_space = self._tensor_space // ONE_DIM_ALIGN * ONE_DIM_ALIGN

        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_buffer_size(storage_bound)

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
            cpt_compute.add("_const_block_dim", self._block_dims)
        else:
            cpt_schedule.add(CompileInfo.MAX_DTYPE, self._max_dtype_bytes)
            cpt_schedule.add(CompileInfo.COEXISTING_QUANTITY, self._coexisting_quantity)
            cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
            cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())
            cpt_schedule.add("_tiling_key", self._schedule.tiling_key)

        operation.add_compile_info_inner(CompileInfo.UB_FACTOR_ALIGN, self._ub_factor_align)

    def __dfs_sub_graph(self, out, visited_tensors: set):
        for tensor_i in out.op.input_tensors:
            util.merge_value(self._in_out_map, tensor_i, out)
            self._dtypes.add(tensor_i.dtype)

            if util.is_placeholder(tensor_i):
                self._input_tensors.add(tensor_i)
            else:
                self._middle_tensors.add(tensor_i)

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

def _copy_node(tensor):
    shape = tensor.shape
    with tvm.tag_scope("dma_copy"):
        res = tvm.compute(shape, lambda *i: tensor(*i), name ="copy_node")
    return res
