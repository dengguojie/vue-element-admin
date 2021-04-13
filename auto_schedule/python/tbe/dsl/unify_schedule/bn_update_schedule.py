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
batch_normalization_forward_training_update
"""
from __future__ import absolute_import
from __future__ import division
from math import sqrt

from tbe.common.utils import shape_to_list
from tbe import tvm
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base import operation
from . import util, DTYPE_BYTE_MAPPING, TERNARY_INSNS, FAKE_NODE_TAG, CompileInfo, INSN_MAPPING
from .bn_update_tilingcase import TilingStrategy
from tbe.dsl.base.expr_compare import expr_equal
from tbe.dsl.base.operation import add_compile_info
import tbe


@register_schedule(pattern=Pattern.BN_UPDATE)
def schedule(outs, tiling_case):
    """
    :param outs:
    :param tiling_case: tiling cases
    :return: schedules
    """
    return BnUpdateSchedule(outs, tiling_case).do_schedule()


class BnUpdateSchedule:
    """
    BnUpdateSchedule
    """

    def __init__(self, outs, tiling_case):
        self._out = None
        self._outs = outs
        self._x_input = None
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")
        self._is_db = self._tiling_case.get("is_need_db", True)
        self._is_pure_eletwise = self._tiling_case.get("is_pure_eletwise", False)

        self._scope = "local.UB"

        self._input_tensors = set()
        self._middle_tensors = set()
        self._out_tensors = set()

        self._broadcast_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._compute_inline_broadcast = set()
        self._broadcast_axis_num = {}

        self._max_dtype_bytes = self._tiling_case["max_dtype_bytes"]
        self._tensor_space = self._tiling_case["tensor_space"]
        self._ub_size = util.get_ub_size()

        self._broadcast_not_last_axis_tensors = []
        self._mid_tensor_dst_tensor_map = {}
        self._out_compute_at_axis = None
        self._fake_node_ub = None
        self._core_num = util.get_core_num()

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

        self._block_bind_axis = None
        self._compute_at_axis = None
        self._emit_insn_axis = None
        self._block_inner_axis = None

        self._constraints = set()

        self._mem_reuse_map = {}
        self._data_reuse_map = {}

        self._emit_insn_map = {}

    def do_schedule(self):
        """
        :return: schedules
        """
        self._construct_compute_graph()

        def check_input_res_num(res):
            is_res_num_valid = len(res) != 5
            if is_res_num_valid:
                raise RuntimeError(
                    "Batch normalization update output nums should be 5, \
                    current is %d." % (len(res)))

        check_input_res_num(self._outs)

        save_mean_reduce = self._outs[3]
        batch_variance = self._outs[4]

        self._x_input = None
        for i in self._input_tensors:
            if i.op.name == "x_input":
                self._x_input = i

        shape_x = shape_to_list(self._x_input.shape)

        self._real_middle_out_tensors = set([save_mean_reduce, batch_variance])

        self._schedule = tvm.create_schedule(self._out.op)
        self._schedule.tiling_key = self._tiling_case["key"]

        if not isinstance(shape_x[2], int):
            h_range = self._tiling_case["h_range"]
            self._schedule.set_var_range(shape_x[2], h_range[0], h_range[1])

        if not isinstance(shape_x[3], int):
            w_range = self._tiling_case["w_range"]
            self._schedule.set_var_range(shape_x[3], w_range[0], w_range[1])

        self._calc_cache_read()
        self._do_cache_read()

        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._do_storage_bound()

        self._do_tiling_one_cut()

        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_compute_inline()
        self._do_compute_inline()

        self._do_double_buffer()

        self._calc_mem_reuse()
        self._do_mem_reuse()

        self._calc_constraints()
        self._do_constraints()

        self._calc_emit_insn()
        self._do_emit_insn()

        self._add_compile_info()

        return self._schedule

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
            self._dfs_sub_graph(out, visited_tensors)

        if len(self._outs) > 1:
            self._out, self._phony_tensor_list = _fake_node(self._outs)
            self._dfs_sub_graph(self._out, visited_tensors)
        else:
            self._out = self._outs[0]

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors

        for tensor_i in self._broadcast_tensors:
            if match_scalar_scene(tensor_i):
                self._absorbable_broadcast_tensors.add(tensor_i)

        ub_idx = 0
        if self._tiling_strategy == TilingStrategy.ONE_CUT:
            ub_idx = self._tiling_case["ub_tiling_axis"]
        for tensor_i in self._broadcast_tensors - self._absorbable_broadcast_tensors:
            if tensor_i.op.tag != "broadcast":
                src_shapes = tensor_i.op.input_tensors[0].shape[ub_idx:]
                dst_shapes = tensor_i.shape[ub_idx:]
                self._broadcast_axis_num[tensor_i] = _no_broadcast(src_shapes, dst_shapes)

        for tensor_i in self._in_out_map:
            if not isinstance(tensor_i.op, tvm.tensor.PlaceholderOp):
                self._mid_tensor_dst_tensor_map[tensor_i] = self._in_out_map[tensor_i]
            if tensor_i.op.tag.find("broadcast") != -1:
                self._broadcast_not_last_axis_tensors.append(tensor_i)

    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)
        self._cache_read_tensors.update(self._out_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._placeholder_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._out_tensors:
                self._middle_out_cache_read_buffer_map[tensor_i] = buffer_tensor

    def _calc_cache_write(self):
        cache_write_tensors = []
        for key in self._mid_tensor_dst_tensor_map:
            if "broadcast" not in key.op.name:
                cache_write_tensors.append(key)
        self._cache_write_tensors.update(list(cache_write_tensors))

    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[buffer_tensor] = tensor_i
            self._cache_write_tensor_map[tensor_i] = buffer_tensor

            if tensor_i in self._real_middle_out_tensors:
                self._middle_out_cache_write_buffer_map[tensor_i] = buffer_tensor

        self._fake_node_ub = self._schedule.cache_write(self._out, self._scope)

    def _set_scope(self):
        sch = self._schedule
        for tensor_i in self._pure_middle_tensors:
            sch[tensor_i].set_scope(self._scope)
        sch[self._out].set_scope(self._scope)

    def _do_storage_bound(self):
        sch = self._schedule
        tensors = self._pure_middle_tensors \
            .union(self._cache_read_buffer_tensor_map.keys()) \
            .union(self._cache_write_buffer_tensor_map.keys())
        tensors.add(self._fake_node_ub)

        for tensor_i in tensors:
            storage_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
            sch[tensor_i].set_storage_bound(storage_bound)

    def _do_tiling_one_cut(self):
        sch = self._schedule
        res = self._out
        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]
        core_num = util.get_core_num()

        ub_factor = operation.var("ub_factor", (1, None))

        reordered_axis_list = list(res.op.axis)

        b_o, b_i = sch[res].split(res.op.axis[b_idx], nparts=core_num)
        del reordered_axis_list[b_idx]
        reordered_axis_list = [b_o, b_i] + reordered_axis_list
        sch[res].reorder(*reordered_axis_list)

        if b_idx == u_idx:
            u_o, u_i = sch[res].split(b_i, factor=ub_factor)
        else:
            u_o, u_i = sch[res].split(res.op.axis[u_idx], factor=ub_factor)

        self._block_bind_axis = b_o
        self._compute_at_axis = b_o
        self._out_compute_at_axis = u_o
        self._emit_insn_axis = u_i
        self._block_inner_axis = b_i

    def _calc_compute_inline(self):
        for key in self._pure_middle_tensors:
            self._compute_inline_tensors.add(key)

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _calc_compute_at(self):
        x_input_shape = util.shape_to_list(self._x_input.shape)
        for tensor_i in self._middle_tensors - self._compute_inline_tensors:
            tensor_i_shape = util.shape_to_list(tensor_i.shape)
            if tensor_i_shape == x_input_shape:
                self._compute_at_map[tensor_i] = [self._out, self._out_compute_at_axis]
            else:
                self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_read_buffer_tensor_map:
            tensor_i_shape = util.shape_to_list(tensor_i.shape)
            if tensor_i_shape == x_input_shape:
                self._compute_at_map[tensor_i] = [self._out, self._out_compute_at_axis]
            else:
                self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]

        for tensor_i in self._cache_write_buffer_tensor_map:
            tensor_i_shape = util.shape_to_list(tensor_i.shape)
            if "broadcast" in tensor_i.op.name:
                pass
            if tensor_i_shape == x_input_shape:
                self._compute_at_map[tensor_i] = [self._out, self._out_compute_at_axis]
            else:
                self._compute_at_map[tensor_i] = [self._out, self._compute_at_axis]
        self._compute_at_map[self._fake_node_ub] = [self._out, self._out_compute_at_axis]

    def _do_compute_at(self):
        sch = self._schedule
        for tensor_i, param in self._compute_at_map.items():
            sch[tensor_i].compute_at(sch[param[0]], param[1])

    def _do_double_buffer(self):
        if self._is_db:
            sch = self._schedule
            tensor = self._placeholder_tensor_map[self._x_input]
            sch[tensor].double_buffer()

    def _calc_mem_reuse(self):
        ternary_reuse_map = {
            "elewise_binary_scalar_axpy": 1,
            "elewise_multiple_mla": 2,
            "elewise_multiple_madd": 1,
            "elewise_multiple_maddrelu": 1
        }

        def _get_ub_tensor(_input_tensor, _output_tensor):
            if _input_tensor in self._placeholder_tensor_map:
                _input_tensor = self._placeholder_tensor_map[_input_tensor]
            if _output_tensor in self._cache_write_tensor_map:
                _output_tensor = self._cache_write_tensor_map[_output_tensor]
            return _input_tensor, _output_tensor

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
                src_tensor, dst_tensor = _get_ub_tensor(src_tensor, tensor_i)
                util.merge_value(self._mem_reuse_map,
                                 src_tensor,
                                 dst_tensor)

        for tensor_i, write_buffer in \
                self._middle_out_cache_write_buffer_map.items():
            util.merge_value(self._mem_reuse_map, self._middle_out_cache_read_buffer_map[tensor_i], write_buffer)

    def _do_mem_reuse(self):
        sch = self._schedule
        for _a, _b in self._mem_reuse_map.items():
            for b_i in _b:
                sch[_a].reused_by(b_i)

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
            if target in self._out_tensors:
                pass
            else:
                self._emit_insn_map[source] = [source.op.axis[0], "dma_copy"]

        for source, target in self._cache_write_buffer_tensor_map.items():
            if target in self._phony_tensor_list:
                self._emit_insn_map[source] = [source.op.axis[0], "phony_insn"]
            else:
                self._emit_insn_map[source] = [source.op.axis[0], get_insn(target)]

    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            if len(param) > 2:
                sch[tensor_i].emit_insn(param[0], param[2])
            compile_broadcast_no_inline = (param[1] == "unified_broadcast" and \
                                           self._broadcast_axis_num.get(tensor_i, 0) > 1)
            if param[1] == "unknown_broadcast" or compile_broadcast_no_inline:
                u_idx = 0
                if self._tiling_strategy != TilingStrategy.NONE_CUT:
                    u_idx = self._tiling_case["ub_tiling_axis"]
                src_shapes = tensor_i.op.input_tensors[0].shape[u_idx:]
                tensor_bound = int(self._tensor_space // DTYPE_BYTE_MAPPING[tensor_i.dtype])
                src_shape = tvm.expr.Call('handle', 'tvm_tuple', src_shapes,
                                          tvm.expr.Call.PureIntrinsic, None, 0)
                attrs = {}
                if compile_broadcast_no_inline:
                    attrs = dict(storage_bound=[tensor_bound])
                if param[1] == "unknown_broadcast":
                    attrs = dict(src_shape=src_shape, storage_bound=[tensor_bound])
                sch[tensor_i].emit_insn(param[0], param[1], attrs)
            else:
                if self._is_db and tensor_i in self._out_tensors and self._is_pure_eletwise:
                    sch[tensor_i].emit_insn(param[0], param[1], attrs=dict(no_overlap=0))
                else:
                    sch[tensor_i].emit_insn(param[0], param[1])

        for source, target in self._cache_write_buffer_tensor_map.items():
            if target in self._out_tensors:
                sch[target].emit_insn(target.op.axis[0], "dma_copy")
        for source, target in self._middle_out_cache_read_buffer_map.items():
            sch[target].emit_insn(target.op.axis[0], "phony_insn")
        sch[self._out].emit_insn(self._emit_insn_axis, "phony_insn")
        sch[self._fake_node_ub].emit_insn(self._fake_node_ub.op.axis[0], "phony_insn")

    def _add_compile_info(self):
        cpt_compute = operation.get_context().get_current_compute()
        cpt_schedule = cpt_compute.get_current_schedule()
        cpt_schedule.add(CompileInfo.MAX_DTYPE, self._max_dtype_bytes)
        cpt_schedule.add(CompileInfo.COEXISTING_QUANTITY, self._tiling_case["coexisting_quantity"])
        cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
        cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())
        add_compile_info("max_ub_count", self._tensor_space // self._max_dtype_bytes)
        add_compile_info("block_dim", self._core_num)

    def _dfs_sub_graph(self, out, visited_tensors):
        for tensor_i in out.op.input_tensors:
            util.merge_value(self._in_out_map, tensor_i, out)
            if util.is_placeholder(tensor_i):
                self._input_tensors.add(tensor_i)
            else:
                self._middle_tensors.add(tensor_i)
                if util.is_broadcast(tensor_i):
                    self._broadcast_tensors.add(tensor_i)

            if tensor_i in visited_tensors:
                continue

            visited_tensors.add(tensor_i)

            self._dfs_sub_graph(tensor_i, visited_tensors)


def _fake_node(res):
    res_y = res[0]
    mean = res[1]
    variance = res[2]
    shape_x = util.shape_to_list(res_y.shape)
    phony_add_1 = tbe.dsl.vadd(mean, variance)
    phony_broadcast = tbe.dsl.broadcast(phony_add_1, shape_x)

    phony_cast = phony_broadcast
    phony_tensor_list = [phony_add_1, phony_broadcast]
    if res_y.dtype == "float16":
        phony_cast = tbe.dsl.cast_to(phony_broadcast, res_y.dtype)
        phony_tensor_list.append(phony_cast)
    phony_out = tbe.dsl.vadd(phony_cast, res_y)
    return phony_out, phony_tensor_list
