#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
transdata base schedule
"""

from tbe import tvm
from tbe.common.platform import scope_ubuf
from tbe.common.utils.shape_util import shape_to_list
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from tbe.dsl.base.operation import get_context
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.schedule import Schedule

from .transdata_graph_info import ComputeGraphInfo
from .transdata_graph_info import get_reshape
from .transdata_graph_info import set_align
from .transdata_graph_info import ceil_div
from .constants import DEFAULT, FP32_ALIGN_SIZE
from .constants import BLOCK, INT8_BLOCK, FP16_BLOCK
from .constants import B8, B16


class TransdataBaseSch(Schedule):
    """
    Class for transdata base schedule
    """

    def __init__(self, outs, tiling_case):
        self.outs = outs
        self.tiling_case = tiling_case
        self.graph_info = get_context().get_current_compute().get("_compute_graph_info")

        self.schedule = None
        self.forward_compute_graph_map = self.graph_info.tensor_consumers_map
        self.forward_stage_graph_map = ComputeGraphInfo.set_map_deepcopy(self.forward_compute_graph_map)
        self.backward_compute_graph_map = self.graph_info.tensor_producers_map
        self.backward_stage_graph_map = ComputeGraphInfo.set_map_deepcopy(self.backward_compute_graph_map)
        self.cache_read_tensors_and_buffer_map = {}
        self.cache_write_tensors_and_buffer_map = {}
        self.double_buffer_tensors = []
        self.need_multi_core = True
        self.multi_core_bind_tensor = None
        self.multi_core_fused_axis = None
        self.compute_at_map = {}
        self.emit_insn_map = {}

        # Tensor
        self.depad_tensors = []
        self.pad_tensors = []
        self.transpose_tensors = []
        self.reshape_tensors = []
        self.tiling_tensor = None
        self.mte2_tensor = None
        self.remove_pad_tensor = None
        self.align_pad_tensor = None

        # Attr
        self.depad_axis_list = []
        self.pad_axis_list = []
        self.dtype = None
        self.split_once = None
        self.align_factor = None
        self.axis_in_ub = []
        self.axis_not_in_ub = []
        self.tiling_axes = []
        self.permutes = []
        self.pad_factor = self.graph_info.c0

        # AxisVar
        self.iter_block_outer = None
        self.iter_block_inner = None
        self.iter_ub_first_outer = None
        self.iter_ub_first_inner = None
        self.iter_ub_second_outer = None
        self.iter_ub_second_inner = None
        self.ub_outer = None
        self.ub_inner = None
        self.reorder_list = []

    @classmethod
    def get_instance(cls, outs, tiling_case):
        return cls(outs, tiling_case)

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.TRANSDATA]

    @staticmethod
    def parses_f_reshape(src_tensor):
        """
        :param src_tensor: f_reshape_tensor
        :return:
            1. Index of mapping between producer of src_tensor and src_tensor.
            2. Find indexes of x1|x0 in producer of src_tensor.
        """
        mapping_indexes, x1, x0 = [], None, None
        axes = get_reshape(src_tensor)
        for index, value in enumerate(axes):
            if isinstance(value, (list, tuple)) and len(value) == 2:
                mapping_indexes.extend([index, ] * 2)
                x1 = value[0]
                x0 = value[1]
            else:
                mapping_indexes.append(index)
        return mapping_indexes, x1, x0

    @staticmethod
    def optimization_vor(a, b, c0, tensor):
        """
        Func: avoid transpose(a,b,c0)->(b,a,c0) by storage-align,
        Return storage-align factor.
        """
        # While C0 is 16 and fp32, m*c0 is 2*m*block that 2*m is even.
        # While C0 is 16 and fp32, src must >= 16.
        bit = DTYPE_BYTE_MAPPING.get(tensor.dtype, 1)
        b *= c0 * bit // BLOCK
        pad_factor = c0
        c0 = BLOCK // bit

        if isinstance(a, int):
            src_cond = b >= a and b % 2 == 0
            dst_cond = a > b and a % 2 == 0
        else:
            src_cond = tvm.all(*[b >= a, b % 2 == 0])
            dst_cond = tvm.all(*[a > b, a % 2 == 0])

        src_tensor_var = tvm.select(src_cond, (b + 1) * c0, pad_factor)
        dst_tensor_var = tvm.select(dst_cond, a + 1, 1) * c0
        return src_tensor_var, dst_tensor_var

    @staticmethod
    def optimization_vnc(a, b, c0, tensor):
        """
        Func: avoid transpose (a,b)->(b,a) by storage-align, c0 is align_value
        Return: storage-align factor
        """
        bit = DTYPE_BYTE_MAPPING.get(tensor.dtype, 1)
        if bit not in [B8, B16]:
            dict_args = {"errCode": "E90003", "detailed_cause": "optimization_vnc is error."}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if isinstance(a, int) and isinstance(b, int):
            src_cond = b >= a and ceil_div(b, c0) % 2 == 0
            dst_cond = a > b and ceil_div(a, c0) % 2 == 0
        else:
            src_cond = tvm.all(*[b >= a, ceil_div(b, c0) % 2 == 0])
            dst_cond = tvm.all(*[a > b, ceil_div(a, c0) % 2 == 0])

        src_tensor_var = tvm.select(src_cond, (ceil_div(b, c0) + 1), 1) * c0
        dst_tensor_var = tvm.select(dst_cond, (ceil_div(a, c0) + 1), 1) * c0
        return src_tensor_var, dst_tensor_var

    def get_all_producers_stages(self, tensor):
        """
        get all produce stages for current tensor
        """
        producers = set()
        for producer in self.backward_stage_graph_map[tensor]:
            producers.add(producer)
            producers.update(self.get_all_producers_stages(producer))
        return producers

    def single_cache_read(self, source_tensor):
        """
        Create tensor after source-tensor
        """
        readers = self.forward_stage_graph_map[source_tensor]
        dst_tensor = self.schedule.cache_read(source_tensor, scope_ubuf, readers)
        self.cache_read_tensors_and_buffer_map[source_tensor] = dst_tensor
        for item in readers:
            self.update_stage(dst_tensor, item, True)
        return dst_tensor

    def single_cache_write(self, source_tensor):
        """
        Create tensor before source-tensor
        """
        writers = self.graph_info.tensor_producers_map[source_tensor]
        dst_tensor = self.schedule.cache_write(source_tensor, scope_ubuf)
        self.cache_write_tensors_and_buffer_map[source_tensor] = dst_tensor
        for item in writers:
            self.update_stage(dst_tensor, item, False)
        return dst_tensor

    def update_stage(self, source_tensor, dst_tensor, before):
        """
        update graph stage map by new tensor
        """
        if before:
            self.forward_stage_graph_map.setdefault(source_tensor, set())
            self.backward_stage_graph_map.setdefault(source_tensor, set())
            for producer in tuple(self.backward_stage_graph_map[dst_tensor]):
                self.forward_stage_graph_map[producer].remove(dst_tensor)
                self.forward_stage_graph_map[producer].add(source_tensor)
                self.backward_stage_graph_map[dst_tensor].remove(producer)
                self.backward_stage_graph_map[source_tensor].add(producer)
            self.forward_stage_graph_map[source_tensor].add(dst_tensor)
            self.backward_stage_graph_map[dst_tensor].add(source_tensor)
        else:
            self.forward_stage_graph_map.setdefault(source_tensor, set())
            self.backward_stage_graph_map.setdefault(source_tensor, set())
            for consumer in tuple(self.forward_stage_graph_map[dst_tensor]):
                self.forward_stage_graph_map[dst_tensor].discard(consumer)
                self.backward_stage_graph_map[consumer].discard(dst_tensor)
                self.backward_stage_graph_map[consumer].add(source_tensor)
                self.forward_stage_graph_map[source_tensor].add(consumer)
            self.forward_stage_graph_map[dst_tensor].add(source_tensor)
            self.backward_stage_graph_map[source_tensor].add(dst_tensor)

    def child(self, src_tensor):
        """
        :param src_tensor:
        :return: direct consumer of src_tensor
        """
        return list(self.forward_stage_graph_map.get(src_tensor, None))[0]

    def parent(self, src_tensor):
        """
        :param src_tensor:
        :return: direct producer of src_tensor
        """
        return list(self.backward_stage_graph_map.get(src_tensor, None))[0]

    def parses_axis_type(self, permute, ub_internal_c=False):
        """
        Func: Classified transpose_tensor's axis into UB Internal and UB External
        """
        split_i = self.tiling_case.ub_split_first_idx
        split_o = self.tiling_case.ub_split_second_idx
        length = len(permute)

        ub_internal_input = {permute.index(x) for x in range(permute[split_i], length, 1)}
        ub_internal_output = set(range(split_o, length, 1))
        self.axis_in_ub = ub_internal_output.union(ub_internal_input)
        self.axis_not_in_ub = set(range(length)).difference(self.axis_in_ub)
        self.axis_not_in_ub = self.axis_not_in_ub.union({split_i, split_o})

        # Regulation of split maybe make c1 in ub-external, we need c1 c0 in ub-internal
        if ub_internal_c:
            for c in self.graph_info.c1c0:
                if c not in self.axis_in_ub:
                    self.axis_in_ub.add(c)
                if c in self.axis_not_in_ub:
                    self.axis_not_in_ub.remove(c)

        # for reorder
        self.axis_in_ub = list(self.axis_in_ub)
        self.axis_not_in_ub = list(self.axis_not_in_ub)
        self.axis_in_ub.sort()
        self.axis_not_in_ub.sort()

    def parses_tiling_info(self, mapping_indexes, x1_index, x0_index, factor):
        """Update real tiling info based on tiling_tensor
        Func:
        1. Do extra split for tiling_tensor(split x as x1 and x0).
        2. Collect tiling_axes.
        3. Update axis_in_ub and axis_not_in_ub base on tiling_tensor.
        """
        split_b = self.tiling_case.block_split_idx
        split_i = self.tiling_case.ub_split_first_idx
        split_o = self.tiling_case.ub_split_second_idx
        is_x1_out_ub = x1_index in self.axis_not_in_ub
        is_x0_in_ub = x0_index in self.axis_in_ub
        is_split_x1 = x1_index in [split_i, split_o]
        is_do_extra_split_x = is_x1_out_ub and is_x0_in_ub and not is_split_x1

        # update factor while split x1:
        # 1. split x1 mean that do not extra split x as x1 and x0
        # 2. ub split firstly, if ub split x1, factor is factor * 16(float16)
        # 3. block split secondly, if block split x1 and ub not split x1, factor is factor * 16(float16)
        if split_i == x1_index:
            self.tiling_case.ub_first_factor *= factor
        if not self.split_once and split_o == x1_index:
            self.tiling_case.ub_second_factor *= factor
        if split_b == x1_index and split_i != x1_index and split_o != x1_index and not is_do_extra_split_x:
            self.tiling_case.block_factor *= factor

        if is_do_extra_split_x:
            # do extra split
            root_var = self.tiling_tensor.op.axis[mapping_indexes[x1_index]]
            x1_axis, x0_axis = self.schedule[self.tiling_tensor].split(root_var, factor=factor)
            for idx, value in enumerate(self.tiling_tensor.op.axis):
                if idx == mapping_indexes[x1_index]:
                    self.tiling_axes.extend([x1_axis, x0_axis])
                else:
                    self.tiling_axes.append(value)
        else:
            for value in self.tiling_tensor.op.axis:
                self.tiling_axes.append(value)

            # update axis_in_ub and axis_not_in_ub that make them based on tiling_tensor
            self.tiling_case.block_split_idx = mapping_indexes[split_b]
            self.tiling_case.ub_split_first_idx = mapping_indexes[split_i]
            self.tiling_case.ub_split_second_idx = mapping_indexes[split_o]
            if x1_index in self.axis_in_ub and x0_index in self.axis_in_ub:
                self.axis_in_ub.remove(x1_index)
            if x1_index in self.axis_not_in_ub and x0_index in self.axis_not_in_ub:
                self.axis_not_in_ub.remove(x0_index)
            self.axis_in_ub = [mapping_indexes[x] for x in self.axis_in_ub]
            self.axis_not_in_ub = [mapping_indexes[x] for x in self.axis_not_in_ub]

    def choose_transpose_insn(self, dtype, perm):
        """
        According to the perm to decide use which instruction
        """
        if not perm:
            insn = "phony_insn"
        elif perm[-1] != len(perm) - 1:
            insn = "vector_transpose"
        else:
            insn = "vector_or"
            if get_block_size(dtype) in [INT8_BLOCK, ]:
                insn = "dma_copy"

        emit_idx = 0
        if insn in ["vector_transpose", ]:
            if get_block_size(dtype) not in [FP16_BLOCK, INT8_BLOCK]:
                # SWAPS FP32: not support perm is [0,2,1], but [2,1]
                invariant = 0
                for k, v in enumerate(perm):
                    if k != v:
                        break
                    invariant += 1
                if invariant == 0:
                    emit_idx = 0
                else:
                    emit_idx = invariant - len(perm)
                    ori_perm = perm[emit_idx:].copy()
                    back = sorted(ori_perm)
                    perm = [back.index(i) for i in ori_perm]
        elif insn in ["vector_or", ]:
            emit_idx = -3
        return emit_idx, insn, perm

    def vnchwconv_insn_map(self, tensor, _insn, _iter, perm):
        # do-last-transpose-insn
        if _insn == "phony_insn":
            self.schedule[self.parent(tensor)].reused_by(tensor)
            self.emit_insn_map[tensor] = {"scope": _iter, "instruction": _insn}
        elif _insn == "vector_transpose":
            attr = tvm.expr.Call('handle', 'tvm_tuple', perm, tvm.expr.Call.PureIntrinsic, None, 0)
            attrs = dict(attrs={"src_in_dst_order": attr, "is_trans_align": 1})

            self.emit_insn_map[tensor] = {"scope": _iter, "instruction": "vector_transpose"}
            self.emit_insn_map[tensor].update(attrs)

    def common_align_insn_map(self, tensor, _insn, _iter):
        # work for remove_pad and align_pad
        attrs = dict(attrs={"enough_buffer": False, "last_axis": 1})
        self.emit_insn_map[tensor] = {"scope": _iter, "instruction": _insn}
        self.emit_insn_map[tensor].update(attrs)

    def do_vnchwconv_align(self, tensor, factor):
        """
        :param tensor: last-transpose-tensor
        :param factor: last dim align value
        Func: (m,n)->(n,m), only consider dst-tensor(n,m).
        While dtype belong to [B8,B16], factor should be [32,16].
        While dtype belong to [B32,B64],factor should be [128,64].
        """
        if get_block_size(tensor.dtype) not in [FP16_BLOCK, INT8_BLOCK]:
            factor = tvm.select(factor >= FP32_ALIGN_SIZE, factor, FP32_ALIGN_SIZE)

        # align m
        if len(tensor.shape) >= 2:
            self.schedule[tensor].storage_align(tensor.op.axis[-2], factor, 0)

        # align n
        shape = shape_to_list(tensor.shape)
        if len(shape) >= 3:
            factor = set_align(shape[-1], factor) * self.align_factor
            self.schedule[tensor].storage_align(tensor.op.axis[-3], factor, 0)

    def update_ub_perm(self, tensor, transpose_work=True):
        """
        Func: judge transpose work in ub.
        :param tensor: transpose_tensor.
        :param transpose_work: transpose maybe not work.
        :return: real permute.
        """
        external_ub_axis = 0
        for k in self.axis_not_in_ub:
            if k not in self.axis_in_ub:
                external_ub_axis += 1

        perm = [int(x) for x in list(tensor.op.attrs["permute"])]
        ori_permute = perm[external_ub_axis:]
        back = sorted(ori_permute.copy())
        return [back.index(i) for i in ori_permute] if transpose_work and back != ori_permute else []

    def _create_schedule(self):
        self.schedule = tvm.create_schedule([tensor.op for tensor in self.graph_info.output_tensor_set])

    def _do_cache_read(self):
        for tensor in self.graph_info.input_tensor_set:
            readers = self.graph_info.tensor_consumers_map[tensor]
            read_buffer = self.schedule.cache_read(tensor, scope_ubuf, readers)
            self.cache_read_tensors_and_buffer_map[tensor] = read_buffer
            for item in readers:
                self.update_stage(read_buffer, item, True)

    def _do_cache_write(self):
        for tensor in self.graph_info.output_tensor_set:
            writers = self.graph_info.tensor_producers_map[tensor]
            write_buffer = self.schedule.cache_write(tensor, scope_ubuf)
            self.cache_write_tensors_and_buffer_map[tensor] = write_buffer
            for item in writers:
                self.update_stage(write_buffer, item, False)

    def _do_set_scope(self):
        for tensor in self.graph_info.mid_tensor_set:
            if tensor not in self.graph_info.output_tensor_set:
                self.schedule[tensor].set_scope(scope_ubuf)

    def _do_ub_tiling(self):
        first_factor = self.tiling_case.ub_first_factor
        second_factor = self.tiling_case.ub_second_factor

        # first ub tiling
        first_axis_var = self.tiling_axes[self.tiling_case.ub_split_first_idx]
        self.iter_ub_first_outer, self.iter_ub_first_inner = \
            self.schedule[self.tiling_tensor].split(first_axis_var, factor=first_factor)

        # second ub tiling
        if not self.split_once:
            sec_axis_var = self.tiling_axes[self.tiling_case.ub_split_second_idx]
            self.iter_ub_second_outer, self.iter_ub_second_inner = \
                self.schedule[self.tiling_tensor].split(sec_axis_var, factor=second_factor)

    def _do_block_tiling(self):
        """
        block tiling only split axis which belong to outsiders of ub
        """
        case = self.tiling_case
        if case.block_split_idx == case.ub_split_first_idx:
            tiling_axis_var = self.iter_ub_first_outer
        elif not self.split_once and case.block_split_idx == case.ub_split_second_idx:
            tiling_axis_var = self.iter_ub_second_outer
        else:
            tiling_axis_var = self.tiling_axes[case.block_split_idx]

        self.iter_block_outer, self.iter_block_inner = \
            self.schedule[self.tiling_tensor].split(tiling_axis_var, factor=case.block_factor)

    def _calc_multi_core(self):
        if self.need_multi_core:
            idx = self.reorder_list.index(self.iter_block_outer)
            fused_list = self.reorder_list[:idx + 1]
            self.multi_core_fused_axis = self.schedule[self.tiling_tensor].fuse(*fused_list)
            self.multi_core_bind_tensor = self.tiling_tensor

    def _do_multi_core(self):
        if self.need_multi_core:
            res = self.multi_core_bind_tensor
            block = tvm.thread_axis("blockIdx.x")
            self.schedule[res].bind(self.multi_core_fused_axis, block)

    def _calc_compute_at(self):
        self.compute_at_map.clear()
        for tensor in self.get_all_producers_stages(self.tiling_tensor):
            if tensor not in self.graph_info.input_tensor_set:
                self.compute_at_map[tensor] = {"parent": self.schedule[self.tiling_tensor], "scope": self.ub_outer}

    def _do_compute_at(self):
        for stage in self.compute_at_map:
            parent_stage = self.compute_at_map[stage].get("parent", None)
            scope_iter_var = self.compute_at_map[stage].get("scope", None)
            self.schedule[stage].compute_at(parent_stage, scope_iter_var)

    def _do_emit_insn(self):
        for stage in self.emit_insn_map:
            scope_iter_var = self.emit_insn_map[stage].get("scope", None)
            instruction = self.emit_insn_map[stage].get("instruction", None)
            attrs = self.emit_insn_map[stage].get("attrs", None)
            self.schedule[stage].emit_insn(scope_iter_var, instruction, attrs=attrs)

    def _do_double_buffer(self):
        for _tensor in self.double_buffer_tensors:
            self.schedule[_tensor].double_buffer()
        operation.add_build_arg("double_buffer_non_reuse", True)

    def _do_reorder(self):
        """
        Regulation: [D,E,C,B,A] ~ [D,E,C.outer,C.inner,B,A.outer,A.inner]
        if [D,E,B] belong to ub_outer, reorder is [D,E,C.outer,B,A.outer,C.inner,A.inner]
        """
        split_i = self.tiling_case.ub_split_first_idx
        split_o = self.tiling_case.ub_split_second_idx
        split_b = self.tiling_case.block_split_idx

        for idx in self.axis_in_ub:
            if idx == split_i:
                self.reorder_list.append(self.iter_ub_first_inner)
            elif idx == split_o:
                self.reorder_list.append(self.iter_ub_second_inner)
            else:
                self.reorder_list.append(self.tiling_axes[idx])

        outside = []
        for idx in self.axis_not_in_ub:
            if idx == split_b:
                outside.extend([self.iter_block_outer, self.iter_block_inner])
            elif idx == split_i:
                outside.append(self.iter_ub_first_outer)
            elif idx == split_o:
                outside.append(self.iter_ub_second_outer)
            else:
                outside.append(self.tiling_axes[idx])

        self.ub_outer = outside[-1]
        self.ub_inner = self.reorder_list[0]
        self.reorder_list = outside + self.reorder_list
        self.schedule[self.tiling_tensor].reorder(*self.reorder_list)

    def _do_storage_bound(self):
        for stage in self.forward_stage_graph_map:
            ub_count = self.tiling_case.tensor_ub_size_list[self.tiling_case.shape_type]
            self.schedule[stage].set_buffer_size(ub_count)

    def _do_fragments(self):
        self.schedule[self.tiling_tensor].pragma(self.iter_block_inner, "local.UB_fragments_memory_size", 256)

    def _do_group(self, tensor, axes):
        group_id = tvm.make.Call("int32", "axis_group", [0, "overwrite"], tvm.expr.Call.Extern, None, 0)
        for i in axes:
            self.schedule[tensor].pragma(i, "axis_group", group_id)


def get_block_size(dtype):
    return BLOCK // DTYPE_BYTE_MAPPING.get(dtype, 1)
