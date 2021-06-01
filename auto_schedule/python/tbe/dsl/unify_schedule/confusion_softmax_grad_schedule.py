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
CONFUSION SOFTMAX GRAD   
"""

from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import register_schedule
from tbe.dsl.base.operation import var
from tbe import tvm

from . import util
from .constants import DTYPE_BYTE_MAPPING, INSN_MAPPING, SUPPORT_SCALAR_INSNS, TERNARY_INSNS
from .confusion_softmax_grad_tilingcase import TilingStrategy

from tbe.common.utils import shape_to_list
from tbe.common.platform.platform_info import get_soc_spec

@register_schedule("ConfusionSoftmaxGrad")
def schedule(outs, tiling_case):

    confusion_softmax_grad_sch: ConfusionSoftmaxGradSchedule = ConfusionSoftmaxGradSchedule(outs, tiling_case)
    real_schedule = confusion_softmax_grad_sch.do_schedule()

    return real_schedule

class ConfusionSoftmaxGradSchedule:

    def __init__(self, outs, tiling_case):
        self._out = None
        self._outs = outs
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.get("tiling_strategy")

        self._scope = "local.UB"

        self._ir_axes = list()
        self._inner_shape = list()

        self._out_tensors = set()
        self._broadcast_tensors = set()
        self._absorbable_broadcast_tensors = set()
        self._pure_middle_tensors = set()
        self._middle_out_tensors = set()
        self._cache_read_tensors = set()
        self._middle_tensors = set()
        self._input_tensors = set()
        self._reduce_tensors = set()
        self._cache_write_tensors = set()
        self._compute_inline_broadcast = set()

        self._in_out_map = dict()
        self._cache_read_buffer_tensor_map = dict()
        self._placeholder_tensor_map = dict()
        self._block_tiling_vars = dict()
        self._ub_tiling_vars = dict()
        self._cache_write_buffer_tensor_map = dict()
        self._cache_write_tensor_map = dict()
        self._compute_at_map = dict()
        self._emit_insn_map = dict()
        self._compute_at_map = dict()

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._emit_insn_axis = None
        self._block_bind_axis = None
        self._compute_at_axis = None

    def do_schedule(self):
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

        self._calc_emit_insn()
        self._do_emit_insn()

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
        self._middle_out_tensors = self._middle_tensors.intersection(self._out_tensors)

        pure_out_tensors = list(self._out_tensors - self._middle_out_tensors)
        self._out = pure_out_tensors[0]

        for tensor_i in self._broadcast_tensors:
            if match_scalar_scene(tensor_i):
                self._absorbable_broadcast_tensors.add(tensor_i)
    
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
    
    def _calc_cache_read(self):
        self._cache_read_tensors.update(self._input_tensors)

    def _do_cache_read(self):
        for tensor_i in self._cache_read_tensors:
            buffer_tensor = self._schedule.cache_read(
                tensor_i, self._scope, self._in_out_map[tensor_i])
            self._cache_read_buffer_tensor_map[buffer_tensor] = tensor_i
            self._placeholder_tensor_map[tensor_i] = buffer_tensor

    def _calc_cache_write(self):
        self._cache_write_tensors.update(self._out_tensors)
        self._cache_write_tensors.update(self._pure_middle_tensors)
    
    def _do_cache_write(self):
        for tensor_i in self._cache_write_tensors:
            buffer_tensor = self._schedule.cache_write(tensor_i, self._scope)
            self._cache_write_buffer_tensor_map[buffer_tensor] = tensor_i
            self._cache_write_tensor_map[tensor_i] = buffer_tensor

    def _set_scope(self):
        for tensor_i in self._pure_middle_tensors:
            self._schedule[tensor_i].set_scope(self._scope)

    def _calc_tiling(self):
        funcs = {
            TilingStrategy.NONE_CUT: self._calc_tiling_none_cut,
            TilingStrategy.ONE_CUT: self._calc_tiling_one_cut
        }
        funcs[self._tiling_strategy]()
    
    def _calc_tiling_none_cut(self):
        pass
    
    def _calc_tiling_one_cut(self):
        res = self._out
        shape = shape_to_list(res.shape)
        b_i = self._tiling_case["block_tiling_axis"]
        u_i = self._tiling_case["ub_tiling_axis"]
        b_bound = (1, None)
        u_bound = (1, None)
        self._block_tiling_vars[b_i] = var("block_factor", b_bound)
        self._ub_tiling_vars[u_i] = var("ub_factor", u_bound)
    
    def _do_tiling(self):
        func = {
            TilingStrategy.NONE_CUT: self._do_tiling_none_cut,
            TilingStrategy.ONE_CUT: self._do_tiling_one_cut
        }
        func[self._tiling_strategy]()
    
    def _do_tiling_none_cut(self):
        res = self._out
        shape = shape_to_list(res.shape)
        for _i, _x in enumerate(res.op.axis):
            self._ir_axes.append([_x, _i])
            self._inner_shape.append([shape[_i], _i])
        self._emit_insn_axis = res.op.axis[0]
    
    def _do_tiling_one_cut(self):
        sch = self._schedule
        res = self._out
        shape = shape_to_list(res.shape)
        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]
        block_axes = []
        ub_axes = []
        inner_axes = []
        core_num = get_soc_spec("CORE_NUM")

        for i in range(b_idx):
            block_axes.append([res.op.axis[i], i])
        b_o, b_i = sch[res].split(res.op.axis[b_idx],
                                  nparts=core_num)
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
    
    def _calc_multi_core(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            self._block_bind_axis = None

    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[self._out].bind(self._block_bind_axis, block)

    def _calc_compute_inline(self):
        self._compute_inline_tensors = self._pure_middle_tensors

    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()
    
    def _calc_compute_at(self):
        if self._tiling_strategy == TilingStrategy.NONE_CUT:
            self._compute_at_map.clear()
            return
        
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
            return INSN_MAPPING.get(insn)
        
        for source, target in self._cache_read_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], "dma_copy"]
        
        for source, target in self._cache_write_buffer_tensor_map.items():
            self._emit_insn_map[source] = [source.op.axis[0], get_insn(target)]
            if target in self._compute_inline_broadcast:
                self._emit_insn_map[source].append("phony_insn")

        self._emit_insn_map[self._out] = [self._emit_insn_axis, "dma_copy"]
    
    def _do_emit_insn(self):
        sch = self._schedule
        for tensor_i, param in self._emit_insn_map.items():
            if tensor_i.op.tag == "unknown_broadcast":
                if self._tiling_strategy == TilingStrategy.NONE_CUT:
                    u_idx = 0
                else:
                    u_idx = self._tiling_case["ub_tiling_axis"]
                src_shapes = tensor_i.op.input_tensors[0].shape[u_idx:]
                src_shape = tvm.expr.Call('handle', 'tvm_tuple', src_shapes,
                                          tvm.expr.Call.PureIntrinsic, None, 0)
                sch[tensor_i].emit_insn(tensor_i.op.axis[0], "unknown_broadcast",
                                        attrs=dict(src_shape=src_shape))
                continue
            if tensor_i.op.tag == "unified_broadcast":
                sch[tensor_i].emit_insn(tensor_i.op.axis[1], "vector_broadcast")
                continue
            if len(param) > 2:
                sch[tensor_i].emit_insn(param[0], param[2])
            else:
                sch[tensor_i].emit_insn(param[0], param[1])
    
    def _calc_storage_bound(self):
        pass
    
    def _do_storage_bound(self):
        sch = self._schedule
        tensors = set(self._cache_read_buffer_tensor_map.keys()) \
           .union(set(self._cache_write_buffer_tensor_map.keys()))
        
        storage_bound = int(1024*15)
        for tensor_i in tensors:
            sch[tensor_i].set_storage_bound(storage_bound)
