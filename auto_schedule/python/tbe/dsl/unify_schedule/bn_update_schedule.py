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

# block size in D architecture
BLOCK_SIZE_BYTE = 32

@register_schedule(pattern = Pattern.BN_UPDATE)
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

        self._dtypes = set()
        self._max_dtype_bytes = 4
        self._coexisting_quantity = 1
        self._tensor_space = None
        self._ub_size = util.get_ub_size()

        self._broadcast_not_last_axis_tensors = []
        self._mid_tensor_dst_tensor_map = {}
        self._out_compute_at_axis = None
        self._fake_node_ub = None
        self._block_dim = 1

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

        self._inner_shape = []

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

        is_update_v3 = False
        
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

        self._calc_cache_read()
        self._do_cache_read()
        
        self._calc_cache_write()
        self._do_cache_write()

        self._set_scope()

        self._calc_storage_bound()
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
            self._dtypes.add(out.dtype)
        byte_len = [DTYPE_BYTE_MAPPING[dtype] for dtype in self._dtypes]
        self._max_dtype_bytes = max(byte_len)

        self._pure_middle_tensors = self._middle_tensors - self._out_tensors
        self._middle_out_tensors = self._middle_tensors.intersection(self._out_tensors)
        pure_out_tensors = list(self._out_tensors - self._middle_out_tensors)
        if len(self._out_tensors) > 1:
            self._out = _fake_node(pure_out_tensors)
            self._dfs_sub_graph(self._out, visited_tensors)
        else:
            self._out = pure_out_tensors[0]
        
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

    def _calc_storage_bound(self):
        self._ub_size = util.get_ub_size()
        self._coexisting_quantity = 7

        tensor_space = self._ub_size // self._coexisting_quantity
        if self._is_db:
            tensor_space = tensor_space // 2
        self._tensor_space = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE

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
        shape = util.shape_to_list(res.shape)
        b_idx = self._tiling_case["block_tiling_axis"]
        u_idx = self._tiling_case["ub_tiling_axis"]
        core_num = util.get_core_num()

        block_factor = operation.var("block_factor", (1, None))
        ub_factor = operation.var("ub_factor", (1, None))

        b_o, b_i = sch[res].split(res.op.axis[b_idx], nparts=block_factor)
        
        if b_idx == u_idx:
            if u_idx == 1 and ub_factor > 1:
                u_o, u_i = sch[res].split(res.op.axis[2], factor=ub_factor)
            elif u_idx == 0:
                u_o, u_i = sch[res].split(res.op.axis[1], factor=ub_factor)
            else:
                u_o, u_i = sch[res].split(b_i, factor=ub_factor)
        else:
            u_o, u_i = sch[res].split(res.op.axis[u_idx], factor=ub_factor)
        
        for i in range(u_idx+1, len(res.op.axis)):
            self._inner_shape.append([shape[i], i])
        
        self._block_bind_axis = b_o
        self._compute_at_axis = b_o
        self._out_compute_at_axis = u_o
        self._emit_insn_axis = u_i
        self._block_inner_axis = b_i

    def _calc_compute_inline(self):
        for key in self._pure_middle_tensors:
            self._compute_inline_tensors.add(key)
        for tensor_i in self._broadcast_axis_num:
            if self._broadcast_axis_num[tensor_i] == 0:
                self._compute_inline_broadcast.add(tensor_i)
    
    def _do_compute_inline(self):
        sch = self._schedule
        for tensor_i in self._compute_inline_tensors:
            sch[tensor_i].compute_inline()
    
    def _do_multi_core(self):
        if self._block_bind_axis is not None:
            self._block_dim = 32
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
        for tensor_i in self._compute_inline_broadcast:
            input_tensor = tensor_i.op.input_tensors[0]
            input_tensor, _broadcast_tensor = _get_ub_tensor(input_tensor, tensor_i)
            util.merge_value(self._data_reuse_map, input_tensor, broadcast_tensor)
    
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
        
        for tensor_i in (self._pure_middle_tensors - self._compute_inline_tensors):
            if tensor_i in self._compute_inline_broadcast:
                self._emit_insn_map[tensor_i].append("phony_insn")

        INSN_MAP = {"vector_add": "vector_add_with_broadcast",
                    "vector_mul": "vector_mul_with_broadcast"}
        for source, target in self._cache_write_buffer_tensor_map.items():
            if "broadcast" in source.op.name:
                for i in self._in_out_map[target]:
                    self._emit_insn_map[source] = [source.op.axis[0], INSN_MAP[get_insn(i)]]
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
                tensor_bound = int(self._tensor_space//DTYPE_BYTE_MAPPING[tensor_i.dtype])
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
        cpt_schedule.add(CompileInfo.MAX_DTYPE, self._max_dtype_bytes)
        cpt_schedule.add(CompileInfo.COEXISTING_QUANTITY, self._coexisting_quantity)
        cpt_schedule.add(CompileInfo.UB_SIZE, self._ub_size)
        cpt_schedule.add(CompileInfo.CORE_NUM, util.get_core_num())
        add_compile_info("max_ub_count", int(self._tensor_space)//4)
        add_compile_info("block_dim", self._block_dim)
    
    def _dfs_sub_graph(self, out, visited_tensors: set):
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

            self._dfs_sub_graph(tensor_i, visited_tensors)

def _get_in_tensor_cnt(out_tensor):
    """get all input tensor count for current tensor
    Parameters
    ----------
    out_tensor : tensor
        need to count all its input tensors

    Return
    ------
        count value for out_tensor inpute tensors
    """
    if out_tensor is None:
        return 0
    stack = [out_tensor]
    visited_list = []
    in_count = 0
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
            in_count = in_count + 1
    return in_count

def _is_shape_contain_prime(shape):
    """
    check shape is contain prime that big than 5000
    :param shape:
    :return:
    """
    h_size = shape[2]
    w_size = shape[3]

    def _is_prime(num):
        for i in range(2, int(sqrt(num) + 1)):
            if num % i == 0:
                return False
        return True

    prime_threadhold = 5000
    return (h_size > prime_threadhold and _is_prime(h_size)) or \
            (w_size > prime_threadhold and _is_prime(w_size))

def _fake_node(tensors):
    dtype = tensors[0].dtype
    dim_length = max([len(t.shape) for t in tensors])
    shape = [1] * dim_length
    for tensor_i in tensors:
        if DTYPE_BYTE_MAPPING[tensor_i.dtype] > DTYPE_BYTE_MAPPING[dtype]:
            dtype = tensor_i.dtype
        shape_i = util.shape_to_list(tensor_i.shape)
        diff = dim_length - len(shape_i)
        shape_i = [1]*diff + shape_i
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

def _map_append(input_map, key, value):
    """
    map append
    """
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
    
def get_ub_tiling(shape, block_tiling_axis, block_tiling_inner_loop,
                  max_ub_count):
    """
    get ub tiling
    """
    last_axis = len(shape) - 1
    ub_split_inner = 1
    ub_split_axis = 0
    if block_tiling_axis < 0 or block_tiling_axis > last_axis:
        return ub_split_axis, ub_split_inner
    
    bound_size = max_ub_count
    split_axis = block_tiling_axis
    step = -1
    temp_size = 1
    need_split = False
    for i in range(last_axis, block_tiling_axis + step, step):
        temp_size = temp_size * shape[i]
        if temp_size >= bound_size:
            split_axis = i
            temp_size = temp_size / shape[i]
            need_split = True
            break
    
    split_size = 1
    if need_split:
        for i in range(1, shape[split_axis]+1, 1):
            if (temp_size * i ) == bound_size:
                split_size = i
                break
            if (temp_size * i) > bound_size:
                split_size = i - 1
                split_size = get_nearest_factor(shape[split_axis], split_size)
                break
    else:
        split_size = block_tiling_inner_loop

    if split_axis == block_tiling_axis and split_size > block_tiling_inner_loop:
        split_size = block_tiling_inner_loop
    
    ub_split_inner = split_size
    ub_split_axis = split_axis

    return ub_split_axis, ub_split_inner


