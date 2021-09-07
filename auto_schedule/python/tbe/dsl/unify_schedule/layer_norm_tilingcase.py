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
Reduce Schedule Remake stage 1 - Tilingcase
"""

# Standard Package
from typing import Any
from typing import Set
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Callable
from copy import deepcopy

from ..base import operation
from ..base.operation import add_compile_info
from ..base.operation import get_compile_info
from ..base.operation import get_context
from ..base.operation import register_build_pointcut
from ..base.operation import register_tiling_case
from tbe.common.utils import op_tiling
from tbe.common.utils import shape_util
from tbe.common.platform.platform_info import get_soc_spec
from tbe.tvm.expr import IntImm
from tbe.tvm.expr import Var
# Third-party Packages
from tbe.tvm.tensor import Tensor

from .constants import Pattern
from tbe.tvm.tensor import PlaceholderOp

from . import util
from .util import get_reduce_all_axes
from .util import get_reduce_axis_indexes
from .vector_tilingcase import TilingCaseBase
from ...common.utils.errormgr import get_error_message

num_tw = 2
num_thr = 3
CONST = "const"
STATIC = "static"
LAST_DIM_RANGE_NUM1 = 64
LAST_DIM_RANGE_NUM2 = 2000


class ComputeGraphInfo:
    """
    Operator Compute Graph Info collector and container
    """

    def __init__(self, output_tensors: Iterable[Tensor]):
        """
        Initialize containers and try to collect info
        """
        # Basic Info
        self.output_tensor_set: Optional[Set[Tensor]] = None
        self.tensor_consumers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_producers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_list: Optional[List[Tensor]] = None
        # Extra info initialized by hooks
        self.reduce_tensor_set: Set[Tensor] = set()
        self.broadcast_tensor_set: Set[Tensor] = set()
        self.elewise_tensor_set: Set[Tensor] = set()
        self.input_tensor_set: Set[Tensor] = set()
        self.non_gm_input_tensor_set: Set[Tensor] = set()
        # Extra info initialized after pre-initialization
        self.mid_output_tensor_set: Set[Tensor] = set()
        self.mid_tensor_set: Set[Tensor] = set()
        self.endpoint_output_tensor_set: Set[Tensor] = set()
        self.max_single_tensor_ub_size: Optional[int] = None
        # Do info collection
        self._collect_info(output_tensors)
        self._init_max_ub_count()

    def _collect_info(self, output_tensors: Iterable[Tensor]):
        self.output_tensor_set = set(output_tensors)
        self.tensor_list, self.tensor_consumers_map, self.tensor_producers_map = \
            self.dfs_compute_graph(self.output_tensor_set,
                                   (   # self.input_tensor_set hook
                                       (lambda _tensor: isinstance(_tensor.op, PlaceholderOp),
                                        lambda _tensor: self.input_tensor_set.add(
                                            _tensor),
                                        lambda _tensor: self.non_gm_input_tensor_set.add(
                                            _tensor)
                                        if not _tensor.op.input_tensors else None),
                                       # self.reduce_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("reduce") != -1,
                                        lambda _tensor: self.reduce_tensor_set.add(
                                            _tensor),
                                        lambda _tensor: None),
                                       # self.broadcast_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("broadcast") != -1,
                                        lambda _tensor: self.broadcast_tensor_set.add(
                                            _tensor),
                                        lambda _tensor: None),
                                       # self.elewise_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("elewise") != -1,
                                        lambda _tensor: self.elewise_tensor_set.add(
                                            _tensor),
                                        lambda _tensor: None)
                                   ))
        # Initialize non-hookable info
        self.gen_mid_tensor_sets()
        # endpoint_output_tensor_set
        self.gen_endpoint_output_tensor_set()

    def gen_endpoint_output_tensor_set(self):
        for output_tensor in self.output_tensor_set:
            if not self.tensor_consumers_map[output_tensor]:
                self.endpoint_output_tensor_set.add(output_tensor)

    def gen_mid_tensor_sets(self):
        # mid_output_tensor_set
        # mid_tensor_set
        for tensor in self.tensor_list:
            if tensor in self.output_tensor_set and self.tensor_consumers_map[tensor]:
                # Tensor in output and has consumers is middle_out_tensor
                self.mid_output_tensor_set.add(tensor)
                self.mid_tensor_set.add(tensor)
            elif tensor not in self.output_tensor_set | self.input_tensor_set | self.non_gm_input_tensor_set:
                self.mid_tensor_set.add(tensor)

    @staticmethod
    def dfs_compute_graph(root_tensor: Union[Iterable[Tensor], Tensor],
                          hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                             Callable[[Tensor], Any],
                                             Callable[[Tensor], Any]], ...]):
        def recursive_func(_root_tensor: Tensor,
                           _visited_list: Set[Tensor],
                           _tensor_consumers_map: Dict[Tensor, Union[Set[Tensor]]],
                           _tensor_producers_map: Dict[Tensor, Union[Set[Tensor]]],
                           _hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                               Callable[[Tensor], Any],
                                               Callable[[Tensor], Any]], ...]):
            _visited_list.add(_root_tensor)
            _tensor_producers_map.setdefault(_root_tensor, set())
            _tensor_consumers_map.setdefault(_root_tensor, set())
            for hook in hooks:
                if hook[0](_root_tensor):
                    hook[1](_root_tensor)
                else:
                    hook[2](_root_tensor)
            for in_tensor in _root_tensor.op.input_tensors:
                _tensor_consumers_map.setdefault(in_tensor, set())
                _tensor_consumers_map[in_tensor].add(_root_tensor)
                _tensor_producers_map[_root_tensor].add(in_tensor)
                recursive_func(in_tensor,
                               _visited_list,
                               _tensor_consumers_map,
                               _tensor_producers_map,
                               _hooks)
        visited_list = set()
        tensor_consumers_map = {}
        tensor_producers_map = {}
        if isinstance(root_tensor, (list, tuple, set)):
            for tensor in root_tensor:
                recursive_func(tensor, visited_list,
                               tensor_consumers_map,
                               tensor_producers_map,
                               hooks)
        elif isinstance(root_tensor, Tensor):
            recursive_func(root_tensor, visited_list,
                           tensor_consumers_map, tensor_producers_map,
                           hooks)
        else:
            raise RuntimeError("dfs_compute_graph() supports list, tuple, Tensor only. Received %s"
                               % str(type(root_tensor)))
        return list(visited_list), tensor_consumers_map, tensor_producers_map

    def _init_max_ub_count(self):
        soc_ub_size = get_soc_spec("UB_SIZE")
        soc_ub_size = soc_ub_size // 2
        total_width = 8
        reduce_tensor = list(self.reduce_tensor_set)[0]
        if reduce_tensor.dtype in ["float16"]:
            total_width = 4
        if not total_width:
            raise RuntimeError("Could not get compute width")
        max_bound = total_width * 128
        max_ub_count = int(soc_ub_size // max_bound * 128)

        self.max_single_tensor_ub_size = max_ub_count

    @staticmethod
    def set_map_deepcopy(_map: Dict[Tensor, Set[Tensor]]) -> Dict[Tensor, Set[Tensor]]:
        return dict((key, _map[key].copy()) for key in _map)


def _get_block_size(dtype):
    if dtype in ["float32", "fp32", "int32"]:
        block_size = 8
    elif dtype in ["bool", "int8", "uint8"]:
        block_size = 32
    elif dtype in ["float16", "fp16"]:
        block_size = 16
    elif dtype in ["int64"]:
        block_size = 4
    else:
        raise RuntimeError("[%s] is not support type" % dtype)
    return block_size


def apply_compile_info(reduce_info, graph_info, tiling_list, mode=None, input_format=None):
    # Common_Info
    atomic = 0
    for item in tiling_list:
        if item.is_atomic:
            atomic = 1
            break

    core_num = get_soc_spec("CORE_NUM")
    soc_version = get_soc_spec("SOC_VERSION")
    max_ub_size_normal_fp16 = 10 * 1024
    max_ub_size_normal_fp32 = 10 * 1024
    if soc_version in ("Ascend310",):
        max_ub_size_normal_fp16 = 8 * 1024
        max_ub_size_normal_fp32 = 8 * 1024

    keep_dims = 1
    min_block_size = _get_block_size(
        list(graph_info.input_tensor_set)[0].dtype)
    common_info = [core_num, keep_dims, min_block_size, atomic]

    pattern = _get_pattern_key(
        reduce_info.shape_before_reduce, reduce_info.reduce_axis_indexes)
    max_ub_count = graph_info.max_single_tensor_ub_size
    pattern_info = [pattern]
    ub_info = [max_ub_count]

    pre_compile_info = get_compile_info()
    if pre_compile_info and "common_info" not in pre_compile_info:
        info_map = {"common_info": common_info, "pattern_info": pattern_info,
                    "ub_info": ub_info, "reduce_axis": reduce_info.reduce_axis_indexes, "max_ub_size_normal_fp16": max_ub_size_normal_fp16, "max_ub_size_normal_fp32": max_ub_size_normal_fp32, "mode": mode}
        for key in info_map.keys():
            if key not in pre_compile_info.keys():
                add_compile_info(key, info_map.get(key))
            else:
                if key != "common_info":
                    key_info = pre_compile_info.get(key)
                    key_info += info_map.get(key)
                    add_compile_info(key, key_info)
    elif pre_compile_info:
        pass
    else:
        raise RuntimeError("pre_compile_info is Null")


def _calc_tiling_key(reduce_info, tiling, dim_ln_range):
    # tiling: single_case
    shape = reduce_info.shape_before_reduce
    reduce_axis_idx = reduce_info.reduce_axis_indexes
    block_split_axis = tiling.block_split_axis_index
    block_split_axis_1 = tiling.block_split_axis_index_1
    ub_split_axis_index_reduce = tiling.ub_split_axis_index_reduce
    ub_split_axis = tiling.ub_split_axis_index
    atomic = tiling.is_atomic
    is_normal = tiling.is_normal
    is_fuse_axis = tiling.is_fuse_axis
    shape_type, db = 0, 0

    tiling_key = _get_tiling_key(atomic, db, shape_type, block_split_axis, block_split_axis_1, ub_split_axis_index_reduce, ub_split_axis,
                                 shape, reduce_axis_idx, is_normal, is_fuse_axis)

    if dim_ln_range[0] == 1:
        range_key = 5
    elif dim_ln_range[-1] is not None and dim_ln_range[-1] <= LAST_DIM_RANGE_NUM1:
        range_key = 0
    elif dim_ln_range[-1] is not None and dim_ln_range[-1] <= LAST_DIM_RANGE_NUM2:
        range_key = 1
    else:
        range_key = 2
    tiling.tiling_key = tiling_key + range_key


def _get_tiling_key(atomic, db, shape_type, block_split_axis, block_split_axis_1, ub_split_axis_index_reduce, ub_split_axis, shape, reduce_idx_list, is_normal, is_fuse_axis):
    """
    :param atomic: "True": atomic_reduce, "False": normal_reduce.
    :param db: int number in [0,1]. "0": enable db, "1": close db.
    :param shape_type: int number in [0,99]. Diff numbers represent diff types of
           shapes. Example: "0": normal shape, "1": const shape, "2": special shape.
    :param block_split_axis: int number in [0,7] that represent index of split axis
    :param ub_split_axis: int number in [0,7] that represent index of split axis.
    :param shape: shape before reduce
    :param reduce_idx_list:

    :return: key(int32)
    """

    pattern = _get_pattern_key(shape, reduce_idx_list, block_split_axis,
                               ub_split_axis_index_reduce, ub_split_axis, is_normal, is_fuse_axis)
    pos = (db, shape_type, block_split_axis, ub_split_axis, pattern, block_split_axis_1)
    val = (10**9, 10**7, 10**6, 10**5, 10**4, 10**3)
    key = 0
    for item, value in enumerate(pos):
        key += value * val[item]
    return key


@register_tiling_case(pattern=Pattern.LayerNorm)
def calc_tiling_case(outs, options=None):
    [options].clear()
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    current_compute = get_context().get_current_compute()

    # construct information of graph
    compute_graph_info = ComputeGraphInfo(outs)
    layer_norm_info = LayerNormInfo(compute_graph_info)
    current_compute.add("compute_graph_info", compute_graph_info)
    current_compute.add("layer_norm_info", layer_norm_info)
    if not compute_graph_info.reduce_tensor_set:
        raise RuntimeError("Couldn't find reduce node for ReduceSchedule")

    tiling_case_list = []
    # get mode
    mode = operation.get_context().get_current_compute().get("mode")
    dim_ln_range = operation.get_context().get_current_compute().get("dim_ln_range")

    input_format = operation.get_context().get_current_compute().get("input_format")
    apply_compile_info(layer_norm_info, compute_graph_info, tiling_case_list, mode, input_format)

    # Normal reduce tiling cases
    if mode == CONST:
        tiling_case_list += _gen_tiling_case_const(layer_norm_info, compute_graph_info, input_format)
    elif input_format == "FRACTAL_NZ":
        tiling_case_list += _gen_tiling_case_NZ(layer_norm_info, input_format, dim_ln_range)
    else:
        tiling_case_list += _gen_tiling_case(layer_norm_info, input_format, dim_ln_range)
    # apply_compile_info(layer_norm_info, compute_graph_info, tiling_case_list)
    # calc_tiling_key
    for tiling_case in tiling_case_list:
        _calc_tiling_key(layer_norm_info, tiling_case, dim_ln_range)

    return tiling_case_list


def _find_idx_in_tensor_list(args: Union[Tuple, List]):
    if len(args) < 2:
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "Size of args should more than 2"
        raise RuntimeError(dict_args, get_error_message(dict_args))

    tensor_list = args[1].get("tensor_list")
    _before_reduce = operation.get_context().get("placeholder_before_reduce")
    _after_reduce = operation.get_context().get("placeholder_after_reduce")

    if not _before_reduce and not _after_reduce:
        return

    for tensors in tensor_list:
        for _idx, _tensor in enumerate(tensors):
            if _tensor in _before_reduce:
                operation.add_compile_info("idx_before_reduce", _idx)
                return
        for _idx, _tensor in enumerate(tensors):
            if _tensor in _after_reduce:
                operation.add_compile_info("idx_before_reduce", _idx)
                return

    dict_args = dict()
    dict_args["errCode"] = "E90003"
    dict_args["detailed_cause"] = "Can not find placeholder_op"
    raise RuntimeError(dict_args, get_error_message(dict_args))


@register_build_pointcut(pattern=Pattern.LayerNorm)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    _find_idx_in_tensor_list(args)
    func(*args, **kwargs)


class LayerNormTilingCase(TilingCaseBase):
    def __init__(self):
        self.is_atomic = False
        self.block_split_axis_index = None
        self.block_split_axis_index_1 = 0
        self.block_factor = None
        self.block_factor_1 = None
        self.ub_split_axis_index_reduce = None
        self.ub_fuse_factor = None
        self.ub_split_axis_index = None
        self.ub_factor = None
        self.multi_core: Optional[bool] = None
        self.tiling_key = None
        self.is_normal = True
        self.is_split_ub = True
        self.is_fuse_axis = False
        self.format = None
        self.reduce_axis_list = []

    def __repr__(self):
        segment0 = "ATOMIC" if self.is_atomic else "NORMAL"
        segment1 = "ENABLED" if self.multi_core else "DISABLED"
        return "%s REDUCE: (%d, %d) with multicore %s" % (segment0, self.block_split_axis_index,
                                                          self.ub_split_axis_index, segment1)

    def __hash__(self):
        return hash((self.is_atomic, self.block_split_axis_index, self.block_factor, self.ub_split_axis_index,
                     self.ub_factor, self.multi_core))

    def __eq__(self, other) -> bool:
        condition0 = other.self.is_atomic == self.is_atomic
        condition1 = other.block_split_axis == self.block_split_axis_index
        condition2 = other.block_factor == self.block_factor
        condition3 = other.ub_split_axis == self.ub_split_axis_index
        condition4 = other.ub_factor == self.ub_factor
        condition5 = other.multi_core == self.multi_core
        return (type(other) == type(self) and condition0 and condition1 and condition2 and condition3 and condition4
                and condition5)

    def __ne__(self, other) -> bool:
        condition0 = other.self.is_atomic != self.is_atomic
        condition1 = other.block_split_axis != self.block_split_axis_index
        condition2 = other.block_factor != self.block_factor
        condition3 = other.ub_split_axis != self.ub_split_axis_index
        condition4 = other.ub_factor != self.ub_factor
        condition5 = other.multi_core != self.multi_core
        return (type(other) != type(self) or condition0 or condition1 or condition2 or condition3 or condition4
                or condition5)


class LayerNormInfo:
    def __init__(self, compute_graph_info: ComputeGraphInfo):
        # Assume only one reduce node

        self.reduce_tensor: Tensor = tuple(
            compute_graph_info.reduce_tensor_set)[0]
        self.all_axes: List[Var] = get_reduce_all_axes(self.reduce_tensor)

        self.shape_before_reduce: List[Union[Var, IntImm]] = list(
            self.reduce_tensor.op.input_tensors[0].shape)
        self.reduce_axis_indexes: List[int] = get_reduce_axis_indexes(
            self.reduce_tensor)
        self.graph_info: ComputeGraphInfo = compute_graph_info

    @staticmethod
    def find_last_reduce_axis(shape, reduce_axis_indexes: Iterable[int]):
        # shape_before_reduce:(a1,a2,a3,...,r2,r1), the last axis must be reduce axis
        # find r1 position, r1 may contain continues axis
        r1_end_index = len(shape) - 1
        for i in range(r1_end_index, -1, -1):
            if i not in reduce_axis_indexes:
                r1_start_index = i + 1
                break
            if i == 0:
                r1_start_index = i

        return r1_start_index, r1_end_index

    @staticmethod
    def find_last_none_reduce_axis(shape_before_reduce: list,
                                   reduce_axis_index: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """
        :param shape_before_reduce
        :param reduce_axis_index
        :return the last axis or the last serials axises that are not in reduce_axis
        """
        #  shape_before_reduce:(a1,a2,a3,...,r2,r1), the last axis must be reduce axis
        # find a1 position, a1 may contain continues axis
        a1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i not in reduce_axis_index:
                a1_end_index = i
                break
        a1_start_index = a1_end_index
        if a1_end_index is None:
            return a1_start_index, a1_end_index
        a1_start_index = 0

        return a1_start_index, a1_end_index


def _gen_tiling_case_common(block_split_axis, input_format, shape_before_reduce, reduce_axis_index, tiling_case_list, dim_ln_range):
    # last dim is one case
    shape_list = shape_util.shape_to_list(shape_before_reduce)
    index = 0
    len_shape = len(shape_list)
    len_reduce = len(reduce_axis_index)
    if tuple(dim_ln_range) == (1, 1):
        index += 1
        for i in range(len_shape - 2, -1, -1):
            if shape_list[i] != 1:
                break
            index += 1
    # all reduce axis are one case
    new_reduce_axis_index = [] if index >= len_reduce else reduce_axis_index[:(len_reduce - index)]
    new_shape_before_reduce = shape_before_reduce[:(len_shape - index)]
    for wj in new_reduce_axis_index:
        # workspace tilingcase
        ub_split_axis = wj
        for k in range(len(new_reduce_axis_index)):
            ub_split_axis_index_reduce = k
            workspace_tiling_case = LayerNormTilingCase()
            workspace_tiling_case.block_split_axis_index = block_split_axis
            workspace_tiling_case.ub_split_axis_index = ub_split_axis
            workspace_tiling_case.ub_split_axis_index_reduce = ub_split_axis_index_reduce
            workspace_tiling_case.is_normal = False
            workspace_tiling_case.multi_core = True
            workspace_tiling_case.is_split_ub = False if ub_split_axis == block_split_axis else True
            workspace_tiling_case.format = input_format
            workspace_tiling_case.reduce_axis_list = reduce_axis_index
            tiling_case_list.append(workspace_tiling_case)
    for nj in range(block_split_axis, len(new_shape_before_reduce)):
        # Normal tilingcase
        n_ub_split_axis = nj
        if n_ub_split_axis in reduce_axis_index and n_ub_split_axis != block_split_axis:
            continue
        normal_tiling_case = LayerNormTilingCase()
        normal_tiling_case.block_split_axis_index = block_split_axis
        normal_tiling_case.ub_split_axis_index = n_ub_split_axis
        normal_tiling_case.ub_split_axis_index_reduce = 0
        normal_tiling_case.multi_core = True
        normal_tiling_case.is_normal = True
        normal_tiling_case.format = input_format
        normal_tiling_case.reduce_axis_list = reduce_axis_index
        if n_ub_split_axis == block_split_axis:
            normal_tiling_case.is_split_ub = False
            tiling_case_list.append(normal_tiling_case)
        else:
            normal_tiling_case.is_split_ub = True
            for ii in range(block_split_axis, block_split_axis + 2):
                fuse_normal_tiling_case = deepcopy(normal_tiling_case)
                fuse_normal_tiling_case.block_split_axis_index_1 = ii
                tiling_case_list.append(fuse_normal_tiling_case)


def _gen_tiling_case_NZ(info: LayerNormInfo, input_format, dim_ln_range):
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indexes
    tiling_case_list = []
    for i in range(0, len(shape_before_reduce)):
        # if i not in reduce_axis_index: NOT ALL REDUCE CASE
        # ELSE: ALL REDUCE
        if i in reduce_axis_index:
            continue
        block_split_axis = i

        _gen_tiling_case_common(block_split_axis, input_format, shape_before_reduce,
                                reduce_axis_index, tiling_case_list, dim_ln_range)
        break
    return tiling_case_list


def _gen_tiling_case(info: LayerNormInfo, input_format, dim_ln_range):
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indexes
    tiling_case_list = []
    shape_list = shape_util.shape_to_list(shape_before_reduce)
    # skip all dims are one in tilingcase
    len_shape = 0 if dim_ln_range == (1, 1) and set(shape_list[:-1]) == {1} else len(shape_before_reduce)
    for i in range(0, len_shape):
        # if i not in reduce_axis_index: NOT ALL REDUCE CASE
        # ELSE: ALL REDUCE
        block_split_axis = i

        _gen_tiling_case_common(block_split_axis, input_format, shape_before_reduce,
                                reduce_axis_index, tiling_case_list, dim_ln_range)
        break
    return tiling_case_list


def _gen_tiling_case_const(info: LayerNormInfo, compute_graph_info, input_format):
    res_tensor = tuple(compute_graph_info.endpoint_output_tensor_set)[0]
    res_shape = util.shape_to_list(res_tensor.shape)
    reduce_axis_index = info.reduce_axis_indexes
    tiling_case_list = []
    inputs = []
    tiling_format = {}
    for _input in compute_graph_info.input_tensor_set:
        input_shape = util.shape_to_list(_input.shape)
        input_name = _input.name
        if input_name == "x":
            inputs.insert(0, {"shape": input_shape, "dtype": _input.dtype})
            for i in range(len(input_shape)):
                tiling_format["dim_%s" % str(i)] = "int"
        else:
            inputs.append({"shape": input_shape, "dtype": _input.dtype})
    outputs = [{"shape": res_shape, "dtype": res_tensor.dtype}]
    run_info = op_tiling.do_op_tiling(get_context().get_op_type(), get_compile_info(), inputs, outputs)
    tiling_format_other = {
        "block_split_axis_index": "int",
        "block_split_axis_index_1": "int",
        "ub_split_axis_index": "int",
        "ub_split_axis_index_reduce": "int",
        "block_factor": "int",
        "block_factor_1": "int",
        "ub_factor": "int",
        "ub_fuse_factor": "int",
        "is_normal": "int"
    }
    tiling_format.update(tiling_format_other)
    tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
    tiling_case = LayerNormTilingCase()
    tiling_case.multi_core = True
    tiling_case.block_split_axis_index = tiling_data["block_split_axis_index"]
    tiling_case.block_split_axis_index_1 = tiling_data["block_split_axis_index_1"]
    tiling_case.ub_split_axis_index = tiling_data["ub_split_axis_index"]
    tiling_case.ub_split_axis_index_reduce = tiling_data["ub_split_axis_index_reduce"]
    tiling_case.block_factor = tiling_data["block_factor"]
    tiling_case.block_factor_1 = tiling_data["block_factor_1"]
    tiling_case.ub_factor = tiling_data["ub_factor"]
    tiling_case.ub_fuse_factor = tiling_data["ub_fuse_factor"]
    tiling_case.is_normal = True if tiling_data["is_normal"] else False
    tiling_case.is_split_ub = True if tiling_data["block_split_axis_index"] != tiling_data["ub_split_axis_index"] else False
    tiling_case.format = input_format
    tiling_case.reduce_axis_list = reduce_axis_index
    tiling_case_list.append(tiling_case)
    return tiling_case_list


def _get_pattern_key(_shape, _reduce_idx_list, block_split_axis=0, ub_split_axis_index_reduce=0, ub_split_axis=0, is_normal=True, is_fuse_axis=False):
    pattern_key = 0
    length = len(_shape)
    for i in range(length):
        if i in _reduce_idx_list:
            pattern_key += num_thr * num_tw**(length - i - 1)
        else:
            pattern_key += num_tw**(length - i - 1)
    pattern_key += block_split_axis * 100 + ub_split_axis * 10 + ub_split_axis_index_reduce
    if not is_normal:
        pattern_key *= num_tw
    if is_fuse_axis:
        return pattern_key
    else:
        return num_thr * pattern_key
