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
Information containers for compute graph of transdata
"""

# Standard Packages
import operator
from functools import reduce
from typing import Dict, List, Optional
from typing import Set, Tuple, Union
from typing import Any, Callable, Iterable

from tbe import tvm
from tbe.tvm.tensor import Tensor
from tbe.tvm.tensor import PlaceholderOp
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.common.utils.errormgr import get_error_message
from tbe.common.platform.platform_info import get_soc_spec
from tbe.dsl.unify_schedule.constants import DTYPE_BYTE_MAPPING
from tbe.dsl.unify_schedule.constants import TransdataCategory as TCategory
from .constants import DO_TRANSPOSE_PAD, DO_PAD
from .constants import COMMON_ALIGN_NEED_NODES
from .constants import UB_CATEGORY_BH, UB_CATEGORY_BN


class ComputeGraphInfo:
    """
    Operator Compute Graph Info collector and container
    """

    def __init__(self, output_tensors: Iterable[Tensor]):
        """
        Initialize containers and try to collect info
        """
        # Basic Info
        self.soc_ub_size = get_soc_spec("UB_SIZE")
        # real_output_tensor_set: set doesn't contain fake_node
        self.output_tensor_set: Optional[Set[Tensor]] = None
        self.real_output_tensor_set: Optional[Set[Tensor]] = None
        self.tensor_consumers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_producers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_list: Optional[List[Tensor]] = None
        # Extra info initialized by hooks
        self.input_tensor_set: Set[Tensor] = set()
        self.non_gm_input_tensor_set: Set[Tensor] = set()
        # Extra info initialized after pre-initialization
        self.mid_output_tensor_set: Set[Tensor] = set()
        self.mid_tensor_set: Set[Tensor] = set()
        self.endpoint_output_tensor_set: Set[Tensor] = set()
        # TransData Computation Tensors
        self.transpose_tensor_set: Set[Tensor] = set()
        self.pad_tensor_set: Set[Tensor] = set()
        self.de_pad_tensor_set: Set[Tensor] = set()
        self.f_reshape_tensor_set: Set[Tensor] = set()
        self.s_reshape_tensor_set: Set[Tensor] = set()
        self.coexisting_quantities: Optional[Dict] = None

        self.c1c0: List = list()
        self.x1x0: List = list()
        self.permute: List = list()
        self.c0 = None
        self.reshape = None
        self.category = None
        self.is_forward = False
        self.is_last_transpose = False
        self.transpose_2_tensor = None

        self._collect_info(output_tensors)
        self._collect_transdata_info()
        self._max_ub_count()

    @staticmethod
    def dfs_compute_graph(root_tensor: Union[Iterable[Tensor], Tensor],
                          hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                             Callable[[Tensor], Any],
                                             Callable[[Tensor], Any]], ...]):
        """
        compute graph using dfs algorithm
        """

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

    @staticmethod
    def update_tensor_ub_sizes(graph_info, tiling_case):
        node_list = graph_info.coexisting_quantities
        soc_ub_size = graph_info.soc_ub_size if not tiling_case.db else graph_info.soc_ub_size // 2
        soc_ub_size -= 1024

        def _calc_value(_sub_graph_nodes):
            value = 0
            for v in _sub_graph_nodes:
                value += DTYPE_BYTE_MAPPING[v]
            return value

        max_value = _calc_value(node_list[0])
        for item in node_list:
            item_value = _calc_value(item)
            if max_value < item_value:
                max_value = item_value

        if tiling_case.ub_category == 0:
            # sch maybe choose storage_align (shape_type is 0)
            tiling_case.tensor_ub_size_list.append(soc_ub_size // max_value // 128 * 128)
            # sch maybe choose common_align (shape_type is 1)
            common_align = COMMON_ALIGN_NEED_NODES * DTYPE_BYTE_MAPPING[node_list[0][0]]
            common_align = common_align if common_align > max_value else max_value
            tiling_case.tensor_ub_size_list.append(soc_ub_size // common_align // 128 * 128)
        elif tiling_case.ub_category in [UB_CATEGORY_BN, UB_CATEGORY_BH]:
            # sch only choose storage_align in bn\bh(shape_type is 0)
            tiling_case.tensor_ub_size_list.append(soc_ub_size // max_value // 128 * 128)
        else:
            dict_args = {"errCode": "E90001", "detailed_cause": "ub_category isn't 0 or 1"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    @staticmethod
    def set_map_deepcopy(_map: Dict[Tensor, Set[Tensor]]) -> Dict[Tensor, Set[Tensor]]:
        """
        deep copy tensor map
        """
        return {key: _map[key].copy() for key in _map}

    def gen_endpoint_output_tensor_set(self):
        """
        get pure endpoint output tensors without middle output tensors
        """
        for output_tensor in self.output_tensor_set:
            if not self.tensor_consumers_map.get(output_tensor, None):
                self.endpoint_output_tensor_set.add(output_tensor)

    def gen_mid_tensor_sets(self):
        """
        get  middle tensors(with middle output tensors) and middle output tensors
        """
        for tensor in self.tensor_list:
            if tensor in self.output_tensor_set and self.tensor_consumers_map.get(tensor, None):
                # Tensor in output and has consumers is middle_out_tensor
                self.mid_output_tensor_set.add(tensor)
                self.mid_tensor_set.add(tensor)
            elif tensor not in self.output_tensor_set | self.input_tensor_set | self.non_gm_input_tensor_set:
                self.mid_tensor_set.add(tensor)

    def get_all_producers_computes(self, tensor):
        """
        Get all parents for tensor.
        """
        parents = set()
        for parent in self.tensor_producers_map.get(tensor, None):
            parents.add(parent)
            parents.update(self.get_all_producers_computes(parent))
        return parents

    def _collect_info(self, output_tensors: Iterable[Tensor]):
        self.output_tensor_set = set(output_tensors)
        self.tensor_list, self.tensor_consumers_map, self.tensor_producers_map = \
            self.dfs_compute_graph(self.output_tensor_set,
                                   (  # self.input_tensor_set hook
                                       (lambda _tensor: isinstance(_tensor.op, PlaceholderOp),
                                        lambda _tensor: self.input_tensor_set.add(_tensor),
                                        lambda _tensor: self.non_gm_input_tensor_set.add(_tensor)
                                        if not _tensor.op.input_tensors else None),
                                       # self.transpose_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("transdata|transpose") != -1,
                                        lambda _tensor: self.transpose_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.pad_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("transdata|pad") != -1,
                                        lambda _tensor: self.pad_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.de_pad_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("transdata|depad") != -1,
                                        lambda _tensor: self.de_pad_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.f_reshape_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("transdata|f_reshape") != -1,
                                        lambda _tensor: self.f_reshape_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.s_reshape_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("transdata|s_reshape") != -1,
                                        lambda _tensor: self.s_reshape_tensor_set.add(_tensor),
                                        lambda _tensor: None)
                                   ))
        # Initialize non-hookable info
        self.gen_mid_tensor_sets()
        # endpoint_output_tensor_set
        self.gen_endpoint_output_tensor_set()

    def _collect_transdata_info(self):
        current_compute = get_context().get_current_compute()
        self.category = current_compute.get("_transdata_category")
        if self.category in [TCategory.GENERAL_BACKWARD, TCategory.GENERAL_FORWARD]:
            self._base_collect()
        elif self.category in [TCategory.BORROW_N_B8B16_BACKWARD, TCategory.BORROW_N_B8B16_FORWARD]:
            self._bn_collect()
        elif self.category in [TCategory.BORROW_H_B8B16_BACKWARD, TCategory.BORROW_H_B8B16_FORWARD]:
            self._bh_collect()

        # pure data-move c1c0 don't swaps that model is 1
        src_pad_mode = get_compile_info().get("_src_pad_mode")
        src_pad_var = get_compile_info().get("_src_pad_var")
        mode = DO_TRANSPOSE_PAD if DO_TRANSPOSE_PAD in src_pad_mode else DO_PAD
        self.c0 = src_pad_var[src_pad_mode.index(mode)]

    def _base_collect(self):
        self.is_forward = True if self.category == TCategory.GENERAL_FORWARD else False
        self.permute = [int(x) for x in list(list(self.transpose_tensor_set)[0].op.attrs["permute"])]
        self.is_last_transpose = self.permute[-1] != len(self.permute) - 1
        self.reshape = get_reshape(list(self.s_reshape_tensor_set)[0]) if self.is_forward else \
            get_reshape(list(self.f_reshape_tensor_set)[0])

        for j in self.reshape:
            if isinstance(j, (list, tuple)) and len(j) >= 2:
                self.c1c0 = [self.permute[x] for x in j] if self.is_forward else j

    def _bn_collect(self):
        # In backward, tiling would not split C (combined by c1 and c0), find C based on transpose_2_tensor
        # In forward, tiling would not split c1 and c0, find c1\c0 based on transpose_2_tensor
        self.is_forward = True if self.category == TCategory.BORROW_N_B8B16_FORWARD else False
        t_0, t_1, t_2 = self._bn_bh_distinguish_transpose()
        self.transpose_2_tensor = t_2

        if not self.is_forward:
            # f0->t2(N,H,C1*C0,16)->(N,H,C,16)->(N,16,H,C)
            f_reshape = list(self.tensor_consumers_map.get(t_1, None))[0]
            self.c1c0 = find_reshape_point(f_reshape, is_forward=False)
            for tensor in [t_2, ]:
                perm = [int(x) for x in tensor.op.attrs["permute"]]
                self.c1c0 = [perm.index(x) for x in self.c1c0]
        else:
            # s1->t2(N,H,C1,C0,16)->(N,C1,H,C0,16)->(N,16,C1,H,C0)
            s_reshape = list(self.tensor_producers_map.get(t_1, None))[0]
            self.c1c0 = find_reshape_point(s_reshape)
            for tensor in [t_1, t_2]:
                perm = [int(x) for x in tensor.op.attrs["permute"]]
                self.c1c0 = [perm.index(x) for x in self.c1c0]
        self.x1x0 = [0, 1]
        add_compile_info_inner("_bn_x1x0", self.x1x0)
        add_compile_info_inner("_bn_c1c0", self.c1c0)
        add_compile_info_inner("_bn_permute", [int(x) for x in t_2.op.attrs["permute"]])

    def _bh_collect(self):
        """
        t_2 is assumed tiling_tensor, real tiling_tensor is res.
        Forward:
            ... -> ... -> s_reshape(split c as c1 c0) -> t_1 -> t_2 -> f_reshape -> res.
            tiling would not split c1 and c0, find c1 c0 base on t_2.
            tiling would bot split h0, find h1 h0 base ont_2.
        Backward:
            ... -> ... -> t_0 -> t_1 -> f_reshape -> depad -> t_2 -> res
            tiling would not split C, find C based on t_2
            tiling would not split h0, find h1 h0 based on t_2
        """
        self.is_forward = True if self.category == TCategory.BORROW_H_B8B16_FORWARD else False
        t_0, t_1, t_2 = self._bn_bh_distinguish_transpose()
        self.transpose_2_tensor = t_2

        if self.is_forward:
            s_reshape = list(self.tensor_producers_map.get(t_1, None))[0]
            self.c1c0 = find_reshape_point(s_reshape)
            for tensor in [t_1, t_2]:
                perm = [int(x) for x in tensor.op.attrs["permute"]]
                self.c1c0 = [perm.index(x) for x in self.c1c0]

            f_reshape = list(self.tensor_consumers_map.get(t_2, None))[0]
            self.x1x0 = find_reshape_point(f_reshape)
        else:
            f_reshape = list(self.tensor_consumers_map.get(t_1, None))[0]
            self.c1c0 = find_reshape_point(f_reshape, is_forward=False)
            for tensor in [t_2, ]:
                perm = [int(x) for x in tensor.op.attrs["permute"]]
                self.c1c0 = [perm.index(x) for x in self.c1c0]
            for k, v in enumerate(perm):
                if k != v:
                    self.x1x0 = [k, k + 1]
                    break
        add_compile_info_inner("_bh_x1x0", self.x1x0)
        add_compile_info_inner("_bh_c1c0", self.c1c0)
        add_compile_info_inner("_bh_permute", [int(x) for x in t_2.op.attrs["permute"]])

    def _bn_bh_distinguish_transpose(self):
        # In bn\bh, transpose-mode is last + n-last + last
        t_0, t_1, t_2 = None, None, None
        for _tensor in self.transpose_tensor_set:
            perm = [int(x) for x in _tensor.op.attrs["permute"]]
            if perm[-1] == len(perm) - 1:
                t_1 = _tensor
                break

        producers = self.get_all_producers_computes(t_1)
        for _tensor in self.transpose_tensor_set:
            if _tensor in producers:
                t_0 = _tensor
                break

        t_2 = list(self.transpose_tensor_set.difference({t_0, t_1}))[0]
        return t_0, t_1, t_2

    def _max_ub_count(self):
        def _analysis_dependent(dependent):
            _require_nodes = []
            for item in dependent.keys():
                _require_nodes.append(item.dtype)
            return _require_nodes

        def _current_compute_need_space(_tensor, _require_nodes):
            def _transpose_space():
                if self.is_last_transpose:
                    if dtype in ["float32", "int32"]:
                        _require_nodes.extend([dtype] * 3)
                    else:
                        _require_nodes.append(dtype)
                else:
                    _require_nodes.append(dtype)

            dtype = _tensor.dtype
            if _tensor in self.transpose_tensor_set:
                _transpose_space()
            else:
                _require_nodes.append(dtype)

        def _refresh_dependent(_tensor):
            for _tensor_i in _tensor.op.input_tensors:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        def _r_coexisting(_tensor, _need_space):
            if _tensor in dependent_map:
                _need_space.append(_analysis_dependent(dependent_map))
                return

            for _tensor_i in _tensor.op.input_tensors:
                _r_coexisting(_tensor_i, _need_space)

            _curr_nodes = _analysis_dependent(dependent_map)
            _current_compute_need_space(_tensor, _curr_nodes)
            _need_space.append(_curr_nodes)

            _refresh_dependent(_tensor)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = self.tensor_consumers_map.get(_tensor, None).copy()

        coexisting_quantities, dependent_map = [], {}
        _out = list(self.output_tensor_set)[0]
        for tensor_i in _out.op.input_tensors:
            _r_coexisting(tensor_i, coexisting_quantities)
        # deal output is not fake node
        curr_nodes = _analysis_dependent(dependent_map)
        _current_compute_need_space(_out, curr_nodes)
        coexisting_quantities.append(curr_nodes)
        self.coexisting_quantities = coexisting_quantities


def get_reshape(tensor_):
    """
    Get reshape-info from reshape-tensor
    """
    axes = []
    if tensor_ is not None:
        for x in list(tensor_.op.attrs["axes"]):
            if isinstance(x, tvm.container.Array):
                axes.append([int(j) for j in list(x)])
            else:
                axes.append(int(x))
    return axes


def choose_transpose_insn(perm):
    """
    According to the perm to decide use which instruction
    """
    emit_idx = 0
    if not perm:
        insn = "phony_insn"
    elif perm[-1] != len(perm) - 1:
        insn = "vector_transpose"
    else:
        insn = "dma_copy"
    return emit_idx, insn


def find_reshape_point(tensor_, is_forward=True):
    """
    Record point of reshape based on tensor_, just like c to [c1,c0] eg:
    -- [0, 1, [2, 3],   4] -- axes
    -- [N, H, [C1, C0], W] -- src
    -- [N, H,    C,     W] -- dst
    """
    result = []
    axes = get_reshape(tensor_)
    for k, v in enumerate(axes):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            if is_forward:
                result.extend(v)
            else:
                result.append(k)
    return result


def ceil_div(dim, factor):
    # CeilDiv
    return tvm.floordiv(dim + factor - 1, factor)


def set_align(dim, factor):
    # Align dim by factor
    return ceil_div(dim, factor) * factor


def math_prod(iterable):
    # Prod for iter
    return reduce(operator.mul, iterable, 1)
