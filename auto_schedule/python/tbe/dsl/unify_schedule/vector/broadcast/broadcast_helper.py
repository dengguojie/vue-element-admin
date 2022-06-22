#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
broadcast helper
"""
from typing import List
from typing import Dict
from typing import Set
from functools import reduce
from operator import mul

from tbe import tvm
from ... import util

BLOCK_SIZE_BYTE = 32


class GraphHelper:
    """
    GraphHelper: graph visit, node calc
    """

    @staticmethod
    def find_outermost_brc(root: tvm.tensor.Tensor) -> List[tvm.tensor.Tensor]:
        def dfs_graph(tensor, in_brc):
            for tensor_i in tensor.op.input_tensors:
                if len(tensor_i.op.input_tensors) == 0 or tensor_i in brc_nodes:
                    continue
                if util.is_broadcast(tensor_i):
                    if in_brc > 0 and per_layer_num[in_brc - 1] > 0:
                        brc_nodes.pop()
                        per_layer_num[in_brc - 1] -= 1
                    brc_nodes.append(tensor_i)
                    per_layer_num[in_brc] += 1
                    if len(per_layer_num) <= in_brc + 1:
                        per_layer_num.append(0)
                    dfs_graph(tensor_i, in_brc + 1)
                else:
                    dfs_graph(tensor_i, in_brc)

        brc_nodes = []
        per_layer_num = [0]
        in_brc = 0
        if util.is_broadcast(root):
            brc_nodes.append(root)
            per_layer_num[0] += 1
            if len(per_layer_num) <= in_brc + 1:
                per_layer_num.append(0)
            dfs_graph(root, in_brc + 1)
        else:
            dfs_graph(root, in_brc)
        return brc_nodes

    @staticmethod
    def brc_grouping(brc_nodes: List[tvm.tensor.Tensor], max_dtype_bytes: int) -> Dict[int, List[tvm.tensor.Tensor]]:
        brc_nodes_size_map = {}
        ele_in_block = BLOCK_SIZE_BYTE // max_dtype_bytes
        for tensor_i in brc_nodes:
            before_brc_shape = util.shape_to_list(tensor_i.op.input_tensors[0].shape)
            before_brc_shape_size = reduce(mul, before_brc_shape, 1)
            before_brc_shape_size_align = (before_brc_shape_size + ele_in_block - 1) // ele_in_block * ele_in_block
            if before_brc_shape_size_align in brc_nodes_size_map:
                brc_nodes_size_map.get(before_brc_shape_size_align).append(tensor_i)
            else:
                brc_nodes_size_map[before_brc_shape_size_align] = [tensor_i]
        return brc_nodes_size_map

    @staticmethod
    def update_groups_by_out(groups: Dict[int, List[tvm.tensor.Tensor]], max_dtype_bytes: int,
                             outs: List[tvm.tensor.Tensor]):
        ele_in_block = BLOCK_SIZE_BYTE // max_dtype_bytes
        for tensor_i in outs:
            shapes = util.shape_to_list(tensor_i.shape)
            shapes_size = reduce(mul, shapes, 1)
            shapes_size_align = (shapes_size + ele_in_block - 1) // ele_in_block * ele_in_block
            has_key = groups.get(shapes_size_align, [])
            if has_key:
                has_key.append(tensor_i)

    @staticmethod
    def get_in_out_map(root: tvm.tensor.Tensor) -> Dict[tvm.tensor.Tensor, Set[tvm.tensor.Tensor]]:
        def dfs_graph(tensor):
            for tensor_i in tensor.op.input_tensors:
                util.merge_value(in_out_map, tensor_i, tensor)
                if tensor_i in visited_tensors:
                    continue
                visited_tensors.add(tensor_i)
                dfs_graph(tensor_i)

        in_out_map = {}
        visited_tensors = set()
        dfs_graph(root)
        return in_out_map

    @staticmethod
    def max_live_node(root: tvm.tensor.Tensor, in_out_map: Dict[tvm.tensor.Tensor, Set[tvm.tensor.Tensor]]) -> int:
        def refresh_dependent(tensor):
            for tensor_i in tensor.op.input_tensors:
                if tensor_i not in dependent_map:
                    continue
                dependent_map.get(tensor_i).remove(tensor)
                if not dependent_map.get(tensor_i):
                    dependent_map.pop(tensor_i)

        def dfs_graph(tensor):
            if tensor in dependent_map:
                return len(dependent_map)
            need_space = []
            for tensor_i in tensor.op.input_tensors:
                need_space.append(dfs_graph(tensor_i))
            current_space = len(dependent_map) + 1
            need_space.append(current_space)
            refresh_dependent(tensor)
            if tensor not in dependent_map:
                dependent_map[tensor] = in_out_map[tensor].copy()
            return max(need_space)

        dependent_map = {}
        coexisting_quantities = []
        for tensor_in in root.op.input_tensors:
            coexisting_quantities.append(dfs_graph(tensor_in))
        coexisting_quantities.append(len(dependent_map) + 1)
        return max(coexisting_quantities)

    @staticmethod
    def get_all_nodes(root: tvm.tensor.Tensor):
        def dfs_graph(tensor):
            for tensor_i in tensor.op.input_tensors:
                if tensor_i in visited_tensors:
                    continue
                visited_tensors.add(tensor_i)
                all_nodes.add(tensor_i)
                dfs_graph(tensor_i)

        all_nodes = set()
        visited_tensors = set()
        all_nodes.add(root)
        dfs_graph(root)
        return all_nodes

    @staticmethod
    def only_last_brc(input_tensors):
        if not input_tensors:
            return False

        max_dim_length = max(len(_input.shape) for _input in input_tensors)
        input_shapes = []
        for _input in input_tensors:
            input_shape = util.shape_to_list(_input.shape)
            input_shapes.append([1] * (max_dim_length - len(input_shape)) + input_shape)

        input_shapes = list(map(list, zip(*input_shapes)))
        last_dim = max_dim_length - 1
        is_last_brc = any(input_shapes[last_dim][0] != s for s in input_shapes[last_dim])
        if not is_last_brc:
            return False
        for i in range(last_dim - 1, -1, -1):
            if any(input_shapes[i][0] != s for s in input_shapes[i]):
                return False
        return True
