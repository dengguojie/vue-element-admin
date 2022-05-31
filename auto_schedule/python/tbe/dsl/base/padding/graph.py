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
Compute graph
"""

import os
from typing import Dict, Iterable, List

import tbe.dsl.base.padding.simulator as m_simulator
import tbe.dsl.base.padding.util as util
import tbe.dsl.base.padding.value as m_value
from tbe.tvm.tensor import Tensor


class Graph:
    def __init__(self, outs):
        # type: (Iterable[Tensor]) -> None
        self._outs = outs # type: Iterable[Tensor]
        self._nodes = [] # type: List[Node]
        self._producer_consumer = {} # type: Dict[Tensor, Tensor]
        self._tensor_node = {} # type: Dict[Tensor, Node]

        self._build()

    def __repr__(self) -> str:
        linesep, indent = os.linesep, "    "
        return f"{{{linesep}" \
               f"{indent}out tensors: {self._outs}, {linesep}" \
               f"{indent}nodes: {self._nodes}, {linesep}" \
               f"}}"

    def get_nodes(self):
        # type: () -> List[Node]
        return self._nodes

    def get_node(self, tensor):
        # type: (Tensor) -> Node
        return self._tensor_node.get(tensor)

    def is_out(self, node):
        # type: (Node) -> bool
        return node.get_tensor() in self._outs

    def get_consumer_tensors(self, producer):
        # type: (Tensor) -> List[Tensor]
        return self._producer_consumer.get(producer, [])

    def _add_producer_consumer(self, producer, consumer):
        # type: (Tensor, Tensor) -> None
        consumers = self._producer_consumer.setdefault(producer, set())
        consumers.add(consumer)

    def _build(self):
        def dfs(tensor):
            # type: (Tensor) -> None
            if tensor in visited:
                return
            visited.add(tensor)

            for tensor_i in tensor.op.input_tensors:
                self._add_producer_consumer(tensor_i, tensor)
                dfs(tensor_i)

            node = Node(self, tensor)
            self._nodes.append(node)
            self._tensor_node[tensor] = node

        visited = set()
        for out in self._outs:
            dfs(out)


class Node:
    def __init__(self, graph, tensor):
        # type: (Graph, Tensor) -> None
        self._graph = graph            # type: Graph
        self._tensor = tensor          # type: Tensor
        self._pvalue = None            # type: m_value.PaddingValue
        self._svalues = []             # type: List[m_value.SettingValue]
        self._simulator = None         # type: m_simulator.Simulator

        self._init()

    def __repr__(self) -> str:
        linesep, indent = os.linesep, "    "
        return f"{{{linesep}" \
               f"{indent}tensor: {self._tensor}, {linesep}" \
               f"{indent}padding value: {self._pvalue}, {linesep}" \
               f"{indent}setting values: {self._svalues}, {linesep}" \
               f"}}"

    def get_tensor(self):
        # type: () -> Tensor
        return self._tensor

    def get_dtype(self):
        # type: () -> str
        return self._tensor.dtype

    def handle_padding(self):
        # type: () -> None
        if util.is_placeholder(self._tensor):
            if util.exist_pad(self):
                self._pvalue = util.new_pvalue_0(self.get_dtype())
            else:
                self._pvalue = util.new_pvalue_tensor(self.get_dtype())
            return

        self._simulator.adjust_calc()

        # TODO: middle out
        if self._graph.is_out(self):
            if util.exist_pad(self) and not util.is_0_pvalue(self.get_pvalue()):
                svalue = m_value.SettingValue(m_value.SettingValueType.NORMAL, self.get_dtype())
                svalue.condition = util.get_normal_condition(self)
                svalue.value = util.new_np_num_0(self.get_dtype())
                self._svalues.append(svalue)

    def get_input_nodes(self):
        # type: () -> List[Node]
        tensors = self._tensor.op.input_tensors
        nodes = [self._graph.get_node(x) for x in tensors]
        return nodes

    def get_simulator(self):
        # type: () -> m_simulator.Simulator
        return self._simulator

    def get_shape(self):
        # type: () -> List
        return self._tensor.shape

    def set_pvalue(self, pvalue):
        # type: (m_value.PaddingValue) -> None
        self._pvalue = pvalue

    def get_pvalue(self):
        # type: () -> m_value.PaddingValue
        return self._pvalue

    def get_svalues(self):
        # type: () -> List[m_value.SettingValue]
        return self._svalues

    def add_svalue(self, svalue):
        # type: (m_value.SettingValue) -> None
        return self._svalues.append(svalue)

    def get_consumer_tensors(self):
        return self._graph.get_consumer_tensors(self._tensor)

    def _init(self):
        if not util.is_placeholder(self._tensor):
            self._simulator = m_simulator.get_simulator(self)
