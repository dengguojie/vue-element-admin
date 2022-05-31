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
Calc padding value in D format
"""

import os
from enum import Enum, auto
from typing import Callable, Iterable, List, Union

import tbe.dsl.base.padding.graph as m_graph
import tbe.dsl.base.padding.util as util
import tbe.dsl.base.padding.value as m_value
from tbe.tvm.expr import ConstExpr
from tbe.tvm.tensor import Tensor


def calc_padding(outs):
    # type: (Union[Tensor, Iterable[Tensor]]) -> List[Action]
    return Padding(outs).calc()


class Padding:
    def __init__(self, outs):
        # type: (Union[Tensor, Iterable[Tensor]]) -> None
        self.graph = m_graph.Graph(outs if isinstance(outs, Iterable) else [outs]) # type: m_graph.Graph

    @classmethod
    def _handle_node_padding(cls, node):
        # type: (m_graph.Node) -> List[Action]
        tensor = node.get_tensor()
        actions = []
        for svalue in node.get_svalues():
            if svalue.type == m_value.SettingValueType.BROADCAST:
                action = Action(tensor, ActionValueType.TENSOR)
                action.set_condition(svalue.condition)
                action.set_value(svalue.value)
                action.add_target_tensor(svalue.target)
                actions.append(action)
            elif svalue.type == m_value.SettingValueType.NORMAL:
                action = Action(tensor, ActionValueType.SCALAR)
                action.set_condition(svalue.condition)
                action.set_value(util.np_num_to_tvm(svalue.value))
                action.add_target_tensor(svalue.target)
                actions.append(action)

        pvalue = node.get_pvalue()
        if (pvalue is None or len(pvalue.target) == 0) and len(actions) == 1:
            actions[0].clear_target_tensors()

        return actions

    def calc(self):
        # type: () -> List[Action]
        for node in self.graph.get_nodes():
            node.handle_padding()

        actions = []
        for node in self.graph.get_nodes():
            x_actions = self._handle_node_padding(node)
            actions.extend(x_actions)

        return actions


class ActionType(Enum):
    SET_VALUE = auto()
    CACHE_READ_AND_SET_VALUE = auto()


class ActionValueType(Enum):
    SCALAR = auto()
    TENSOR = auto()


class Action:
    def __init__(self, tensor, value_type):
        # type: (Tensor, ActionValueType) -> None
        self._value_type = value_type # type: ActionValueType
        self._tensor = tensor # type: Tensor
        self._condition = None # type: Callable
        self._value = None # type: Union[ConstExpr, Callable]
        self._target_tensors = [] # type: List[Tensor]

    def __repr__(self) -> str:
        linesep, indent = os.linesep, "    "
        return f"{{{linesep}" \
               f"{indent}action type: {self.get_action_type()}, {linesep}" \
               f"{indent}value type: {self._value_type}, {linesep}" \
               f"{indent}tensor: {self._tensor}, {linesep}" \
               f"{indent}condition: {self._condition}, {linesep}" \
               f"{indent}value: {self._value}, {linesep}" \
               f"{indent}target tensors: {self._target_tensors}{linesep}" \
               f"}}"

    def get_action_type(self):
        # type: () -> ActionType
        if len(self._target_tensors) == 0:
            return ActionType.SET_VALUE
        return ActionType.CACHE_READ_AND_SET_VALUE

    def get_value_type(self):
        return self._value_type

    def get_tensor(self):
        # type: () -> Tensor
        return self._tensor

    def get_condition(self):
        # type: () -> Callable
        return self._condition

    def set_condition(self, condition):
        # type: (Callable) -> None
        self._condition = condition

    def get_value(self):
        # type: () -> Union[ConstExpr, Callable]
        return self._value

    def set_value(self, value):
        # type: (Union[ConstExpr, Callable]) -> None
        self._value = value

    def get_target_tensors(self):
        # type: () -> List[Tensor]
        return self._target_tensors

    def add_target_tensor(self, tensor):
        # type: (Tensor) -> None
        if tensor is None:
            return
        if isinstance(tensor, Tensor):
            self._target_tensors.append(tensor)
        elif isinstance(tensor, Iterable):
            self._target_tensors.extend(tensor)

    def clear_target_tensors(self):
        # type: () -> None
        self._target_tensors.clear()
