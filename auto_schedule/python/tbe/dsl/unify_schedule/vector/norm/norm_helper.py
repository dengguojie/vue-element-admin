#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
norm helper
"""
import dataclasses
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

from tbe.dsl.padding.padding import Action
from tbe.dsl.padding.padding import ActionType
from tbe.dsl.padding.padding import ActionValueType
from tbe.tvm.expr import ConstExpr
from tbe.tvm.tensor import Tensor


def classify_actions(actions):
    """
    classify actions
    """
    ori_tensor_and_actions_map = {}
    for single_action in actions:
        cur_tensor = single_action.get_tensor()
        if cur_tensor in ori_tensor_and_actions_map:
            ori_tensor_and_actions_map.get(cur_tensor).append(single_action)
        else:
            ori_tensor_and_actions_map[cur_tensor] = [single_action]

    return ori_tensor_and_actions_map


@dataclasses.dataclass
class SubGraphAndAction:
    """
    sub graph and action
    """
    split_tensor: Tensor = None
    cur_tensor: Tensor = None
    action_type: ActionType = None
    target_tensors: list = dataclasses.field(default_factory=list)
    condition: Callable = None
    value: Union[ConstExpr, Callable] = None
    value_type: ActionValueType = None


class SubGraphsAndActionsMapping:
    """
    mapping of sub graphs and actions
    """
    def __init__(self, actions, split_tensor_and_sub_graph_map, end_tensor):
        self._actions: List[Action] = actions
        self._split_tensor_and_sub_graph_map: Dict[Tensor, Dict] = split_tensor_and_sub_graph_map
        self._end_tensor: Tensor = end_tensor
        self.mapping_list: List[SubGraphAndAction] = []
        self._process()

    def get_mapping(self):
        """
        get sub graphs and actions mapping
        """
        return self.mapping_list

    def _process(self):
        def _assign_sub_graph_and_action_obj(_sub_graph_and_action_obj, _action,
                                             _split_tensor, _cur_tensor, _target_tensors):
            _sub_graph_and_action_obj.split_tensor = _split_tensor
            _sub_graph_and_action_obj.cur_tensor = _cur_tensor
            _sub_graph_and_action_obj.action_type = _action.get_action_type()
            _sub_graph_and_action_obj.target_tensors = _target_tensors
            _sub_graph_and_action_obj.condition = _action.get_condition()
            _sub_graph_and_action_obj.value = _action.get_value()
            _sub_graph_and_action_obj.value_type = _action.get_value_type()

        def _recursive_func(_remaining_target_tensor):
            if not _remaining_target_tensor:
                return

            _first_target_tensor = _remaining_target_tensor[0]
            for _sub_graph_split_tensor, _sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                _sub_tensor_list = _sub_graph_map.get("sub_tensor_list")
                if cur_tensor in _sub_tensor_list and _first_target_tensor in _sub_tensor_list:
                    # if target tensor is one of split tensors,
                    # it should be in the sub graph that split tensor is iterself
                    if _first_target_tensor in self._split_tensor_and_sub_graph_map and \
                            _first_target_tensor != _sub_graph_split_tensor:
                        continue
                    _sub_graph_and_action_obj = SubGraphAndAction()
                    _assign_sub_graph_and_action_obj(_sub_graph_and_action_obj, single_action,
                                                     _sub_graph_split_tensor, cur_tensor, [])
                    for _single_target_tensor in set(_sub_tensor_list) & set(_remaining_target_tensor):
                        _sub_graph_and_action_obj.target_tensors.append(_single_target_tensor)
                        _remaining_target_tensor.remove(_single_target_tensor)
                    self.mapping_list.append(_sub_graph_and_action_obj)

                    _recursive_func(_remaining_target_tensor)

        def _no_target_tensor_func():
            for _sub_graph_split_tensor, _sub_graph_map in self._split_tensor_and_sub_graph_map.items():
                _sub_tensor_list = _sub_graph_map.get("sub_tensor_list")
                if cur_tensor in _sub_tensor_list:
                    # if cur tensor is one of split tensors,
                    # it should be in the sub graph that split tensor is not iterself(except end tensor)
                    if cur_tensor != self._end_tensor and cur_tensor in self._split_tensor_and_sub_graph_map and \
                            cur_tensor == _sub_graph_split_tensor:
                        continue
                    _sub_graph_and_action_obj = SubGraphAndAction()
                    _assign_sub_graph_and_action_obj(_sub_graph_and_action_obj, single_action,
                                                     _sub_graph_split_tensor, cur_tensor, target_tensors)
                    self.mapping_list.append(_sub_graph_and_action_obj)

        for single_action in self._actions:
            cur_tensor = single_action.get_tensor()
            target_tensors = single_action.get_target_tensors()
            if target_tensors:
                _recursive_func(target_tensors)
            else:
                _no_target_tensor_func()
