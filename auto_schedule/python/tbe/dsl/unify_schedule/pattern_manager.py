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
pattern parser interface, manager, mather
"""

import abc
from typing import Dict
from typing import List, Type, Union

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.tvm.tensor import Tensor


class PatternParser(abc.ABC):
    """
    pattern interface
    """

    def __init__(self, outs, compute_type_size_map, compute_type_tensor_map):
        # type: (Union[Tensor, List[Tensor]], Dict[ComputeType, int], Dict[ComputeType, List[Tensor]]) -> None
        self.outs = outs
        self.compute_type_size_map = compute_type_size_map
        self.compute_type_tensor_map = compute_type_tensor_map

    def __init_subclass__(cls):
        # type: () -> None
        PatternParserManager.add_parser_cls(cls)

    @abc.abstractmethod
    def match(self):
        # type: () -> bool
        """
        Implemented by subclass, and return whether to match compute graph.
        :return:
        """

    @abc.abstractmethod
    def get_pattern(self):
        # type: () -> str
        """
        Implemented by subclass, and return its pattern.
        :return:
        """


class PatternParserManager:
    """
    Manage pattern parsers, such as ElewisePatternParser, BroadcastPatternParser, and so on.
    """
    _parser_classes = []

    @classmethod
    def add_parser_cls(cls, parser_cls):
        # type: (Type[PatternParser]) -> None
        cls._parser_classes.append(parser_cls)

    @classmethod
    def get_parsers(cls,
                    outs: Union[Tensor, List[Tensor]],
                    compute_type_size_map: Dict[ComputeType, int],
                    compute_type_tensor_map: Dict[ComputeType, List[Tensor]]) -> List[PatternParser]:
        return [parse_cls(outs, compute_type_size_map, compute_type_tensor_map) for parse_cls in cls._parser_classes]


def parse(outs, compute_type_size_map, compute_type_tensor_map):
    # type: (Union[Tensor, List[Tensor]], Dict[ComputeType, int], Dict[ComputeType, List[Tensor]]) -> Union[str, None]
    for parser in PatternParserManager.get_parsers(outs, compute_type_size_map, compute_type_tensor_map):
        if parser.match():
            return parser.get_pattern()
    return None
