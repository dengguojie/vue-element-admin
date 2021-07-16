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
computation
"""
import abc
import itertools
from typing import Any

from tbe.common.platform import SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec
from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_tiling_case

from .constants import Pattern

DEFAULT = "default"


class Computation(abc.ABC):
    """
    computation interface
    """

    def __init_subclass__(cls, **kwargs):
        ComputationManager.add_computation(cls)

    @abc.abstractmethod
    def do_tiling_case(self):
        # type: () -> list[Any]
        """
        Implemented by sub-class, and return tiling case.
        :return:
        """

    @abc.abstractmethod
    def get_sub_pattern(self):
        # type: () -> str
        """
        Implemented by sub-class, and return sub-pattern.
        Usually used in performance optimization scenarios.
        :return:
        """

    @classmethod
    @abc.abstractmethod
    def get_instance(cls, outs, option):
        # type: (list[Any], dict[str, Any]) -> "Computation"
        """
        Implemented by sub-class, and return computation instance.
        :param outs:
        :param option:
        :return:
        """

    @classmethod
    @abc.abstractmethod
    def get_supported_pattern(cls):
        # type: () -> list[str]
        """
        Implemented by sub-class, and return supported pattern.
        :return:
        """

    @classmethod
    @abc.abstractmethod
    def get_supported_soc(cls):
        # type: () -> list[str]
        """
        Implemented by sub-class, and return supported socs.
        If supported all or as default schedule, can return default.
        :return:
        """


class ComputationManager:
    """
    computation manager
    """
    _computations = []  # type: list[Computation]

    @classmethod
    def add_computation(cls, schedule):
        """
        :param schedule:
        :return:
        """
        cls._computations.append(schedule)

    @classmethod
    def get_computation(cls, soc, pattern):
        # type: (str, str) -> Computation
        """
        :param soc:
        :param pattern:
        :return:
        """
        def _match_soc(sch, target_soc):
            # type: (Computation, str) -> bool
            socs = _listize(sch.get_supported_soc())
            return target_soc in socs

        def _match_pattern(sch, target_pattern):
            # type: (Computation, str) -> bool
            patterns = _listize(sch.get_supported_pattern())
            return target_pattern in patterns

        computations = filter(lambda sch: _match_soc(sch, soc), cls._computations)
        default_computations = filter(lambda sch: _match_soc(sch, DEFAULT), cls._computations)
        computations = itertools.chain(computations, default_computations)

        computations = filter(lambda sch: _match_pattern(sch, pattern), computations)

        return next(computations, None)


def _listize(_x):
    return list(_x) if isinstance(_x, (list, tuple)) else [_x]


@register_tiling_case(pattern=[Pattern.ELEMWISE, Pattern.BROADCAST, Pattern.REDUCE, Pattern.NORM])
def _tiling_case(outs, option=None):
    cpt = operation.get_context().get_current_compute()
    pattern = cpt.get_pattern()
    soc = get_soc_spec(SOC_VERSION)
    computation = ComputationManager.get_computation(soc, pattern)
    instance = computation.get_instance(outs, option)
    sub_pattern = instance.get_sub_pattern()
    cpt.set_sub_pattern(sub_pattern)

    return instance.do_tiling_case()
