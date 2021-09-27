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
schedule interface, manager, dispatcher
"""
import abc
import itertools
from typing import Any

from tbe.common.platform import SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec
from tbe.dsl.base import operation
from tbe.tvm.schedule import Schedule as TVM_Schedule

DEFAULT = "default"


class Schedule(abc.ABC):
    """
    schedule interface
    """

    def __init_subclass__(cls, **kwargs):
        ScheduleManager.add_schedule(cls)
        operation.add_schedule(cls.get_supported_pattern(), _schedule)

    @abc.abstractmethod
    def do_schedule(self):
        # type: () -> TVM_Schedule
        """
        Implemented by sub-class, and return tvm schedule.
        :return:
        """

    @classmethod
    @abc.abstractmethod
    def get_instance(cls, outs, tiling_case):
        # type: (list[Any], Any) -> "Schedule"
        """
        Implemented by sub-class, and return schedule instance.
        :param outs:
        :param tiling_case:
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
    def get_supported_sub_pattern(cls):
        # type: () -> list[str]
        """
        Implemented by sub-class, and return supported sub-pattern.
        Usually used in performance optimization scenarios.
        :return:
        """


class ScheduleManager:
    """
    schedule manager
    """
    _schedules = []  # type: list[Schedule]

    @classmethod
    def add_schedule(cls, schedule):
        """
        :param schedule:
        :return:
        """
        cls._schedules.append(schedule)

    @classmethod
    def get_schedule(cls, soc, pattern, sub_pattern):
        """
        :param soc:
        :param pattern:
        :param sub_pattern:
        :return:
        """
        def _match_soc(sch, target_soc):
            # type: (Schedule, str) -> bool
            socs = _listize(sch.get_supported_soc())
            return target_soc in socs

        def _match_pattern(sch, target_pattern):
            # type: (Schedule, str) -> bool
            patterns = _listize(sch.get_supported_pattern())
            return target_pattern in patterns

        def _match_sub_pattern(sch, target_sub_pattern):
            # type: (Schedule, str) -> bool
            sub_patterns = _listize(sch.get_supported_sub_pattern())
            return target_sub_pattern in sub_patterns

        schedules = filter(lambda sch: _match_soc(sch, soc), cls._schedules)
        default_schedules = filter(lambda sch: _match_soc(sch, DEFAULT), cls._schedules)
        schedules = itertools.chain(schedules, default_schedules)

        schedules = filter(lambda sch: _match_pattern(sch, pattern), schedules)
        schedules = filter(lambda sch: _match_sub_pattern(sch, sub_pattern), schedules)

        return next(schedules, None)


def _listize(_x):
    return list(_x) if isinstance(_x, (list, tuple)) else [_x]


def _schedule(outs, tiling_case):
    cpt = operation.get_context().get_current_compute()
    sub_pattern = cpt.get_sub_pattern()
    pattern = cpt.get_pattern()
    soc = get_soc_spec(SOC_VERSION)

    if isinstance(sub_pattern, (list, tuple)):
        sch_idx = cpt.get_current_schedule().get("_sch_idx")
        sch = ScheduleManager.get_schedule(soc, pattern, sub_pattern[sch_idx])
    else:
        sch = ScheduleManager.get_schedule(soc, pattern, sub_pattern)
    instance = sch.get_instance(outs, tiling_case)

    return instance.do_schedule()
