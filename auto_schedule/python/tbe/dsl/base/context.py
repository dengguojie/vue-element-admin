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
context
"""
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from tbe.common.utils.errormgr import get_error_message

from . import operation
from .var import AttrVarDesc
from .var import Var


class OperatorContext:
    """
    OperatorContext
    """

    def __init__(self):
        self._pattern = None  # type: Optional[str]

        self._vars = []  # type: List[Var]
        self._attr_vars_desc = []  # type: List[AttrVarDesc]
        self._computes = []  # type: List[ComputeContext]
        self._current_compute = None  # type: Optional[ComputeContext]

        self._addition = {"compile_info": {}, "build_args": {}}  # type: Dict[str, Any]
        self._exclude_bound_vars = []  # type: List[Var]

    def set_pattern(self, pattern):
        # type: (str) -> None
        """
        :param pattern:
        :return:
        """
        self._pattern = pattern

    def get_pattern(self):
        # type: () -> Optional[str]
        """
        :return:
        """
        # TODO how to support custom pattern
        return self._pattern

    @staticmethod
    def get_mode():
        # type: () -> str
        """
        :return:
        """
        return operation.get_op_context().get_op_mode()

    @staticmethod
    def get_op_type():
        # type: () -> Optional[str]
        """
        :return:
        """
        op_infos = operation.get_op_context().get_op_info()
        if len(op_infos) == 1:
            return op_infos[0].op_type
        elif len(op_infos) >= 2:
            _name = "fusion"
            for value in op_infos:
                _name += "_" + str(value.op_type)
            return _name

        return None

    def add_var(self, var_):
        # type: (Var) -> None
        """
        :param var_:
        :return:
        """
        if self._computes:
            self._computes[-1].add_var(var_)
        else:
            self._vars.append(var_)

    def get_var(self, name):
        # type: (str) -> Optional[Var]
        """
        :param name:
        :return:
        """
        c_cmp = self.get_current_compute()
        if c_cmp is not None:
            var0 = c_cmp.get_var(name)
            if var0 is not None:
                return var0
        for var_i in self._vars:
            if var_i.get_name() == name:
                return var_i
        return None

    def get_vars(self):
        # type: () -> List[Var]
        """
        :return:
        """
        return self._vars

    def add_attr_var_desc(self, attr_var_desc):
        # type: (AttrVarDesc) -> None
        """
        :param attr_var_desc:
        :return:
        """
        if self._computes:
            self._computes[-1].add_attr_var_desc(attr_var_desc)
        else:
            self._attr_vars_desc.append(attr_var_desc)

    def get_attr_vars_desc(self):
        # type: () -> List[AttrVarDesc]
        """
        :return:
        """
        return self._attr_vars_desc

    def begin_compute(self, _compute):
        # type: (ComputeContext) -> None
        """
        :param _compute:
        :return:
        """
        if self._current_compute is not None:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Exist not finished compute context."
            raise RuntimeError(dict_args, get_error_message(dict_args))

        self._computes.append(_compute)
        self._current_compute = _compute

    def end_compute(self, _compute):
        # type: (ComputeContext) -> None
        """
        :param _compute:
        :return:
        """
        if self._current_compute != _compute:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Compute context not match."
            raise RuntimeError(dict_args, get_error_message(dict_args))
        self._current_compute = None

    def get_computes(self):
        # type: () -> List[ComputeContext]
        """
        :return:
        """
        return self._computes

    def get_current_compute(self):
        # type: () -> Optional["ComputeContext"]
        """
        :return:
        """
        return self._computes[-1] if self._computes else None

    def add(self, key, value):
        # type: (str, Any) -> None
        """
        :param key:
        :param value:
        :return:
        """
        self._addition[key] = value

    def get(self, key):
        # type: (str) -> Any
        """
        :param key:
        :return:
        """
        return self._addition.get(key)

    def set_default(self, key, value):
        # type: (str, Any) -> Any
        """
        :param key:
        :param value:
        :return:
        """
        if key not in self._addition:
            self.add(key, value)
        return self.get(key)

    def compute_default(self, key, func):
        # type: (str, Callable) -> Any
        """
        :param key:
        :param func:
        :return:
        """
        if key not in self._addition:
            self.add(key, func())
        return self.get(key)

    def add_exclude_bound_var(self, var_):
        # type: (Var) -> None
        """
        :param var_:
        :return:
        """
        if self._computes:
            self._computes[-1].add_exclude_bound_var(var_)
        else:
            self._exclude_bound_vars.append(var_)

    def get_exclude_bound_vars(self):
        # type: () -> List[Var]
        """
        :return:
        """
        return self._exclude_bound_vars


class ComputeContext:
    """
    ComputeContext
    """

    def __init__(self, _operator=None):
        # type: (Optional[OperatorContext]) -> None
        self._operator = operation.get_context() if _operator is None else _operator  # type: Optional[OperatorContext]

        self._pattern = None  # type: Optional[str]
        self._sub_pattern = None  # type: Optional[str]
        self._vars = []  # type: List[Var]
        self._attr_vars_desc = []  # type: List[AttrVarDesc]
        self._schedules = []  # type: List[ScheduleContext]
        self._current_schedule = None  # type: Optional[ScheduleContext]

        self._addition = {}  # type: Dict[str, Any]
        self._exclude_bound_vars = []  # type: List[Var]

    def __enter__(self):
        self._operator.begin_compute(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._operator.end_compute(self)

    def begin_schedule(self, _schedule):
        # type: (ScheduleContext) -> None
        """
        :param _schedule:
        :return:
        """
        if self._current_schedule is not None:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Exist not finished compute context."
            raise RuntimeError(dict_args, get_error_message(dict_args))
        self._schedules.append(_schedule)
        self._current_schedule = _schedule

    def end_schedule(self, _schedule):
        # type: (ScheduleContext) -> None
        """
        :param _schedule:
        :return:
        """
        if self._current_schedule != _schedule:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Schedule context not match."
            raise RuntimeError(dict_args, get_error_message(dict_args))
        self._current_schedule = None

    def get_schedules(self):
        # type: () -> List[ScheduleContext]
        """
        :return:
        """
        return self._schedules

    def get_current_schedule(self):
        # type: () -> Optional[ScheduleContext]
        """
        :return:
        """
        return self._current_schedule

    def get_operator_context(self):
        # type: () -> OperatorContext
        """
        :return:
        """
        return self._operator

    def set_pattern(self, pattern):
        # type: (str) -> None
        """
        :param pattern:
        :return:
        """
        self._pattern = pattern

    def get_pattern(self):
        # type: () -> Optional[str]
        """
        :return:
        """
        return self._pattern

    def set_sub_pattern(self, sub_pattern):
        # type: (str) -> None
        """
        :param sub_pattern:
        :return:
        """
        self._sub_pattern = sub_pattern

    def get_sub_pattern(self):
        # type: () -> Optional[str]
        """
        :return:
        """
        return self._sub_pattern

    def add_var(self, var_):
        # type: (Var) -> None
        """
        :param var_:
        :return:
        """
        if self._schedules:
            self._schedules[-1].add_var(var_)
        else:
            self._vars.append(var_)

    def get_vars(self):
        # type: () -> List[Var]
        """
        :return:
        """
        return self._vars

    def get_var(self, name):
        # type: (str) -> Optional[Var]
        """
        :param name:
        :return:
        """
        c_sch = self.get_current_schedule()
        if c_sch is not None:
            var0 = c_sch.get_var(name)
            if var0 is not None:
                return var0
        for var_i in self._vars:
            if var_i.get_name() == name:
                return var_i
        return None

    def add_attr_var_desc(self, attr_var_desc):
        # type: (AttrVarDesc) -> None
        """
        :param attr_var_desc:
        :return:
        """
        if self._schedules:
            self._schedules[-1].add_attr_var_desc(attr_var_desc)
        else:
            self._attr_vars_desc.append(attr_var_desc)

    def get_attr_vars_desc(self):
        # type: () -> List[AttrVarDesc]
        """
        :return:
        """
        return self._attr_vars_desc

    def add(self, key, value):
        # type: (str, Any) -> None
        """
        :param key:
        :param value:
        :return:
        """
        self._addition[key] = value

    def get(self, key):
        # type: (str) -> Any
        """
        :param key:
        :return:
        """
        return self._addition.get(key)

    def set_default(self, key, value):
        # type: (str, Any) -> Any
        """
        :param key:
        :param value:
        :return:
        """
        if key not in self._addition:
            self.add(key, value)
        return self.get(key)

    def compute_default(self, key, func):
        # type: (str, Callable) -> Any
        """
        :param key:
        :param func:
        :return:
        """
        if key not in self._addition:
            self.add(key, func())
        return self.get(key)

    def add_exclude_bound_var(self, var_):
        # type: (Var) -> None
        """
        :param var_:
        :return:
        """
        if self._schedules:
            self._schedules[-1].add_exclude_bound_var(var_)
        else:
            self._exclude_bound_vars.append(var_)

    def get_exclude_bound_vars(self):
        # type: () -> List[Var]
        """
        :return:
        """
        return self._exclude_bound_vars


class ScheduleContext:
    """
    ScheduleContext
    """

    def __init__(self, _compute=None):
        # type: (Optional[ComputeContext]) -> None
        self._compute = operation.get_context().get_current_compute() \
            if _compute is None else _compute  # type: Optional[ComputeContext]

        self._vars = []  # type: List[Var]
        self._attr_vars_desc = []  # type: List[AttrVarDesc]
        self._addition = {}  # type: Dict[str, Any]
        self._exclude_bound_vars = []  # type: List[Var]

    def __enter__(self):
        self._compute.begin_schedule(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._compute.end_schedule(self)

    def get_compute_context(self):
        # type: () -> ComputeContext
        """
        :return:
        """
        return self._compute

    def add_var(self, var_):
        # type: (Var) -> None
        """
        :param var_:
        :return:
        """
        self._vars.append(var_)

    def get_vars(self):
        # type: () -> List[Var]
        """
        :return:
        """
        return self._vars

    def get_var(self, name):
        # type: (str) -> Optional[Var]
        """
        :param name:
        :return:
        """
        for var_i in self._vars:
            if var_i.get_name() == name:
                return var_i
        return None

    def add_attr_var_desc(self, attr_var_desc):
        # type: (AttrVarDesc) -> None
        """
        :param attr_var_desc:
        :return:
        """
        self._attr_vars_desc.append(attr_var_desc)

    def get_attr_vars_desc(self):
        # type: () -> List[AttrVarDesc]
        """
        :return:
        """
        return self._attr_vars_desc

    def add(self, key, value):
        # type: (str, Any) -> None
        """
        :param key:
        :param value:
        :return:
        """
        self._addition[key] = value

    def get(self, key):
        # type: (str) -> Any
        """
        :param key:
        :return:
        """
        return self._addition.get(key)

    def set_default(self, key, value):
        # type: (str, Any) -> Any
        """
        :param key:
        :param value:
        :return:
        """
        if key not in self._addition:
            self.add(key, value)
        return self.get(key)

    def compute_default(self, key, func):
        # type: (str, Callable) -> Any
        """
        :param key:
        :param func:
        :return:
        """
        if key not in self._addition:
            self.add(key, func())
        return self.get(key)

    def add_exclude_bound_var(self, var_):
        # type: (Var) -> None
        """
        :param var_:
        :return:
        """
        self._exclude_bound_vars.append(var_)

    def get_exclude_bound_vars(self):
        # type: () -> List[Var]
        """
        :return:
        """
        return self._exclude_bound_vars
