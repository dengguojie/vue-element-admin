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
operation interface
"""
import functools
import threading
from enum import Enum
from enum import auto
from typing import Optional

from te.tvm import api as tvm
from te.utils.error_manager.error_manager_util import get_error_message

# 'pylint: disable=C0103
_contexts = {}

operators = {}
fusion_computes = {}
schedules = {}
tiling_cases = {}
builds = {}


def _get_contexts():
    return _contexts.setdefault(threading.currentThread().ident, [])


class OpMode(Enum):
    """
    OpMode
    """
    STATIC = auto()
    DYNAMIC = auto()


class OperatorContext:
    """
    OperatorContext
    """

    def __init__(self, mode: OpMode):
        self.mode = mode

        self.op_type = None
        self.pattern = None

        self.vars = []
        self.computes = []
        self.current_compute = None

        self._addition = {"compile_info": {}, "build_args": {}}
        self._exclude_bound_vars = []

    def __enter__(self):
        _get_contexts().append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _get_contexts().pop()

    def get_mode(self):
        """
        :return:
        """
        return self.mode

    def set_op_type(self, op_type):
        """
        :param op_type:
        :return:
        """
        self.op_type = op_type

    def get_op_type(self):
        """
        :return:
        """
        return self.op_type

    def set_pattern(self, pattern):
        """
        :param pattern:
        :return:
        """
        self.pattern = pattern

    def get_pattern(self):
        """
        :return:
        """
        return self.pattern

    def add_var(self, var_):
        """
        :param var_:
        :return:
        """
        if self.computes:
            self.computes[-1].add_var(var_)
        else:
            self.vars.append(var_)

    def get_vars(self):
        """
        :return:
        """
        return self.vars

    def begin_compute(self, _compute):
        """
        :param _compute:
        :return:
        """
        if self.current_compute is not None:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Exist not finished compute context."
            raise RuntimeError(dict_args, get_error_message(dict_args))

        self.computes.append(_compute)
        self.current_compute = _compute

    def end_compute(self, _compute):
        """
        :param _compute:
        :return:
        """
        if self.current_compute != _compute:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Compute context not match."
            raise RuntimeError(dict_args, get_error_message(dict_args))
        self.current_compute = None

    def get_computes(self):
        """
        :return:
        """
        return self.computes

    def get_current_compute(self):
        """
        :return:
        """
        return self.computes[-1] if self.computes else None

    def add(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        self._addition[key] = value

    def get(self, key):
        """
        :param key:
        :return:
        """
        return self._addition.get(key)

    def set_default(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        if key not in self._addition:
            self.add(key, value)
        return self.get(key)

    def compute_default(self, key, func):
        """
        :param key:
        :param func:
        :return:
        """
        if key not in self._addition:
            self.add(key, func())
        return self.get(key)

    def get_var(self, name):
        """
        :param name:
        :return:
        """
        c_cmp = self.get_current_compute()
        if c_cmp is not None:
            var0 = c_cmp.get_var(name)
            if var0 is not None:
                return var0
        for var_i in self.vars:
            if var_i.get_name() == name:
                return var_i
        return None

    def add_exclude_bound_var(self, var_):
        """
        :param var_:
        :return:
        """
        if self.computes:
            self.computes[-1].add_exclude_bound_var(var_)
        else:
            self._exclude_bound_vars.append(var_)

    def get_exclude_bound_vars(self):
        """
        :return:
        """
        return self._exclude_bound_vars


class ComputeContext:
    """
    ComputeContext
    """

    def __init__(self, _operator=None):
        self.operator = get_context() if _operator is None else _operator
        self.vars = []
        self.schedules = []
        self.current_schedule = None

        self._addition = {}
        self._exclude_bound_vars = []

    def __enter__(self):
        self.operator.begin_compute(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.operator.end_compute(self)

    def begin_schedule(self, _schedule):
        """
        :param _schedule:
        :return:
        """
        if self.current_schedule is not None:
            raise RuntimeError("Exist not finished schedule context.")
        self.schedules.append(_schedule)
        self.current_schedule = _schedule

    def end_schedule(self, _schedule):
        """
        :param _schedule:
        :return:
        """
        if self.current_schedule != _schedule:
            raise RuntimeError("Schedule context not match.")
        self.current_schedule = None

    def get_schedules(self):
        """
        :return:
        """
        return self.schedules

    def get_current_schedule(self):
        """
        :return:
        """
        return self.current_schedule

    def get_operator_context(self):
        return self.operator

    def add_var(self, var_):
        """
        :param var_:
        :return:
        """
        if self.schedules:
            self.schedules[-1].add_var(var_)
        else:
            self.vars.append(var_)

    def get_vars(self):
        """
        :return:
        """
        return self.vars

    def get_var(self, name):
        """
        :param name:
        :return:
        """
        c_sch = self.get_current_schedule()
        if c_sch is not None:
            var0 = c_sch.get_var(name)
            if var0 is not None:
                return var0
        for var_i in self.vars:
            if var_i.get_name() == name:
                return var_i
        return None

    def add(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        self._addition[key] = value

    def get(self, key):
        """
        :param key:
        :return:
        """
        return self._addition.get(key)

    def set_default(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        if key not in self._addition:
            self.add(key, value)
        return self.get(key)

    def compute_default(self, key, func):
        """
        :param key:
        :param func:
        :return:
        """
        if key not in self._addition:
            self.add(key, func())
        return self.get(key)

    def add_exclude_bound_var(self, var_):
        """
        :param var_:
        :return:
        """
        if self.schedules:
            self.schedules[-1].add_exclude_bound_var(var_)
        else:
            self._exclude_bound_vars.append(var_)

    def get_exclude_bound_vars(self):
        """
        :return:
        """
        return self._exclude_bound_vars


class ScheduleContext:
    """
    ScheduleContext
    """

    def __init__(self, _compute=None):
        self.compute = get_context().get_current_compute() \
            if _compute is None else _compute
        self.vars = []

        self._addition = {}
        self._exclude_bound_vars = []

    def __enter__(self):
        self.compute.begin_schedule(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.compute.end_schedule(self)

    def get_compute_context(self):
        return self.compute

    def add_var(self, var_):
        """
        :param var_:
        :return:
        """
        self.vars.append(var_)

    def get_vars(self):
        """
        :return:
        """
        return self.vars

    def get_var(self, name):
        """
        :param name:
        :return:
        """
        for var_i in self.vars:
            if var_i.get_name() == name:
                return var_i
        return None

    def add(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        self._addition[key] = value

    def get(self, key):
        """
        :param key:
        :return:
        """
        return self._addition.get(key)

    def set_default(self, key, value):
        """
        :param key:
        :param value:
        :return:
        """
        if key not in self._addition:
            self.add(key, value)
        return self.get(key)

    def compute_default(self, key, func):
        """
        :param key:
        :param func:
        :return:
        """
        if key not in self._addition:
            self.add(key, func())
        return self.get(key)

    def add_exclude_bound_var(self, var_):
        """
        :param var_:
        :return:
        """
        self._exclude_bound_vars.append(var_)

    def get_exclude_bound_vars(self):
        """
        :return:
        """
        return self._exclude_bound_vars


class Var:
    """
    Var
    """

    def __init__(self, name, bound, dtype, addition=None):
        self.tvm_var = tvm.var(name, dtype=dtype)
        self.name = name
        self.bound = bound
        self.addition = addition

    def get_tvm_var(self):
        """
        :return:
        """
        return self.tvm_var

    def get_name(self):
        """
        :return:
        """
        return self.name

    def get_bound(self):
        """
        :return:
        """
        return self.bound

    def get_addition(self):
        """
        :return:
        """
        return self.addition


def register_operator(op_type, pattern=None):
    """
    :param op_type:
    :param pattern:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = get_context()
            context.set_op_type(op_type)
            context.set_pattern(pattern)
            return func(*args, **kwargs)

        operators[op_type] = wrapper
        return wrapper

    return decorator


def get_operator(op_type):
    """
    :param op_type:
    :return:
    """
    return operators.get(op_type)


def register_schedule(pattern):
    """
    :param pattern:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        schedules[pattern] = wrapper
        return wrapper

    return decorator


def get_schedule(pattern):
    """
    :param pattern:
    :return:
    """
    return schedules.get(pattern)


def register_tiling_case(pattern):
    """
    :param pattern:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        tiling_cases[pattern] = wrapper
        return wrapper

    return decorator


def get_tiling_case(pattern):
    """
    :param pattern:
    :return:
    """
    return tiling_cases.get(pattern)


def register_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        fusion_computes[op_type] = wrapper
        return wrapper

    return decorator


def get_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """
    return fusion_computes.get(op_type)


def register_build_pointcut(pattern):
    """
    :param pattern:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        builds[pattern] = wrapper
        return wrapper

    return decorator


def get_build_pointcut(pattern):
    """
    :param pattern:
    :return:
    """
    return builds.get(pattern)


def var(name, bound=None, dtype="int32", addition=None):
    """
    :param name:
    :param bound:
    :param dtype:
    :param addition:
    :return:
    """
    var_ = Var(name, bound, dtype, addition)
    context = get_context()
    if context is not None:
        context.add_var(var_)
    return var_.get_tvm_var()


def get_te_var(name) -> Var:
    """
    :param name:
    :return:
    """
    context = get_context()
    return context.get_var(name) if context else None


def in_dynamic():
    """
    :return:
    """
    context = get_context()
    return context is not None and context.get_mode() == OpMode.DYNAMIC


def get_context() -> Optional[OperatorContext]:
    """
    :return:
    """
    return _get_contexts()[-1] if _get_contexts() else None


def add_compile_info(key, value):
    """
    :param key:
    :param value:
    :return:
    """
    get_compile_info()[key] = value


def compute_compile_info(key, func):
    """
    :param key:
    :param func:
    :return:
    """
    value = func(get_compile_info().get(key))
    add_compile_info(key, value)


def get_compile_info() -> dict:
    """
    :return:
    """
    context = get_context()
    return context.get("compile_info")


def add_build_arg(key, value):
    """
    :param key:
    :param value:
    :return:
    """
    get_build_args()[key] = value


def compute_build_arg(key, func):
    """
    :param key:
    :param func:
    :return:
    """
    value = func(get_build_args().get(key))
    add_build_arg(key, value)


def get_build_args() -> dict:
    """
    :return:
    """
    context = get_context()
    return context.get("build_args")


def add_exclude_bound_var(var_):
    """
    :param var_:
    :return:
    """
    context = get_context()
    context.add_exclude_bound_var(var_)


def static():
    """
    :return:
    """
    return OperatorContext(OpMode.STATIC)


def dynamic():
    """
    :return:
    """
    return OperatorContext(OpMode.DYNAMIC)


def operator(mode=OpMode.STATIC):
    """
    :param mode:
    :return:
    """
    return OperatorContext(mode)


def compute(_operator=None):
    """
    :param _operator:
    :return:
    """
    return ComputeContext(_operator)


def schedule(_compute=None):
    """
    :param _compute:
    :return:
    """
    return ScheduleContext(_compute)
