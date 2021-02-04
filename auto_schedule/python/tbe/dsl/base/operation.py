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
operation
"""
import functools
import threading
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from tbe.common.context import op_context
from tbe.common.utils.errormgr.error_manager_util import get_error_message
from te.tvm import api as tvm

_schedules = {}
_tiling_cases = {}
_builds = {}

_contexts = {}

_operators = {}
_patterns = {}
_fusion_computes = {}
_computes = {}  # type: Dict[Tuple[str, str], Compute]


@dataclass
class Compute:
    """
    The attribute of registered compute.
    """
    func: Callable[..., Any]
    op_mode: str
    support_fusion: bool


def _get_contexts():
    return _contexts.setdefault(threading.currentThread().ident, [])


class OperatorContextProxy:
    """
    OperatorContextProxy
    """

    def __init__(self, mode, operator_context):
        # type: (str, OperatorContext) -> None
        """

        :param mode:
        :param operator_context:
        """
        self._operator_context = operator_context
        self._mode = mode
        self._op_type = None

    def get_mode(self):
        """

        :return:
        """
        return self._mode

    def set_op_type(self, op_type):
        """

        :param op_type:
        :return:
        """
        self._op_type = op_type

    def get_op_type(self):
        """

        :return:
        """
        return self._op_type

    def __getattr__(self, item):
        return getattr(self._operator_context, item)


class OperatorContext:
    """
    OperatorContext
    """

    def __init__(self):
        self._pattern = None

        self._vars = []
        self._computes = []
        self._current_compute = None

        self._addition = {"compile_info": {}, "build_args": {}}
        self._exclude_bound_vars = []

    def set_pattern(self, pattern):
        """
        :param pattern:
        :return:
        """
        self._pattern = pattern

    def get_pattern(self):
        """
        :return:
        """
        # TODO how to support custom pattern
        return self._pattern

    @staticmethod
    def get_mode():
        """
        :return:
        """
        return get_op_context().get_op_mode()

    @staticmethod
    def get_op_type():
        """
        :return:
        """
        op_infos = get_op_context().get_all_op_info()
        if len(op_infos) == 1:
            return op_infos[0].op_type

        return None

    def add_var(self, var_):
        """
        :param var_:
        :return:
        """
        if self._computes:
            self._computes[-1].add_var(var_)
        else:
            self._vars.append(var_)

    def get_vars(self):
        """
        :return:
        """
        return self._vars

    def begin_compute(self, _compute):
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
        """
        :return:
        """
        return self._computes

    def get_current_compute(self) -> Optional['ComputeContext']:
        """
        :return:
        """
        return self._computes[-1] if self._computes else None

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
        for var_i in self._vars:
            if var_i.get_name() == name:
                return var_i
        return None

    def add_exclude_bound_var(self, var_):
        """
        :param var_:
        :return:
        """
        if self._computes:
            self._computes[-1].add_exclude_bound_var(var_)
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
        self._operator = get_context() if _operator is None else _operator
        self._vars = []
        self._schedules = []
        self._current_schedule = None

        self._addition = {}
        self._exclude_bound_vars = []

    def __enter__(self):
        self._operator.begin_compute(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._operator.end_compute(self)

    def begin_schedule(self, _schedule):
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
        """
        :return:
        """
        return self._schedules

    def get_current_schedule(self):
        """
        :return:
        """
        return self._current_schedule

    def get_operator_context(self):
        """

        :return:
        """
        return self._operator

    def add_var(self, var_):
        """
        :param var_:
        :return:
        """
        if self._schedules:
            self._schedules[-1].add_var(var_)
        else:
            self._vars.append(var_)

    def get_vars(self):
        """
        :return:
        """
        return self._vars

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
        for var_i in self._vars:
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
        if self._schedules:
            self._schedules[-1].add_exclude_bound_var(var_)
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
        self._compute = get_context().get_current_compute() \
            if _compute is None else _compute
        self._vars = []

        self._addition = {}
        self._exclude_bound_vars = []

    def __enter__(self):
        self._compute.begin_schedule(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._compute.end_schedule(self)

    def get_compute_context(self):
        """

        :return:
        """
        return self._compute

    def add_var(self, var_):
        """
        :param var_:
        :return:
        """
        self._vars.append(var_)

    def get_vars(self):
        """
        :return:
        """
        return self._vars

    def get_var(self, name):
        """
        :param name:
        :return:
        """
        for var_i in self._vars:
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
        self._tvm_var = tvm.var(name, dtype=dtype)
        self._name = name
        self._bound = bound
        self._addition = addition

    def get_tvm_var(self):
        """
        :return:
        """
        return self._tvm_var

    def get_name(self):
        """
        :return:
        """
        return self._name

    def get_bound(self):
        """
        :return:
        """
        return self._bound

    def get_addition(self):
        """
        :return:
        """
        return self._addition


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
            if _in_compatible_mode():
                context.set_op_type(op_type)
                context.set_pattern(pattern)
            return func(*args, **kwargs)

        _operators[op_type] = wrapper
        _patterns[op_type] = pattern
        return wrapper

    return decorator


def get_operator(op_type):
    """
    :param op_type:
    :return:
    """
    return _operators.get(op_type)


def get_pattern(op_type):
    """

    :param op_type:
    :return:
    """
    return _patterns[op_type]


def register_schedule(pattern):
    """
    :param pattern:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if isinstance(pattern, (tuple, list)):
            for p in pattern:
                _schedules[p] = wrapper
        else:
            _schedules[pattern] = wrapper
        return wrapper

    return decorator


def get_schedule(pattern):
    """
    :param pattern:
    :return:
    """
    return _schedules.get(pattern)


def register_tiling_case(pattern):
    """
    :param pattern:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if isinstance(pattern, (tuple, list)):
            for p in pattern:
                _tiling_cases[p] = wrapper
        else:
            _tiling_cases[pattern] = wrapper

        return wrapper

    return decorator


def get_tiling_case(pattern):
    """
    :param pattern:
    :return:
    """
    return _tiling_cases.get(pattern)


def register_build_pointcut(pattern):
    """
    :param pattern:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if isinstance(pattern, (tuple, list)):
            for p in pattern:
                _builds[p] = wrapper
        else:
            _builds[pattern] = wrapper

        return wrapper

    return decorator


def get_build_pointcut(pattern):
    """
    :param pattern:
    :return:
    """
    return _builds.get(pattern)


def register_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _fusion_computes[op_type] = wrapper
        return wrapper

    return decorator


def get_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """
    return _fusion_computes.get(op_type)


def register_op_compute(op_type, op_mode="dynamic", support_fusion=True):
    # type: (str, str, bool) -> Callable
    """
    :param op_type:
    :param op_mode:
    :param support_fusion:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _computes[(op_type, op_mode)] = Compute(wrapper, op_mode, support_fusion)
        return wrapper

    return decorator


def get_op_compute(op_type, op_mode="dynamic", verbose=False):
    # type: (str, str, bool) -> Union[Callable, Compute, None]
    """
    :param op_type:
    :param op_mode:
    :param verbose:
    :return:
    """
    compute_ = _computes.get((op_type, op_mode))
    if compute_:
        return compute_ if verbose else compute_.func

    return None


def var(name, bound=None, dtype="int32", addition=None):
    """
    add var for external
    :param name:
    :param bound:
    :param dtype: such as int32, float16...
    :param addition:
    :return:
    """
    # TODO add check, cannot start with '_'
    return _var(name, bound, dtype, addition)


def var_inner(name, bound=None, dtype="int32", addition=None):
    """
    add var for internal
    :param name:
    :param bound:
    :param dtype:
    :param addition:
    :return:
    """
    # TODO add check, must start with '_'
    return _var(name, bound, dtype, addition)


def _var(name, bound, dtype, addition):
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
    return get_op_mode() == "dynamic"


def get_op_mode():
    """
    :return:
    """
    if _in_compatible_mode():
        return get_context().get_mode()

    op_context_obj = op_context.get_context()
    return op_context_obj.get_op_mode() if op_context_obj else None


def get_context() -> Union[OperatorContext, OperatorContextProxy, None]:
    """
    :return:
    """
    if _in_compatible_mode():
        return _get_contexts()[-1]

    op_context_obj = get_op_context()
    if op_context_obj:
        return op_context_obj.get_custom_context("dsl")

    return None


def get_op_context() -> Optional[op_context.OpContext]:
    """

    :return:
    """
    return op_context.get_context()


def add_compile_info(key, value):
    """
    add compile info for external
    :param key:
    :param value:
    :return: None
    """
    # TODO add check, cannot start with '_'
    _add_compile_info(key, value)


def add_compile_info_inner(key, value):
    """
    add compile info for internal
    :param key:
    :param value:
    :return:
    """
    # TODO add check, must start with '_'
    _add_compile_info(key, value)


def _add_compile_info(key, value):
    if _in_compatible_mode():
        get_compile_info()[key] = value
    else:
        get_op_context().add_compile_info(key, value)


def get_compile_info() -> dict:
    """
    :return:
    """
    if _in_compatible_mode():
        return get_context().get("compile_info")

    return get_op_context().get_all_compile_info()


def add_build_arg(key, value):
    """
    :param key:
    :param value:
    :return:
    """
    get_build_args()[key] = value


def get_build_args() -> dict:
    """
    :return:
    """
    return get_context().get("build_args")


def add_exclude_bound_var(var_):
    """
    :param var_:
    :return:
    """
    get_context().add_exclude_bound_var(var_)


def static():
    """
    :return:
    """
    return operator("static")


def dynamic():
    """
    :return:
    """
    return operator("dynamic")


def operator(mode="static"):
    """
    :param mode:
    :return:
    """
    return op_context.OpContext(mode)


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


@op_context.register_custom_context("dsl")
def _():
    return OperatorContext()


def _in_compatible_mode():
    return _get_contexts()[-1] if _get_contexts() else None
