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
import re
import threading
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from tbe.common.context import op_context
from tbe.common.utils.errormgr import get_error_message

from .context import ComputeContext
from .context import OperatorContext
from .context import ScheduleContext
from .var import AttrVarDesc
from .var import Category
from .var import Var

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
    # type: () -> List[OperatorContextProxy]
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
        # type: () -> str
        """
        :return:
        """
        return self._mode

    def set_op_type(self, op_type):
        # type: (str) -> None
        """
        :param op_type:
        :return:
        """
        self._op_type = op_type

    def get_op_type(self):
        # type: () -> str
        """
        :return:
        """
        return self._op_type

    def __getattr__(self, item):
        return getattr(self._operator_context, item)


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
    if name.startswith("_"):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "the name of var must not start with '_'."
        raise RuntimeError(dict_args, get_error_message(dict_args))
    return _var(name, bound, dtype, Category.CUSTOM, addition)


def var_attr(name, bound=None, dtype="int32", addition=None):
    """
    var attribute
    :param name:
    :param bound:
    :param dtype: such as int32, float16, int32[4]
    :param addition:
    :return:
    """
    # simple dtype, like int32, float16
    simple_dtype_pattern = r"^\w+$"
    if re.match(simple_dtype_pattern, dtype):
        get_context().add_attr_var_desc(AttrVarDesc(name, dtype))

        return _var(name, bound, dtype, Category.ATTR, addition)

    # list dtype, like int32[4], float16[2]
    list_dtype_pattern = r"^(\w+)\[(\d+)]$"
    m2 = re.match(list_dtype_pattern, dtype)
    if m2:
        length = int(m2.group(2))
        if length < 0:
            _raise_error("If var is list dtype, the size must greater than zero.")

        if bound is None:
            bound = [(1, None)] * length

        if not (isinstance(bound, (tuple, list)) and len(bound) == length):
            _raise_error("If var is list dtype, the bound must be list and have the same size as var.")

        s_dtype = m2.group(1)
        attr_vars = []
        for i in range(length):
            attr_vars.append(_var("{}_{}".format(name, i), bound[i], s_dtype, Category.ATTR, addition))

        get_context().add_attr_var_desc(AttrVarDesc(name, s_dtype, length))

        return attr_vars

    _raise_error("Invalid dtype, dtype must like: int32, float16, int32[4], etc.")


def var_inner(name, bound=None, dtype="int32", addition=None):
    """
    add var for internal
    :param name:
    :param bound:
    :param dtype:
    :param addition:
    :return:
    """
    if not name.startswith("_"):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "the name of var must start with '_' in internal use scenarios."
        raise RuntimeError(dict_args, get_error_message(dict_args))
    return _var(name, bound, dtype, Category.NORMAL, addition)


def _var(name, bound, dtype, category, addition):
    var_ = Var(name, bound, dtype, category, addition)
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
    if key.startswith("_"):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "the key of compile_info must not start with '_'."
        raise RuntimeError(dict_args, get_error_message(dict_args))
    _add_compile_info(key, value)


def add_compile_info_inner(key, value):
    """
    add compile info for internal
    :param key:
    :param value:
    :return:
    """
    if not key.startswith("_"):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "the key of compile_info must start with '_' " \
                                      "in internal use scenarios."
        raise RuntimeError(dict_args, get_error_message(dict_args))
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

    return get_op_context().get_compile_info()


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


def _raise_error(message):
    dict_args = dict()
    dict_args["errCode"] = "E90001"
    dict_args["detailed_cause"] = message
    raise RuntimeError(dict_args, get_error_message(dict_args))
