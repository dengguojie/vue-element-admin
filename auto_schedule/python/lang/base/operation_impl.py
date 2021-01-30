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
operation impl
"""
import contextlib
from typing import Any
from typing import Callable
from typing import Union


def register_operator(op_type, pattern=None):
    """
    :param op_type:
    :param pattern:
    :return:
    """
    return _get_new_operation().register_operator(op_type, pattern)


def get_operator(op_type):
    """
    :param op_type:
    :return:
    """
    return _get_new_operation().get_operator(op_type)


def register_schedule(pattern):
    """
    :param pattern:
    :return:
    """
    return _get_new_operation().register_schedule(pattern)


def get_schedule(pattern):
    """
    :param pattern:
    :return:
    """
    return _get_new_operation().get_schedule(pattern)


def register_tiling_case(pattern):
    """
    :param pattern:
    :return:
    """
    return _get_new_operation().register_tiling_case(pattern)


def get_tiling_case(pattern):
    """
    :param pattern:
    :return:
    """
    return _get_new_operation().get_tiling_case(pattern)


def register_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """
    return _get_new_operation().register_fusion_compute(op_type)


def get_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """
    return _get_new_operation().get_fusion_compute(op_type)


def register_op_compute(op_type, op_mode="dynamic", support_fusion=True):
    # type: (str, str, bool) -> Callable
    """
    :param op_type:
    :param op_mode:
    :param support_fusion:
    :return:
    """
    return _get_new_operation().register_op_compute(op_type, op_mode, support_fusion)


def get_op_compute(op_type, op_mode="dynamic", verbose=False):
    # type: (str, str, bool) -> Union[Callable, Any, None]
    """
    :param op_type:
    :param op_mode:
    :param verbose:
    :return:
    """
    return _get_new_operation().get_op_compute(op_type, op_mode, verbose)


def register_build_pointcut(pattern):
    """
    :param pattern:
    :return:
    """
    return _get_new_operation().register_build_pointcut(pattern)


def get_build_pointcut(pattern):
    """
    :param pattern:
    :return:
    """
    return _get_new_operation().get_build_pointcut(pattern)


def var(name, bound=None, dtype="int32", addition=None):
    """
    :param name:
    :param bound:
    :param dtype:
    :param addition:
    :return:
    """
    return _get_new_operation().var(name, bound, dtype, addition)


def get_te_var(name):
    """
    :param name:
    :return:
    """
    return _get_new_operation().get_te_var(name)


def in_dynamic():
    """
    :return:
    """
    return _get_new_operation().in_dynamic()


def get_op_mode():
    """
    :return:
    """
    return _get_new_operation().get_op_mode()


def get_context():
    """
    :return:
    """
    return _get_new_operation().get_context()


def add_compile_info(key, value):
    """
    :param key:
    :param value:
    :return:
    """
    _get_new_operation().add_compile_info(key, value)


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
    return _get_new_operation().get_compile_info()


def add_build_arg(key, value):
    """
    :param key:
    :param value:
    :return:
    """
    _get_new_operation().add_build_arg(key, value)


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
    return _get_new_operation().get_build_args()


def add_exclude_bound_var(var_):
    """
    :param var_:
    :return:
    """
    _get_new_operation().add_exclude_bound_var(var_)


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


# noinspection PyProtectedMember
@contextlib.contextmanager
def operator(mode="static"):
    """
    :param mode:
    :return:
    """
    context = _get_new_operation().OperatorContext()
    context_proxy = _get_new_operation().OperatorContextProxy(mode, context)
    _get_new_operation()._get_contexts().append(context_proxy)
    try:
        yield context
    finally:
        _get_new_operation()._get_contexts().pop()


def compute(_operator=None):
    """
    :param _operator:
    :return:
    """
    return _get_new_operation().compute(_operator)


def schedule(_compute=None):
    """
    :param _compute:
    :return:
    """
    return _get_new_operation().schedule(_compute)


def _get_new_operation():
    from tbe.dsl.base import operation as new_operation
    return new_operation
