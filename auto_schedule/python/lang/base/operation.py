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
from typing import Optional
from te.lang.base import operation_impl
from te.lang.base.operation_impl import OpMode
from te.lang.base.operation_impl import OperatorContext
from te.lang.base.operation_impl import ComputeContext


def register_operator(op_type, pattern=None):
    """
    :param op_type:
    :param pattern:
    :return:
    """
    return operation_impl.register_operator(op_type, pattern)


def register_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """
    return operation_impl.register_fusion_compute(op_type)


def var(name, bound=None, dtype="int32", addition=None):
    """
    :param name:
    :param bound:
    :param dtype:
    :param addition:
    :return:
    """
    return operation_impl.var(name, bound, dtype, addition)


def add_compile_info(key, value):
    """
    :param key:
    :param value:
    :return:
    """
    return operation_impl.add_compile_info(key, value)


def get_compile_info() -> dict:
    """
    :return:
    """
    return operation_impl.get_compile_info()


def add_build_arg(key, value):
    """
    :param key:
    :param value:
    :return:
    """
    return operation_impl.add_build_arg(key, value)


def compute_build_arg(key, func):
    """
    :param key:
    :param func:
    :return:
    """
    return operation_impl.compute_build_arg(key, func)


def add_exclude_bound_var(var_):
    """
    :param var_:
    :return:
    """
    return operation_impl.add_exclude_bound_var(var_)


def static():
    """
    :return:
    """
    return operation_impl.static()


def dynamic():
    """
    :return:
    """
    return operation_impl.dynamic()


def operator(mode=operation_impl.OpMode.STATIC):
    """
    :param mode:
    :return:
    """
    return operation_impl.operator(mode)


def compute(_operator=None):
    """
    :param _operator:
    :return:
    """
    return operation_impl.compute(_operator)


def schedule(_compute=None):
    """
    :param _compute:
    :return:
    """
    return operation_impl.schedule(_compute)


def get_context() -> Optional[OperatorContext]:
    """
    :return:
    """
    return operation_impl.get_context()


def get_operator(op_type):
    """
    :param op_type:
    :return:
    """
    return operation_impl.get_operator(op_type)


def get_fusion_compute(op_type):
    """
    :param op_type:
    :return:
    """
    return operation_impl.get_fusion_compute(op_type)
