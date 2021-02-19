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
op context
"""
import functools
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .op_info import OpInfo

# 'pylint: disable=C0103
_contexts = {}
_custom_contexts = {}


def _get_contexts():
    return _contexts.setdefault(threading.currentThread().ident, [])


class OpContext:
    """
    Op Context
    """

    def __init__(self, op_mode):
        # type: (str) -> None
        """

        :param op_mode: dynamic, static, pre-static
        """
        self._op_mode = op_mode  # type: str

        self._op_info = []  # type: List[OpInfo]
        self._compile_info = {}  # type: dict
        self._build_res = {}  # type: dict

        self._buffer_manager = None  # type: Any

        # enum: initially_build, fuzzily_build, accurately_build
        self._build_type = None  # type: Optional[str]

        # json format string, used in initially/fuzzily build
        self._missing_support_info = None  # type: Optional[str]

        # used in initially/fuzzily build
        self._build_json_result = {}  # type: Dict[str, Any]

        self._additional_params = {}  # type: Dict[str, Any]

        self._custom_context = {}  # type: Dict[str, Callable[[], Any]]
        for _name, _clz in _custom_contexts.items():
            self._custom_context[_name] = _clz()

    def __enter__(self):
        _get_contexts().append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _get_contexts().pop()

    def get_op_mode(self):
        # type: () -> str
        """

        :return:
        """
        return self._op_mode

    def add_op_info(self, op_info):
        # type: (OpInfo) -> None
        """

        :param op_info:
        :return:
        """
        self._op_info.append(op_info)

    def get_op_info(self, name=None):
        # type: (Optional[str]) -> Union[List[OpInfo], Optional[OpInfo]]
        """

        :param name: If none, return all op info.
        :return:
        """
        if name is None:
            return self._op_info

        for x in self._op_info:
            if x.op_name == name:
                return x
        return None

    def add_compile_info(self, k, v):
        # type: (str, Any) -> None
        """

        :param k:
        :param v:
        :return:
        """
        self._compile_info[k] = v

    def get_compile_info(self, k=None):
        # type: (Optional[str]) -> Union[Dict[str, Any], Any]
        """

        :param k: If none, return all compile info.
        :return:
        """
        if k is None:
            return self._compile_info

        return self._compile_info.get(k)

    def add_build_res(self, k, v):
        # type: (str, Any) -> None
        """

        :param k:
        :param v:
        :return:
        """
        self._build_res[k] = v

    def get_build_res(self, k=None):
        # type: (Optional[str]) -> Union[Dict[str, Any], Any]
        """

        :param k: If none, return all build res.
        :return:
        """
        if k is None:
            return self._build_res

        return self._build_res.get(k)

    def get_buffer_manager(self):
        # type: () -> Any
        """

        :return:
        """
        return self._buffer_manager

    def set_buffer_manager(self, buffer_manager):
        # type: (Any) -> None
        """

        :param buffer_manager:
        :return:
        """
        self._buffer_manager = buffer_manager

    def get_custom_context(self, k):
        # type: (str) -> Any
        """

        :param k:
        :return:
        """
        return self._custom_context.get(k)

    def get_build_type(self):
        # type: () -> Optional[str]
        """

        :return:
        """
        return self._build_type

    def set_build_type(self, build_type):
        # type: (str) -> None
        """

        :param build_type:
        :return:
        """
        self._build_type = build_type

    def get_missing_support_info(self):
        # type: () -> Optional[str]
        """

        :return:
        """
        return self._missing_support_info

    def set_missing_support_info(self, missing_support_info):
        # type: (str) -> None
        """

        :param missing_support_info:
        :return:
        """
        self._missing_support_info = missing_support_info

    def add_build_json_result(self, k, v):
        # type: (str, Any) -> None
        """

        :param k:
        :param v:
        :return:
        """
        self._build_json_result[k] = v

    def get_build_json_result(self, k=None):
        # type: (Optional[str]) -> Union[Dict[str, Any], Any]
        """

        :param k: If none, return all build json result.
        :return:
        """
        if k is None:
            return self._build_json_result

        return self._build_json_result.get(k)

    def add_addition(self, key, value):
        # type: (str, Any) -> None
        """

        :param key:
        :param value:
        :return:
        """
        self._additional_params[key] = value

    def get_addition(self, key):
        # type: (str) -> Any
        """

        :param key:
        :return:
        """
        return self._additional_params.get(key)


def get_context():
    # type: () -> Optional[OpContext]
    """

    :return:
    """
    return _get_contexts()[-1] if _get_contexts() else None


def register_custom_context(name):
    """

    :param name:
    :return:
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _custom_contexts[name] = wrapper
        return wrapper

    return decorator
