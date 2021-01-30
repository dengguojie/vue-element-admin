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

    def get_op_info(self, name):
        # type: (str) -> Optional[OpInfo]
        """

        :param name:
        :return:
        """
        for x in self._op_info:
            if x.op_name == name:
                return x
        return None

    def get_all_op_info(self):
        # type: () -> List[OpInfo]
        """

        :return:
        """
        return self._op_info

    def add_compile_info(self, k, v):
        # type: (str, Any) -> None
        """

        :param k:
        :param v:
        :return:
        """
        self._compile_info[k] = v

    def get_compile_info(self, k):
        # type: (str) -> Any
        """

        :param k:
        :return:
        """
        return self._compile_info.get(k)

    def get_all_compile_info(self):
        # type: () -> Dict[str, Any]
        """

        :return:
        """
        return self._compile_info

    def add_build_res(self, k, v):
        # type: (str, Any) -> None
        """

        :param k:
        :param v:
        :return:
        """
        self._build_res[k] = v

    def get_build_res(self, k):
        # type: (str) -> Any
        """

        :param k:
        :return:
        """
        return self._build_res.get(k)

    def get_all_build_res(self):
        # type: () -> Dict[str, Any]
        """

        :return:
        """
        return self._build_res

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
