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
var
"""
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from tbe import tvm


class Category(Enum):
    """
    category for var
    """
    NORMAL = auto()
    ATTR = auto()
    CUSTOM = auto()


class Var:
    """
    Var
    """

    def __init__(self, name, bound, dtype, category=Category.NORMAL, addition=None):
        # type: (str, Tuple[int, Optional[int]], str, Category, Optional[Dict[str, Any]]) -> None
        """
        :param name:
        :param bound:
        :param dtype:
        :param category:
        :param addition:
        """
        # noinspection PyTypeChecker
        self._tvm_var = tvm.var(name, dtype=dtype)
        self._name = name  # type: str
        self._bound = bound  # type: Tuple[int, Optional[int]]
        self._category = category  # type: Category
        self._addition = addition  # type: Optional[Dict[str, Any]]

    def get_tvm_var(self):
        """
        :return:
        """
        return self._tvm_var

    def get_name(self):
        # type: () -> str
        """
        :return:
        """
        return self._name

    def get_bound(self):
        # type: () -> Tuple[int, Optional[int]]
        """
        :return:
        """
        return self._bound

    def get_category(self):
        # type: () -> Category
        """
        :return:
        """
        return self._category

    def get_addition(self):
        # type: () -> Optional[Dict[str, Any]]
        """
        :return:
        """
        return self._addition


@dataclass
class AttrVarDesc:
    """
    attribute var description
    """
    # the name of var, whether it's primitive type or list type
    name: str
    # such as int32, float16, etc. If list type, take primitive part
    dtype: str
    # None means primitive type, other list type
    length: Optional[int] = None
