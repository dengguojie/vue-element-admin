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
Padding value
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Union

import numpy as np
from tbe.tvm.expr import ConstExpr
from tbe.tvm.tensor import Tensor


class PaddingValueType(Enum):
    EXACT = auto()
    TENSOR = auto()
    ANY = auto()


@dataclass
class PaddingValue:
    type: PaddingValueType
    dtype: str
    value: Union[np.integer, np.floating] = None
    target: List[Tensor] = field(default_factory = list)

    def add_target(self, tensor):
        if tensor not in self.target:
            self.target.append(tensor)


class SettingValueType(Enum):
    NORMAL = auto()
    BROADCAST = auto()


@dataclass
class SettingValue:
    type: SettingValueType
    dtype: str
    condition: Callable = None
    value: Union[np.integer, np.floating, Callable] = None
    target: List[Tensor] = field(default_factory = list)

    def add_target(self, tensor):
        if tensor not in self.target:
            self.target.append(tensor)
