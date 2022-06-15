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
elewise schedule
"""
from typing import Optional

import tbe
from tbe import tvm
from tbe.dsl.base import operation
from tbe.dsl.base.operation import register_schedule
import te.lang.cce
from te import platform as cce

from .constants import CompileInfo
from .constants import DTYPE_BYTE_MAPPING
from .constants import FAKE_NODE_TAG
from .constants import INSN_MAPPING
from .constants import Pattern
from .constants import SUPPORT_SCALAR_INSNS
from .constants import TERNARY_INSNS

from . import util
from .softmax_cross_entropy_with_logits_tilingcase import TilingStrategy
from .vector.softmax_norm.softmax_norm_schedule import SoftmaxNormSchedule


# block size in D architecture
BLOCK_SIZE_BYTE = 32

CONST = "const"
ORIGINAL = "original"
COPY = "copy"
VECTOR = "vector"
PHONY = "phony"


@register_schedule(pattern=Pattern.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS)
def schedule(outs, tiling_case):
    """
    :param outs:
    :param tiling_case:
    :return:
    """
    return SoftmaxNormSchedule(outs, tiling_case)
