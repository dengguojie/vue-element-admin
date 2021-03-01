# Copyright 2021 Huawei Technologies Co., Ltd
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
platform adapter
"""
import te
from tbe.dsl.base import operation
import tbe as platform_tbe
from te import platform
from te import tik
from te import tvm


tbe_platform = platform
register_operator = te.op.register_operator


# pylint: disable=unused-argument
def register_operator_compute(op_type, op_mode="dynamic", support_fusion=False):
    return te.op.register_fusion_compute(op_type)


class OpPatternMode:
    """
    Mode
    """
    NONE = ""
    ELEWISE = "elewise"
    ELEWISE_WITH_BROADCAST = "broadcast"
    REDUCE = "reduce"


class OpImplMode:
    HIGH_PERFORMANCE = "high_performance"
    HIGH_PRECISION = "high_precision"


class TbeContextKey:
    """
    TbeContextKey
    """
    PATTERN = "pattern"


tbe = platform_tbe.dsl
para_check = platform_tbe.common.utils.para_check
shape_util = platform_tbe.common.utils.shape_util
error_manager_vector = platform_tbe.common.utils.errormgr
classify = tbe.classify
operation = operation
tik = tik
tvm = tvm
tbe_context = platform_tbe.common.context

