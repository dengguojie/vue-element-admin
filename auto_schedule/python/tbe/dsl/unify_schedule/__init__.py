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
unify schedule
"""
from .constants import Pattern, INSN_MAPPING, SUPPORT_SCALAR_INSNS, BROADCAST_INSNS, \
    DTYPE_BYTE_MAPPING, FAKE_NODE_TAG, NEED_TEMP_SPACE_INSNS, CompileInfo, VSEL_INSNS, \
    VCMPSEL_INSNS, NEED_SPACE_WITH_DIFF_TYPE, TERNARY_INSNS, NEED_EXTENT_NODE_INSNS, VCMP_INSNS
from . import elewise_schedule, elewise_tilingcase
from . import reduce_schedule
from . import reduce_tilingcase

from .unify_auto_schedule import schedule_cce
from .unify_auto_schedule import build
