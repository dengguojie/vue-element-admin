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
base init
"""
from __future__ import absolute_import as _abs

from .shape_classifier import classify
from .shape_classifier import Mode
from .operation import register_operator
from .operation import compute
from .operation import add_compile_info
from .operation import var
from .operation import add_exclude_bound_var
from .operation import register_fusion_compute
from .operation_impl import register_tiling_case
from .operation_impl import register_schedule
from .operation_impl import get_te_var
