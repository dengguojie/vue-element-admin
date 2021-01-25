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
from te import platform

tbe_platform = platform
register_operator = te.op.register_operator


# pylint: disable=unused-argument
def register_operator_compute(op_type, op_mode="dynamic", support_fusion=False):
    return te.op.register_fusion_compute(op_type)
