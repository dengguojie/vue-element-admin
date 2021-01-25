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
cce schedule
"""
from .constants import Pattern
from . import conv2d_schedule, conv2d_tilingcase
from . import conv2d_bp_input_tilingcase, conv2d_bp_input_schedule
from . import conv2d_bp_filter_tilingcase, cube_schedule
from . import conv3d_schedule, conv3d_tilingcase
from . import gemm_tilingcase
