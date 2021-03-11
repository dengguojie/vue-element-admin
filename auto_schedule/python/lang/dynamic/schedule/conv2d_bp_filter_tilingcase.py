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
conv2d backprop filter tiling case
"""
import warnings


def calc_conv2dbp_filter(outs, option=None):
    warnings.warn("te.lang.dynamic.schedule.conv2d_bp_filter_tilingcase is expired, "
        "please replace it with the func tbe.dsl.unify_schedule.conv2d_bp_filter_tilingcase",
        DeprecationWarning)
    from tbe.dsl.unify_schedule.conv2d_bp_filter_tilingcase import calc_conv2dbp_filter
    return calc_conv2dbp_filter(outs, option)
