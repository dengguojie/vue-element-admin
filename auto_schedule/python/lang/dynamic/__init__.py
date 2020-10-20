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
compute schedule init
"""
# 'pylint: disable=W0622
from .schedule.auto_schedule import build

from .compute.elewise_compute import vmuls, vadds, vmaxs, vmins, vlog, vexp, \
    vabs, vrec, vrelu, vnot, vsqrt, vrsqrt, vadd, vsub, vmul, vdiv, vmin, \
    vmax, vor, vand, vaxpy, vmla, vmadd, vmaddrelu, vcmpsel, vmod
from .compute.broadcast_compute import broadcast
from .compute.cast_compute import cast_to, round_to, ceil, floor, trunc, \
    round_d

from .compute.util import dsl_check_support, shape_to_list
from .compute.reduction_compute import sum, reduce_min, reduce_max, \
    reduce_prod, tuple_sum

from te.utils.cce import auto_schedule
