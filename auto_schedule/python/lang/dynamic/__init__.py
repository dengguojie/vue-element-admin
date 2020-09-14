"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

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
