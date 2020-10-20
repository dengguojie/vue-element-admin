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
dispatcher for static and dynamic
"""
import functools

from topi import generic
from te.utils import cce
from te.platform import operation
from te.lang.dynamic.compute import elewise_compute as dynamic_elewise
from te.lang.dynamic.compute import cast_compute as dynamic_cast
from te.lang.dynamic.compute import reduction_compute as dynamic_reduce
from te.lang.dynamic.compute import broadcast_compute as \
    dynamic_broadcast
from te.lang.dynamic.schedule import auto_schedule as dynamic_schedule
from te.lang.dynamic.schedule.auto_schedule import build as _dynamic_build
from te.lang.cce.te_schedule import cce_schedule as static_schedule

# dsl of dynamic mapping
dsl_mapping = {
    "vmuls": dynamic_elewise.vmuls,
    "vadds": dynamic_elewise.vadds,
    "vmaxs": dynamic_elewise.vmaxs,
    "vmins": dynamic_elewise.vmins,
    "vlog": dynamic_elewise.vlog,
    "vexp": dynamic_elewise.vexp,
    "vabs": dynamic_elewise.vabs,
    "vrec": dynamic_elewise.vrec,
    "vrelu": dynamic_elewise.vrelu,
    "vnot": dynamic_elewise.vnot,
    "vsqrt": dynamic_elewise.vsqrt,
    "vrsqrt": dynamic_elewise.vrsqrt,
    "vadd": dynamic_elewise.vadd,
    "vsub": dynamic_elewise.vsub,
    "vmul": dynamic_elewise.vmul,
    "vdiv": dynamic_elewise.vdiv,
    "vmin": dynamic_elewise.vmin,
    "vmax": dynamic_elewise.vmax,
    "vor": dynamic_elewise.vor,
    "vand": dynamic_elewise.vand,
    "vaxpy": dynamic_elewise.vaxpy,
    "vmla": dynamic_elewise.vmla,
    "vmadd": dynamic_elewise.vmadd,
    "vmaddrelu": dynamic_elewise.vmaddrelu,
    "vcmpsel": dynamic_elewise.vcmpsel,
    "vmod": dynamic_elewise.vmod,

    "broadcast": dynamic_broadcast.broadcast,

    "cast_to": dynamic_cast.cast_to,
    "round_to": dynamic_cast.round_to,
    "ceil": dynamic_cast.ceil,
    "floor": dynamic_cast.floor,
    "trunc": dynamic_cast.trunc,
    "round_d": dynamic_cast.round_d,

    "sum": dynamic_reduce.sum,
    "reduce_min": dynamic_reduce.reduce_min,
    "reduce_max": dynamic_reduce.reduce_max,
    "reduce_prod": dynamic_reduce.reduce_prod,
    "tuple_sum": dynamic_reduce.tuple_sum,
}


def dispatch_dsl(func):
    """
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if operation.in_dynamic():
            return dsl_mapping[func.__name__](*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def dispatch_build(func):
    """
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if operation.in_dynamic():
            return _dynamic_build(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


@cce.auto_schedule.register("cce")
@generic.auto_schedule.register("cce")
def schedule_cce(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    if operation.in_dynamic():
        return dynamic_schedule.schedule_cce(outs, option)

    return static_schedule.schedule_cce(outs, option)
