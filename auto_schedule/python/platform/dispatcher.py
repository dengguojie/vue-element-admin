#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

dispatcher for static and dynamic
"""
import functools

from topi import generic

from ..lang import dynamic
from . import operation

# dsl of dynamic mapping
dsl_mapping = {
    "vmuls": dynamic.vmuls,
    "vadds": dynamic.vadds,
    "vmaxs": dynamic.vmaxs,
    "vmins": dynamic.vmins,
    "vlog": dynamic.vlog,
    "vexp": dynamic.vexp,
    "vabs": dynamic.vabs,
    "vrec": dynamic.vrec,
    "vrelu": dynamic.vrelu,
    "vnot": dynamic.vnot,
    "vsqrt": dynamic.vsqrt,
    "vrsqrt": dynamic.vrsqrt,
    "vadd": dynamic.vadd,
    "vsub": dynamic.vsub,
    "vmul": dynamic.vmul,
    "vdiv": dynamic.vdiv,
    "vmin": dynamic.vmin,
    "vmax": dynamic.vmax,
    "vor": dynamic.vor,
    "vand": dynamic.vand,
    "vaxpy": dynamic.vaxpy,
    "vmla": dynamic.vmla,
    "vmadd": dynamic.vmadd,
    "vmaddrelu": dynamic.vmaddrelu,
    "vcmpsel": dynamic.vcmpsel,
    "vmod": dynamic.vmod,

    "broadcast": dynamic.broadcast,

    "cast_to": dynamic.cast_to,
    "round_to": dynamic.round_to,
    "ceil": dynamic.ceil,
    "floor": dynamic.floor,
    "trunc": dynamic.trunc,
    "round_d": dynamic.round_d,

    "sum": dynamic.sum,
    "reduce_min": dynamic.reduce_min,
    "reduce_max": dynamic.reduce_max,
    "reduce_prod": dynamic.reduce_prod,
    "tuple_sum": dynamic.tuple_sum,
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
            return dynamic.build(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


@generic.auto_schedule.register("cce")
def schedule_cce(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    if operation.in_dynamic():
        from ..lang.dynamic.schedule import auto_schedule as dynamic_schedule
        return dynamic_schedule.schedule_cce(outs, option)
    from ..lang.cce.te_schedule import cce_schedule as static_schedule
    return static_schedule.schedule_cce(outs, option)
