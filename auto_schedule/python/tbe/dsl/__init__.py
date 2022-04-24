#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
tbe dsl
"""
from .api import ceil
from .api import floor
from .api import round
from .api import trunc
from .api import round_half_up
from .api import cast_to

from .api import vadd
from .api import vsub
from .api import vmul
from .api import vdiv
from .api import vrec
from .api import vmod
from .api import vmax
from .api import vmin
from .api import vlog
from .api import vexp
from .api import vabs
from .api import vsqrt
from .api import vrsqrt
from .api import vnot
from .api import vor
from .api import vand
from .api import vlogic
from .api import vadds
from .api import vmuls
from .api import vmaxs
from .api import vmins
from .api import vaxpy
from .api import vmla
from .api import vmadd
from .api import vcmp
from .api import vsel
from .api import vcmpsel

from .api import vmaddrelu
from .api import vaddrelu
from .api import vsubrelu
from .api import vrelu
from .api import vlrelu
from .api import clip
from .api import broadcast
from .api import set_value

from .api import transpose

from .api import reduce_sum
from .api import reduce_min
from .api import reduce_max
from .api import reduce_prod
from .api import tuple_sum

from .api import split
from .api import concat

from .api import inplace_add
from .api import inplace_sub
from .api import inplace_update

from .api import pooling2d
from .api import pooling3d
from .api import max_pooling3d_grad_grad

from .api import auto_schedule

from .api import build

from .api import classify

from .api import var
from .api import var_attr
from .api import add_build_arg
from .api import add_exclude_bound_var
from .api import compute
from .api import schedule

#cube
from .api import conv2d_backprop_filter
from .api import conv2d_backprop_input
from .api import conv3d_backprop_filter
from .api import conv3d_backprop_input
from .api import conv3d
from .api import depthwise_conv2d_backprop_filter
from .api import depthwise_conv2d_backprop_input
from .api import depthwise_conv2d
from .api import dilation
from .api import gemm
from .api import matmul

#gather
from .api import gather
from .api import gather_nd

from .api import slice

from .api import transdata

from .api import conv
