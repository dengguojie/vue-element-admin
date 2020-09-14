#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cce schedule
"""

from .constants import Pattern, INSN_MAPPING, SUPPORT_SCALAR_INSNS, BROADCAST_INSNS, \
    DTYPE_BYTE_MAPPING, FAKE_NODE_TAG, NEED_TEMP_SPACE_INSNS
from . import elewise_schedule, elewise_tilingcase
from . import reduce_schedule
from . import conv2d_schedule, conv2d_tilingcase
from . import conv2d_bp_input_tilingcase, conv2d_bp_input_schedule
