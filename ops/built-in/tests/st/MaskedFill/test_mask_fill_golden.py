# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
test_mask_fill_golden.py
"""
import numpy as np


def do_mask_fill_by_numpy(x, mask, value):
    """do_mask_fill_by_numpy
    res = (value * mask) + x * (1 - mask)
    """
    mask_data = np.broadcast_to(mask, x.shape).astype(np.bool)
    value_data = np.broadcast_to(value, x.shape)
    res = mask_data.astype(value.dtype) * value_data + (1 - mask_data.astype(value.dtype)) * x

    return res


def calc_expect_func(x, mask, value, y):
    """calc_expect_func
    """
    res = do_mask_fill_by_numpy(x["value"], mask["value"], value["value"])
    return [res]