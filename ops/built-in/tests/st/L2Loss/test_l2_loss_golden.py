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
gen l2loss golden
"""
import numpy as np


def l2_loss_by_numpy(x):
    x = x.reshape(-1)
    x = x * (1.0 / (2**0.5))
    y = np.sum(x * x, axis=0)
    return y


def calc_expect_func(x, y):
    res = l2_loss_by_numpy(x["value"])
    return [res]
