# Copyright 2021 Huawei Technologies Co., Ltd
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
test_sync_batch_norm_backward_elemt_golden.py
"""

def calc_expect_func(grad_output, save_input, mean, invstd, weight, mean_dy, mean_dy_xmu, grad_input):
    """
    calc_expect_func
    """
    grad_output = grad_output.get("value")
    save_input = save_input.get("value")
    mean = mean.get("value")
    invstd = invstd.get("value")
    weight = weight.get("value")
    mean_dy = mean_dy.get("value")
    mean_dy_xmu = mean_dy_xmu.get("value")

    output_dy = grad_output - mean_dy
    input_mean = save_input - mean
    invstd_sq = invstd * invstd
    invstd_dy_xmu = invstd_sq * mean_dy_xmu
    input_invstd = input_mean * invstd_dy_xmu
    ouput_input = output_dy - input_invstd
    invstd_w = invstd * weight
    grad_input = ouput_input * invstd_w
    return [grad_input]