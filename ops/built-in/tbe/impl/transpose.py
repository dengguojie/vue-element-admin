# Copyright 2019 Huawei Technologies Co., Ltd
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
transpose
"""

# pylint: disable=unused-argument
def check_supported(input_x, perm, output_y, kernel_name="dynamic_transpose"):
    """
    dynamic transpose is selected when any condition is true
        -1 in input_x shape
        -1 in output_y shape
        -2 in input_x shape
    """
    x_shape = input_x.get("ori_shape")
    y_shape = output_y.get("ori_shape")
    x_dtype = input_x.get("dtype")

    if (-1 in x_shape or -1 in y_shape or -2 in x_shape) and\
            x_dtype in ("float", "float32", "int32", "uint32", "int16", "uint16", "float16"):
        return True

    return False

