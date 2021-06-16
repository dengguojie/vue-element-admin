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

from te import platform as cce

AICORE_NUM = cce.cce_conf.get_soc_spec(cce.cce_conf.CORE_NUM)


def _by_dynamic_static_union_version(shape, core_num):
    """
    temporary function, for dynamic & static union version not fully verified
    """
    if core_num == 1:
        return False

    white_list_shape = [
                         [2, 512, 1024], [1024, 91], [2, 512, 1024], [256, 784, 91],
                         [1024, 364], [2, 128, 91, 28, 28], [2, 128, 28, 28, 91],
                         [1024, 1024], [2, 512, 1024], [12544, 1024], [2, 512, 12544],
                         [4, 2, 4, 2, 3, 64], [1100, 1100], [2, 100, 1], [200, 116, 116, 4],
                         [1100], [1100, 512], [1, 512, 1, 24], [1, 512, 24], [38, 67, 512], [67, 38, 512],
                         [1, 24, 5, 5], [1, 486, 5, 5], [1, 24, 10, 10], [1, 486, 10, 10],
                         [1, 24, 20, 20], [1, 486, 20, 20], [1, 24, 40, 40], [1, 486, 40, 40],
                         [1, 24, 80, 80], [1, 486, 80, 80], [12, 8, 8, 36, 120],
                         [1, 100, 28, 28, 91], [4, 100, 28, 28, 91], [8, 100, 28, 28, 91], [16, 100, 28, 28, 91],
                         [80, 8, 1, 240], [80, 240, 8], [80, 240, 1, 8], [8, 80, 240], [240, 8, 64], [80, 8, 84], [8, 80, 64]
                       ]
    shape_t = list(shape)
    if shape_t in white_list_shape:
        return True
    return False


def _is_dynamic(input_x, perm, output_y):
    x_shape = list(input_x.get("ori_shape"))
    p_shape = [perm.get("ori_shape")]
    y_shape = list(output_y.get("ori_shape"))
    if (-1 in x_shape) or (-1 in p_shape) or (-1 in y_shape) or \
       (-2 in x_shape) or (-2 in p_shape) or (-2 in y_shape):
        return True
    return False


# pylint: disable=unused-argument
def check_supported(input_x, perm, output_y, kernel_name="dynamic_transpose"):
    """
    dynamic transpose is selected when any condition is true: \n
        -1 in input_x shape \n
        -1 in output_y shape \n
        -2 in input_x shape \n
    """
    x_shape = input_x.get("ori_shape")
    x_dtype = input_x.get("dtype")

    if x_dtype not in ("float", "float32", "int32", "uint32", "int16", "uint16", "float16"):
        return False

    if _is_dynamic(input_x, perm, output_y):
        return True

    if _by_dynamic_static_union_version(x_shape, AICORE_NUM):
        return True

    return False

