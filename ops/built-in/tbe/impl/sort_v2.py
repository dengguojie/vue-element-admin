# Copyright 2022 Huawei Technologies Co., Ltd
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
sort_v2
"""
from te import tik
from te.utils import para_check
import te.platform as tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from .sort_v2_910 import SortV2In910


def op_select_format(x, y, axis=-1, descending=False, kernel_name="sort_v2"):
    available_core_num = tik.Dprofile().get_aicore_num()
    if available_core_num == 32:
        dtype = "float16"
        input_format = "ND"
        output_format = "ND"
        ori_shape = x.get("ori_shape")
        ori_format = x.get("ori_format")

        input0 = gen_param(classify="input0", name="x", datatype=dtype, format=input_format)
        output0 = gen_param(classify="output0", name="y", datatype=dtype, format=output_format)

        param_list = [input0, output0]
        param_dynamic_in_json = get_dynamic_param_in_json(param_list)
        return param_dynamic_in_json
    else:
        raise RuntimeError("Unexpected version available_core_num:", available_core_num) 


# 'pylint: disable=too-few-public-methods
@tbe_platform.fusion_manager.fusion_manager.register("sort_v2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sort_v2(x, y, axis=-1, descending=False, kernel_name="sort_v2"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.
    Modify : 2020-11-11

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y): dict
        data of output
    dim(axis): int
    descending: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    available_core_num = tik.Dprofile().get_aicore_num()
    if available_core_num == 32:
        op_obj = SortV2In910(x, y, axis, descending, kernel_name)
        return op_obj.sort_v2_compute()
    else:
        raise RuntimeError("Unexpected version available_core_num:", available_core_num)
