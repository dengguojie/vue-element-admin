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
inplace_update
"""
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.dynamic.unsorted_segment_sum import UnsortedSegmentSum
from impl.dynamic.unsorted_segment_sum_no_atomic import UnsortedSegmentSumNoAtomoic

# 'pylint: disable=unused-argument
def check_supported(x, segment_ids, y, kernel_name="unsorted_segment_sum"):
    """
    dynamic -1 support
    segment_ids int64 not support
    static shape x_shape ends with 1 or lens equals 1 not support
    temporary support x_dtype of "float32" in compilestatic process
    """
    shapex = x.get("ori_shape")
    shapeid = segment_ids.get("ori_shape")
    shapey = y.get("ori_shape")
    id_dtype = segment_ids.get("dtype").lower()
    x_dtype = x.get("dtype").lower()
    dynamic_x = True
    dynamic_id = True
    dynamic_seg = True
    dynamic_y = True

    if id_dtype != "int32":
        reason = "the segment_ids's dytpe not equeal int32, segment_ids_dtype=%s" % id_dtype
        return False, reason
    if x_dtype in ("int8", "uint8"):
        reason = "the x_dtype in (\"int8\", \"uint8\"), x_dtype=%s" % x_dtype
        return False, reason

    return True, ""

def op_select_format(x, segment_ids, y, kernel_name="unsorted_segment_sum"):
    """
    select format dynamically
    """
    segment_ids_shape = list(segment_ids.get("ori_shape"))
    atomic_add = tbe_platform.api_check_support("tik.set_atomic_add")
    if not atomic_add:
        input0_dtype = "float16,int32"
        input0_format = "ND,ND"
        input1_dtype = "int32,int32"
        input1_format = "ND,ND"
    else:
        input0_dtype = "float16,int32,float"
        input0_format = "ND,ND,ND"
        input1_dtype = "int32,int32,int32"
        input1_format = "ND,ND,ND"
    input0 = gen_param(classify="input0", name="x",
                       datatype=input0_dtype,
                       format=input0_format,
                       unknownshape_format=input0_format)
    input1 = gen_param(classify="input1", name="segment_ids",
                       datatype=input1_dtype,
                       format=input1_format,
                       unknownshape_format=input1_format)
    output0 = gen_param(classify="output0", name="y",
                        datatype=input0_dtype,
                        format=input0_format,
                        unknownshape_format=input0_format)

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json

# pylint: disable=locally-disabled,invalid-name,unused-argument,too-many-branches
# pylint: disable=superfluous-parens
@register_operator("SegmentSum")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def segment_sum(x, segment_ids, y, kernel_name="segment_sum"):
    """
    Updates specified rows with values in v

    Parameters
    ----------
    x : dict
        shape and dtype of input tensor x
    segment_id: dict
    y : dict
        shape and dtype of output tensor
    kernel_name : str
        kernel name, default value is "segment_sum"

    Returns
    -------
    tik_instance
    """
    num_segments_dict = {"dtype": "int32"}
    x_dtype = x.get("dtype").lower()
    x_dtype_check_list = ("float32", "float16", "int32")
    para_check.check_dtype(x_dtype, x_dtype_check_list, param_name="x")

    segment_ids_dtype = segment_ids.get("dtype").lower()
    segment_ids_dtype_check_list = ("int32")
    para_check.check_dtype(segment_ids_dtype, segment_ids_dtype_check_list, param_name="segment_ids")

    y_dtype = y.get("dtype").lower()
    para_check.check_dtype(y_dtype, x_dtype_check_list, param_name="y")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x", "y", x_dtype, y_dtype)
    if x_dtype != "float32":
        obj = UnsortedSegmentSumNoAtomoic(x, segment_ids, num_segments_dict,
                                          y, kernel_name, opname="segment_sum")
        obj.unsorted_segment_sum()
    else:
        obj = UnsortedSegmentSum(x, segment_ids, num_segments_dict, y, kernel_name, opname="segment_sum")
        obj.unsorted_segment_sum()
