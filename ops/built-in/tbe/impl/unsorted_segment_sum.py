# Copyright 2020 Huawei Technologies Co., Ltd
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
unsorted_segment_sum
"""
# pylint: disable=too-many-lines
from te import platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments,too-many-statements
# pylint: disable=too-many-locals,too-few-public-methods,unused-argument
def op_select_format(x, segment_ids, num_segments, y,
                     kernel_name="unsorted_segment_sum"):
    """
    select format dynamically
    """
    segment_ids_shape = list(segment_ids.get("shape"))
    atomic_add = platform.api_check_support("tik.set_atomic_add")
    input_dtype = x.get("dtype").lower()
    if input_dtype in ("int8", "uint8") and atomic_add and len(segment_ids_shape) == 1:
        input0_dtype = "float,float"
        input0_format = "NC1HWC0,ND"
        input1_dtype = "int32,int32"
        input1_format = "ND,ND"
        input2_dtype = "int32,int32"
        input2_format = "ND,ND"
    elif input_dtype in ("int8", "uint8") and len(segment_ids_shape) == 1 and not atomic_add:
        input0_dtype = "float16,float16"
        input0_format = "NC1HWC0,ND"
        input1_dtype = "int32,int32"
        input1_format = "ND,ND"
        input2_dtype = "int32,int32"
        input2_format = "ND,ND"
    elif input_dtype not in ("int8", "uint8") and len(segment_ids_shape) == 1 and atomic_add:
        input0_dtype = "float16,float16,float,float,int32,int32"
        input0_format = "NC1HWC0,ND,NC1HWC0,ND,NC1HWC0,ND"
        input1_dtype = "int32,int32,int32,int32,int32,int32"
        input1_format = "ND,ND,ND,ND,ND,ND"
        input2_dtype = "int32,int32,int32,int32,int32,int32"
        input2_format = "ND,ND,ND,ND,ND,ND"
    elif input_dtype not in ("int8", "uint8") and len(segment_ids_shape) == 1 and not atomic_add:
        input0_dtype = "float16,float16,,int32,int32"
        input0_format = "NC1HWC0,ND,NC1HWC0,ND"
        input1_dtype = "int32,int32,int32,int32"
        input1_format = "ND,ND,ND,ND"
        input2_dtype = "int32,int32,int32,int32"
        input2_format = "ND,ND,ND,ND"
    elif input_dtype not in ("int8", "uint8") and len(segment_ids_shape) > 1 and not atomic_add:
        input0_dtype = "float16,int32"
        input0_format = "ND,ND"
        input1_dtype = "int32,int32"
        input1_format = "ND,ND"
        input2_dtype = "int32,int32"
        input2_format = "ND,ND"
    elif input_dtype not in ("int8", "uint8") and len(segment_ids_shape) > 1 and atomic_add:
        input0_dtype = "float16,int32,float"
        input0_format = "ND,ND,ND"
        input1_dtype = "int32,int32,int32"
        input1_format = "ND,ND,ND"
        input2_dtype = "int32,int32,int32"
        input2_format = "ND,ND,ND"
    elif input_dtype in ("int8", "uint8") and len(segment_ids_shape) > 1 and atomic_add:
        input0_dtype = "float"
        input0_format = "ND"
        input1_dtype = "int32"
        input1_format = "ND"
        input2_dtype = "int32"
        input2_format = "ND"
    else:
        input0_dtype = "float16"
        input0_format = "ND"
        input1_dtype = "int32"
        input1_format = "ND"
        input2_dtype = "int32"
        input2_format = "ND"

    input0 = gen_param(classify="input0", name="x",
                       datatype=input0_dtype,
                       format=input0_format,
                       unknownshape_format=input0_format)
    input1 = gen_param(classify="input1", name="segment_ids",
                       datatype=input1_dtype,
                       format=input1_format,
                       unknownshape_format=input1_format)
    input2 = gen_param(classify="input2", name="num_segments",
                       datatype=input2_dtype,
                       format=input2_format,
                       unknownshape_format=input2_format)
    output0 = gen_param(classify="output0", name="y",
                        datatype=input0_dtype,
                        format=input0_format,
                        unknownshape_format=input0_format)

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def unsorted_segment_sum(x_dict, segment_ids_dict, num_segments_dict, y_dict,
                         kernel_name="UnsortedSegmentSum"):
    """
    unsorted_segment_sum entry interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    segment_ids_dict: segment_ids shape, dtype and range
    num_segments_dict: num_segments shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of UnsortedSegmentSum op

    Returns
    -------
    compile info
    """
    pass