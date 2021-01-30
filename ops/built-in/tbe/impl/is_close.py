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
is_close
"""
import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check


SHAPE_SIZE_LIMIT = 2147483648
@fusion_manager.register("is_close")
#pylint: disable=unused-argument
def is_close_compute(input_x1, input_x2, output_y, rtol=1e-05, atol=1e-08, equal_nan=False, kernel_name="is_close"):
    """
    is_close function compute
    :param input_x1: the shape and dtype of input tensor
    :param input_x2: the shape and dtype of input tensor
    :param output_y: the shape and dtype of output tensor
    :param rtol: (float, optional) - relative tolerance. Default: 1e-05
    :param atol: (float, optional) - absolute tolerance. Default: le-08
    :param equal_nan: (bool, optional) - if True, then two NaN s will be considered equal. Default: False
    :param kernel_name: cce kernel name, default value is 'is_close'
    :return: value of is_close
    """
    lhs = tbe.vabs(tbe.vsub(input_x1, input_x2))
    temp = tbe.vabs(tbe.vmuls(input_x2, rtol))
    rhs = tbe.vadds(temp, atol)
    return tbe.vcmp(lhs, rhs, operation='le')


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, 
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, 
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, 
                            para_check.KERNEL_NAME)
def is_close(input_x1, input_x2, output_y, rtol=1e-05, atol=1e-08, equal_nan=False, kernel_name="is_close"):
    """
    is_close function compute
    :param input_x1: the shape and dtype of input tensor
    :param input_x2: the shape and dtype of input tensor
    :param output_y: the shape and dtype of output tensor
    :param rtol: (float, optional) - relative tolerance. Default: 1e-05
    :param atol: (float, optional) - absolute tolerance. Default: le-08
    :param equal_nan: (bool, optional) - if True, then two NaN s will be considered equal. Default: False
    :param kernel_name: cce kernel name, default value is 'is_close'
    :return: value of is_close
    """
    shape_x1 = input_x1.get("shape")
    shape_x2 = input_x2.get("shape")
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x1)
    para_check.check_shape_rule(shape_x2)
    para_check.check_shape_size(shape_x1, SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(shape_x2, SHAPE_SIZE_LIMIT)

    input_data_type = input_x1.get("dtype").lower()

    check_tuple = ("float16", "float32", "int32")
    para_check.check_dtype_rule(input_data_type, check_tuple)
    data_x1 = tvm.placeholder(shape_x1, name="data_1", dtype=input_data_type)
    data_x2 = tvm.placeholder(shape_x2, name="data_2", dtype=input_data_type)
    if input_data_type == "float16":
        data_x1_trans = tbe.cast_to(data_x1, "float32")
        data_x2_trans = tbe.cast_to(data_x2, "float32")
        res = is_close_compute(data_x1_trans, data_x2_trans, output_y, rtol, atol, equal_nan, kernel_name)
    else:
        res = is_close_compute(data_x1, data_x2, output_y, rtol, atol, equal_nan, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x1, data_x2, res],
              "bool_storage_as_1bit": False}

    tbe.cce_build_code(schedule, config)