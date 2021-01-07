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
floor_div
"""
import te.lang.cce as tbe
from te import tvm
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector

# pylint: disable=locally-disabled,unused-argument
# pylint: disable=too-many-locals
@tbe_platform.fusion_manager.fusion_manager.register("floor_div")
def floor_div_compute(input_x, input_y, output_z, kernel_name='floor_div'):
    """
       floordiv compute
       calculating data's floordiv, res =floor(x / y)

       Parameters
       ----------
       input_x: TVM tensor
           the placeholder of input_x
       input_y: TVM tensor
           the placeholder of input_y
       output_z: dict
           dict with keys(shape and dtype) of output
       kernel_name: str
           kernel name, default value is "floordiv"

       Returns
       -------
       res: TVM tensor
           the result of floordiv compute
    """
    input_x_shape = shape_util.shape_to_list(input_x.shape)
    input_y_shape = shape_util.shape_to_list(input_y.shape)
    shape_list = shape_util.broadcast_shapes(input_x_shape, input_y_shape,
                                             param_name_input1="input_x",
                                             param_name_input2="input_y")

    if input_x.dtype != 'float16' and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv",
                                                    "float32"):
        cast_x = tbe.cast_to(input_x, 'float32')
        cast_y = tbe.cast_to(input_y, 'float32')
        broadcast_x = tbe.broadcast(cast_x, shape_list[2])
        broadcast_y = tbe.broadcast(cast_y, shape_list[2])
    else:
        broadcast_x = tbe.broadcast(input_x, shape_list[2])
        broadcast_y = tbe.broadcast(input_y, shape_list[2])

    div_res = tbe.vdiv(broadcast_x, broadcast_y)
    floor_res = tbe.floor(div_res)
    res = tbe.cast_to(floor_res, input_x.dtype)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def floor_div(input_x, input_y, output_z, kernel_name="floor_div"):
    """
      algorithm: floordiv
      calculating data's floordiv, res =floor(x / y)

      Parameters
      ----------
      input_x: dict
          dict with keys(shape and dtype) of input_x
      input_y: dict
          dict with keys(shape and dtype) of input_y
      output_z: dict
          dict with keys(shape and dtype) of output
      kernel_name: str
          kernel name, default value is "floordiv"

      Returns
      -------
      None
    """
    # check dtype of input_x/input_y
    input_dtype_x = input_x.get("dtype").lower()
    input_dtype_y = input_y.get("dtype").lower()
    check_list = ('int8', 'uint8', 'int32', 'float16', 'float32')
    para_check.check_dtype(input_dtype_x, check_list, param_name="input_x")
    if input_dtype_x != input_dtype_y:
        error_detail = "dtype of input_x and input_y should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_x", "input_y", \
                                                               error_detail)

    # check shape of input_x/input_y
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")
    shape_list = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                             param_name_input2="input_y")

    # compute result for floordiv() with floordiv_compute()
    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_list[0],
                                                   shape_list[1])
    data_x = tvm.placeholder(shape_x, dtype=input_dtype_x, name='data_x')
    data_y = tvm.placeholder(shape_y, dtype=input_dtype_y, name='data_y')
    res = floor_div_compute(data_x, data_y, output_z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    tbe.cce_build_code(sch, config)
