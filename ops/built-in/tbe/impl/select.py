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
select
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base

# define a VALUE, value = 1
VALUE_ONE = 1


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# pylint: disable=locally-disabled,too-many-statements,too-many-branches
def op_select_format(condition, x1, x2, y, kernel_name="select"):
    """
    1. when all input(condition, x1, x2) have the same ori_shape, ori_format,
       and the format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       the Op Select can support ND, FRACTAL_NZ, NC1HWC0 and FRACTAL_Z.\n

        for example:\n
        conditon : Tensor (shape=(16, 16, 16, 16), "NCHW")\n
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")\n
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")\n
        the Op Select can process with NC1HWC0:\n
        conditon : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")\n
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")\n
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")\n

    2. when all input(x1, x2) have the same ori_shape, ori_format, and the
       format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       and conditon is a scaler. The Op Select can support ND, FRACTAL_NZ,
       NC1HWC0 and FRACTAL_Z.\n

        for example:\n
        conditon : Tensor of (shape=(2), "NCHW")\n
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")\n
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")\n
        the Op Select can process with NC1HWC0:\n
        conditon : Tensor of (shape=(2), "NCHW")\n
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")\n
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")\n
    """
    shape_condition = condition.get("ori_shape")
    shape_x1 = x1.get("ori_shape")
    shape_x2 = x2.get("ori_shape")

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]

    format_condition = condition.get("ori_format")
    format_x1 = x1.get("ori_format")
    format_x2 = x2.get("ori_format")

    format_list = []
    if tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        dtype_list = ["float16", "float", "int32", "int8", "uint8"]
    else:
        dtype_list = ["float16", "int32", "int8", "uint8"]
    dtype_total = []
    dtype_total0 = []
    dtype_total0.append("bool")
    format_list1 = []
    # NZ+NZ ND+ND 5HD+5HD FZ+FZ
    if (len(shape_condition) != 1) or \
            (len(shape_condition) == 1 and len(shape_x1) == 1
             and len(shape_x2) == 1):
        format_list.append("ND")
        if format_condition == format_x1 == format_x2 and \
                format_x1 in format_4d_list and \
                list(shape_condition) == list(shape_x1) == list(shape_x2):
            format_list.append("FRACTAL_Z")
            format_list.append("FRACTAL_NZ")
            format_list.append("NC1HWC0")
        # the bool can not support the 6HD and FZ_3D,
        # so the select can not support the 6HD and FZ_3D, and when transdata support will open this feather
        if format_condition == format_x1 == format_x2 and \
                format_x1 in format_5d_list and \
                list(shape_condition) == list(shape_x1) == list(shape_x2):
            # do nothing now
            pass
            # modify: format_list.append("FRACTAL_Z_3D")
            # modify: format_list.append("NDC1HWC0")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list)
        dtype_total0 = dtype_total0 * len(dtype_total)
        format_list = format_list * len(dtype_list)
        input0 = util_select_op_base.gen_param(classify="input0", name="condition",
                                               datatype=",".join(dtype_total0),
                                               unknownshape_format=",".join(format_list),
                                               format=",".join(format_list))
        input1 = util_select_op_base.gen_param(classify="input1", name="x1",
                                               datatype=",".join(dtype_total),
                                               unknownshape_format=",".join(format_list),
                                               format=",".join(format_list))
        input2 = util_select_op_base.gen_param(classify="input2", name="x2",
                                               datatype=",".join(dtype_total),
                                               unknownshape_format=",".join(format_list),
                                               format=",".join(format_list))
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=",".join(dtype_total),
                                                unknownshape_format=",".join(format_list),
                                                format=",".join(format_list))
    else:
        format_list.append("ND")
        format_list1.append("ND")
        if format_x1 == format_x2:
            if len(shape_x1) == 4 and len(shape_x2) == 4 and \
                    format_x1 in format_4d_list and format_x2 in format_4d_list:
                format_list1.append("FRACTAL_NZ")
                if format_x1 in ("NHWC", "NCHW"):
                    format_list1.append("NC1HWC0")
            elif len(shape_x1) > 2 and len(shape_x2) > 2 and \
                    format_x1 in format_4d_list and format_x2 in format_4d_list:
                format_list1.append("FRACTAL_NZ")
            elif len(shape_x1) == 5 and len(shape_x2) == 5 and \
                    format_x1 in format_5d_list and format_x2 in format_5d_list:
                format_list1.append("NDC1HWC0")

        for dtype in dtype_list:
            dtype_total = dtype_total + [dtype] * len(format_list1)
        dtype_total0 = dtype_total0 * len(dtype_total)
        format_list1 = format_list1 * len(dtype_list)
        format_list = format_list * len(dtype_total)
        input0 = util_select_op_base.gen_param(classify="input0", name="condition",
                                               datatype=",".join(dtype_total0),
                                               unknownshape_format=",".join(format_list),
                                               format=",".join(format_list))
        input1 = util_select_op_base.gen_param(classify="input1", name="x1",
                                               datatype=",".join(dtype_total),
                                               unknownshape_format=",".join(format_list1),
                                               format=",".join(format_list1))
        input2 = util_select_op_base.gen_param(classify="input2", name="x2",
                                               datatype=",".join(dtype_total),
                                               unknownshape_format=",".join(format_list1),
                                               format=",".join(format_list1))
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=",".join(dtype_total),
                                                unknownshape_format=",".join(format_list1),
                                                format=",".join(format_list1))

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# pylint: disable=too-many-locals, invalid-name, unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("select")
def select_compute(condition, x1, x2, y, kernel_name="select"):
    """
    compute for select

    Parameters
    ----------
    condition: TVM tensor
        the placeholder of input condition
    x1: TVM tensor
        the placeholder of input x1
    x2: TVM tensor
        the placeholder of input x2
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "select"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape = shape_util.shape_to_list(x1.shape)
    con_shape = shape_util.shape_to_list(condition.shape)
    num_dtype = x1.dtype

    if (num_dtype in ("float32", "int32")) and \
            (not tbe_platform.api_check_support("te.lang.cce.vsel", "float32")):
        if num_dtype == "int32":
            condition = tbe.ceil(condition)
        else:
            condition = tbe.cast_to(condition, num_dtype)
        condition = tbe.broadcast(condition, shape)
        ones = tbe.broadcast(tvm.const(VALUE_ONE, dtype=num_dtype), shape, output_dtype=num_dtype)
        condition_opp = tbe.vsub(ones, condition)
        temp_x = tbe.vmul(x1, condition)
        temp_y = tbe.vmul(x2, condition_opp)
        res = tbe.vadd(temp_x, temp_y)
        return res

    if num_dtype in ("int8", "uint8", "int32"):
        if tbe_platform.api_check_support("te.lang.cce.vsel", "float32"):
            x1_dtype = "float32"
            ones = tbe.broadcast(tvm.const(VALUE_ONE, dtype="float32"), shape, output_dtype="float32")
            x1 = tbe.cast_to(x1, "float32")
            x2 = tbe.cast_to(x2, "float32")
        else:
            x1_dtype = "float16"
            ones = tbe.broadcast(tvm.const(VALUE_ONE, dtype="float16"), shape, output_dtype="float16")
            x1 = tbe.cast_to(x1, "float16")
            x2 = tbe.cast_to(x2, "float16")
    else:
        x1_dtype = num_dtype
        ones = tbe.broadcast(tvm.const(VALUE_ONE, dtype=num_dtype), shape, output_dtype=num_dtype)
    if list(con_shape) == list(shape):
        res = tbe.vsel(condition, x1, x2)
    else:
        condition = tbe.cast_to(condition, x1_dtype)
        condition = tbe.broadcast(condition, shape)
        res = tbe.vcmpsel(condition, rhs=ones, operation='eq', slhs=x1, srhs=x2)
    if num_dtype in ("int8", "uint8", "int32"):
        res = tbe.cast_to(res, num_dtype)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def select(condition, x1, x2, y, kernel_name="select"):
    """
      Selects elements from `x1` or `x2`, depending on `condition`.

      Parameters
      ----------
      condition: dict
          dict of condition, include keys(shape and dtype),
          only support int8,int32
      x1: dict
          dict of x1, only support float16, float32, int32, int8, uint8
      x2: dict
          dict of x2, only support float16, float32, int32, int8, uint8
      y: dict
          dict of output
      kernel_name: str
          cce kernel name, default value is "select"

      Returns
      -------
      None
      """
    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype").lower()
    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype").lower()
    con_shape = condition.get("shape")
    bool_dtype = condition.get("dtype").lower()
    if bool_dtype == "bool":
        bool_dtype = "int8"
    para_check.check_shape(shape_x1, param_name="x1")
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_x1, check_list, param_name="x1")

    if shape_x1 != shape_x2:
        error_detail = "Shape of tensor x1 and x2 must be equal!"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x1", \
                                                               "x2", error_detail)

    if dtype_x1 != dtype_x2:
        error_detail = "Dtype of tensor x1 and x2 must be equal!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", \
                                                               "x2", error_detail)

    x_len = len(shape_x1)
    con_shape = list(con_shape)
    if len(con_shape) == 1 and x_len != 1:
        if con_shape[0] != shape_x1[0]:
            error_detail = "Shape of tensor condition and x1 dim[0] must be equal!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "condition", \
                                                                   "x1", error_detail)
        while x_len > len(con_shape):
            con_shape += [1]
    else:
        if list(con_shape) != list(shape_x1):
            error_detail = "Shape of tensor condition and x1 must be equal!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "condition", \
                                                                   "x1", error_detail)

    con_shape, shape_x1 = shape_util.refine_shapes_for_broadcast(con_shape, shape_x1)

    flag_cloud = tbe_platform.api_check_support("te.lang.cce.vsel", "float32")
    flag_dtype = dtype_x1 in ("float32", "int32")
    if (list(con_shape) != list(shape_x1)) or \
            ((not flag_cloud) and flag_dtype):
        condition = tvm.placeholder(con_shape, name="condition", dtype=bool_dtype)
    else:
        condition = tvm.placeholder(con_shape, name="condition", dtype="bool")
    input_x1 = tvm.placeholder(shape_x1, name="input_x1", dtype=dtype_x1)
    input_x2 = tvm.placeholder(shape_x1, name="input_x2", dtype=dtype_x2)

    with tvm.target.cce():
        res = select_compute(condition, input_x1, input_x2, y, kernel_name)
        sch = tbe.auto_schedule(res)

    if list(con_shape) == list(shape_x1):
        config = {"name": kernel_name,
                  "tensor_list": [condition, input_x1, input_x2, res],
                  "bool_storage_as_1bit": False}
    else:
        config = {"name": kernel_name,
                  "tensor_list": [condition, input_x1, input_x2, res]}
    tbe.cce_build_code(sch, config)
