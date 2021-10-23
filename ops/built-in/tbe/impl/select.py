#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

VALUE_ONE = 1


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
# 'pylint: disable=locally-disabled,too-many-statements,too-many-branches
def op_select_format(condition, x1, x2, y, kernel_name="select"):
    """1.when all input(condition, x1, x2) have the same ori_shape, ori_format,
       and the format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       the Op Select can support ND, FRACTAL_NZ, NC1HWC0 and FRACTAL_Z.

        for example:
        conditon : Tensor (shape=(16, 16, 16, 16), "NCHW")
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        conditon : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

    2.when all input(x1, x2) have the same ori_shape, ori_format, and the
       format is in ["NCHW", "NHWC", "HWCN"] or ["NDHWC", "DHWCN", "NCDHW"],
       and conditon is a scaler. The Op Select can support ND, FRACTAL_NZ,
       NC1HWC0 and FRACTAL_Z.

        for example:
        conditon : Tensor of (shape=(2), "NCHW")
        x1 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        x2 : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        conditon : Tensor of (shape=(2), "NCHW")
        x1 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
        x2 : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    shape_condition = condition.get("ori_shape")
    shape_x1 = x1.get("ori_shape")
    shape_x2 = x2.get("ori_shape")

    format_4d_list = ["NCHW", "NHWC", "HWCN"]
    format_5d_list = ["NDHWC", "DHWCN", "NCDHW"]

    format_condition = condition.get("ori_format")
    format_x1 = x1.get("ori_format")
    format_x2 = x2.get("ori_format")

    format_support_flag = {("ND", "ND", "ND", "ND"): 1,
                           ("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z"): 0,
                           ("FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ"): 0,
                           ("NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0"): 0,
                           ("ND", "NC1HWC0", "NC1HWC0", "NC1HWC0"): 0,
                           ("ND", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ"): 0,
                           ("ND", "NDC1HWC0", "NDC1HWC0", "NDC1HWC0"): 0}

    # NZ+NZ ND+ND 5HD+5HD FZ+FZ
    if (len(shape_condition) != 1) or (len(shape_condition) == len(shape_x1) == len(shape_x2) == 1):
        if format_condition == format_x1 == format_x2 and format_x1 in format_4d_list \
                and list(shape_condition) == list(shape_x1) == list(shape_x2):
            format_support_flag[("FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z", "FRACTAL_Z")] = 1
            format_support_flag[("FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ")] = 1
            format_support_flag[("NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
        # the bool can not support the 6HD and FZ_3D,
        # so the select can not support the 6HD and FZ_3D, and when transdata support will open this feather
        if format_condition == format_x1 == format_x2 and format_x1 in format_5d_list\
                and list(shape_condition) == list(shape_x1) == list(shape_x2):
            # do nothing now
            pass

    elif format_x1 == format_x2:
        if len(shape_x1) == 4 and len(shape_x2) == 4 and format_x1 in ("NHWC", "NCHW"):
            format_support_flag[("ND", "NC1HWC0", "NC1HWC0", "NC1HWC0")] = 1
        if len(shape_x1) > 2 and len(shape_x2) > 2 and format_x1 in format_4d_list:
            format_support_flag[("ND", "FRACTAL_NZ", "FRACTAL_NZ", "FRACTAL_NZ")] = 1
        if len(shape_x1) == 5 and len(shape_x2) == 5 and format_x1 in format_5d_list:
            format_support_flag[("ND", "NDC1HWC0", "NDC1HWC0", "NDC1HWC0")] = 1

    # gen format and dtype
    format_list_input0 = [format_tuple[0] for format_tuple in format_support_flag if format_support_flag[format_tuple]]
    format_list_input1 = [format_tuple[1] for format_tuple in format_support_flag if format_support_flag[format_tuple]]
    format_list_input2 = [format_tuple[2] for format_tuple in format_support_flag if format_support_flag[format_tuple]]
    format_list_output = [format_tuple[3] for format_tuple in format_support_flag if format_support_flag[format_tuple]]

    dtype_x1_x2_y_list = ["float16", "int32", "int8", "uint8"]
    if tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        dtype_x1_x2_y_list.append("float")

    dtype_x1_x2_y_total_list = []
    for dtype in dtype_x1_x2_y_list:
        dtype_x1_x2_y_total_list = dtype_x1_x2_y_total_list + [dtype] * len(format_list_output)
    dtype_condition_total_list = ["bool"] * len(dtype_x1_x2_y_list) * len(format_list_output)
    len_dtype_x1_x2_y_list = len(dtype_x1_x2_y_list)
    format_input0_total_list = format_list_input0 * len_dtype_x1_x2_y_list
    format_input1_total_list = format_list_input1 * len_dtype_x1_x2_y_list
    format_input2_total_list = format_list_input2 * len_dtype_x1_x2_y_list
    format_output_total_list = format_list_output * len_dtype_x1_x2_y_list

    input0 = util_select_op_base.gen_param(classify="input0", name="condition",
                                           datatype=",".join(dtype_condition_total_list),
                                           unknownshape_format=",".join(format_input0_total_list),
                                           format=",".join(format_input0_total_list))
    input1 = util_select_op_base.gen_param(classify="input1", name="x1",
                                           datatype=",".join(dtype_x1_x2_y_total_list),
                                           unknownshape_format=",".join(format_input1_total_list),
                                           format=",".join(format_input1_total_list))
    input2 = util_select_op_base.gen_param(classify="input2", name="x2",
                                           datatype=",".join(dtype_x1_x2_y_total_list),
                                           unknownshape_format=",".join(format_input2_total_list),
                                           format=",".join(format_input2_total_list))
    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                            datatype=",".join(dtype_x1_x2_y_total_list),
                                            unknownshape_format=",".join(format_output_total_list),
                                            format=",".join(format_output_total_list))
    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=too-many-locals, invalid-name, unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("select")
def select_compute(condition, x1, x2, y, kernel_name="select"):
    """compute for select

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
        ones = tbe.broadcast(tvm.const(1, dtype=num_dtype), shape, output_dtype=num_dtype)
        condition_opp = tbe.vsub(ones, condition)
        temp_x = tbe.vmul(x1, condition)
        temp_y = tbe.vmul(x2, condition_opp)
        res = tbe.vadd(temp_x, temp_y)
        return res

    if num_dtype in ("int8", "uint8", "int32"):
        if tbe_platform.api_check_support("te.lang.cce.vsel", "float32"):
            x1_dtype = "float32"
            ones = tbe.broadcast(tvm.const(1, dtype="float32"), shape, output_dtype="float32")
            x1 = tbe.cast_to(x1, "float32")
            x2 = tbe.cast_to(x2, "float32")
        else:
            x1_dtype = "float16"
            ones = tbe.broadcast(tvm.const(1, dtype="float16"), shape, output_dtype="float16")
            x1 = tbe.cast_to(x1, "float16")
            x2 = tbe.cast_to(x2, "float16")
    else:
        x1_dtype = num_dtype
        ones = tbe.broadcast(tvm.const(1, dtype=num_dtype), shape, output_dtype=num_dtype)
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
    """Selects elements from `x1` or `x2`, depending on `condition`.

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
