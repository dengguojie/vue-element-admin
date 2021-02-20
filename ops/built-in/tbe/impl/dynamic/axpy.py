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
dynamic axpy
"""
import te.lang.cce
import te.lang.base as tbe_base
from te import tvm
from te import platform as tbe_platform
from te.utils import shape_util
from te.utils import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

# constant, value is 16
SIZE_SIXTEEN = 16


# pylint: disable=unused-argument, too-many-nested-blocks
# pylint: disable=invalid-name,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-boolean-expressions
def _add_check_format(x, y):
    """
    check format of add

    Parameters
    ----------
    x: dict
    y: dict

    Returns
    -------
    format_pattern: int
    """
    shape1 = x.get("shape")
    shape2 = y.get("shape")
    list_format = [x.get("format"), y.get("format")]
    shape1 = shape_util.scalar2tensor_one(shape1)
    shape2 = shape_util.scalar2tensor_one(shape2)
    list_shape = [shape1, shape2]

    format_list = ("ND", "NCHW", "NHWC")
    if (list_format[0] == "FRACTAL_NZ" and len(list_shape[1]) == 1 \
            and list_shape[1][0] % SIZE_SIXTEEN == 0) \
            or (list_format[1] == "FRACTAL_NZ" and len(list_shape[0]) == 1 \
            and list_shape[0][0] % SIZE_SIXTEEN == 0):
        format_pattern = 3
    elif list_format[0] == "FRACTAL_NZ" and list_format[1] in format_list \
            and (len(shape2) != 1 or (len(shape2) == 1 and shape2[0] != 1)):
        format_pattern = 1
    elif list_format[0] in format_list and list_format[1] == "FRACTAL_NZ" \
            and (len(shape1) != 1 or (len(shape1) == 1 and shape1[0] != 1)):
        format_pattern = 2
    else:
        format_pattern = 0

    return format_pattern


def _infer_shape(format_pattern, x, y):
    """
    infer shape for x and y

    Parameters
    ----------
    format_pattern: format type
    x: dict
    y: dict

    Returns
    -------
    shape_x: shape of x
    shape_y: shape of y
    """
    shape_x = x.get("shape")
    shape_y = y.get("shape")
    ori_shape_x = x.get("ori_shape")
    ori_shape_y = y.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_y = shape_util.scalar2tensor_one(shape_y)

    if format_pattern == 1:
        ori_shape_x, shape_y, _ = shape_util.broadcast_shapes(ori_shape_x, shape_y,
                                                              param_name_input1='x',
                                                              param_name_input2='y')
        if shape_y[-2] == 1 and shape_y[-1] == ori_shape_x[-1]:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-3] = 1
            shape_y[-1] = shape_x[-1]
            shape_y[-4] = shape_x[-4]
        elif shape_y[-2] == ori_shape_x[-2] and shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)
            shape_y[-4] = 1
            shape_y[-2] = shape_x[-2]
            shape_y[-3] = shape_x[-3]
        elif shape_y[-2] == shape_y[-1] == 1:
            shape_y.append(1)
            shape_y.append(1)

    elif format_pattern == 2:
        shape_x, ori_shape_y, _ = shape_util.broadcast_shapes(shape_x, ori_shape_y,
                                                              param_name_input1='x',
                                                              param_name_input2='y')
        if shape_x[-2] == 1 and shape_x[-1] == ori_shape_y[-1]:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-3] = 1
            shape_x[-1] = shape_y[-1]
            shape_x[-4] = shape_y[-4]
        elif shape_x[-2] == ori_shape_y[-2] and shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
            shape_x[-4] = 1
            shape_x[-2] = shape_y[-2]
            shape_x[-3] = shape_y[-3]
        elif shape_x[-2] == shape_x[-1] == 1:
            shape_x.append(1)
            shape_x.append(1)
    elif format_pattern == 3:
        def _get_new_shape(_nz_shape, _nd_shape):
            _nd_new_shape = [1 for _ in _nz_shape]
            _nd_new_shape[-1] = _nz_shape[-1]
            _nd_new_shape[-4] = _nz_shape[-4]

            return _nz_shape, _nd_new_shape

        if len(shape_y) == 1:
            shape_x, shape_y = _get_new_shape(shape_x, shape_y)
        else:
            shape_y, shape_x = _get_new_shape(shape_y, shape_x)

    return shape_x, shape_y


@register_operator_compute("Axpy", op_mode="dynamic", support_fusion=False)
def axpy_compute(x1, x2, y, alpha, kernel_name="axpy"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of input_x
    x2 : TVM tensor
        the placeholder of x2
    y : dict
        dict of y, include keys(shape and dtype)
    alpha : float
        scalar of mul-factor
    kernel_name : str
        kernel name, default value is "axpy"

    Returns
    -------
    output tensor
    """
    # broadcast
    shape_x = te.lang.cce.util.shape_to_list(x1.shape)
    shape_y = te.lang.cce.util.shape_to_list(x2.shape)
    dtype = x1.dtype.lower()

    # neg_1_axis_flag
    neg_1_axis_flag = 0
    if shape_x != shape_y:
        # if shape not equal, then apply broadcast.
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                                  shape_y,
                                                                  param_name_input1='x1',
                                                                  param_name_input2='x2')

        for i in range(len(shape_x) - 1):
            if shape_x[i] != shape_y[i]:
                neg_1_axis_flag = 1
                break
        x1 = te.lang.cce.broadcast(x1, shape_max)
        x2 = te.lang.cce.broadcast(x2, shape_max)

    # start the main logic
    if tbe_platform.cce_conf.get_soc_spec("SOC_VERSION") == "Ascend910":
        if dtype in ("float16", "float32"):
            # fp16 or fp32
            if neg_1_axis_flag:
                res_muls = te.lang.cce.vmuls(x2, alpha)
                res = te.lang.cce.vadd(x1, res_muls)
            else:
                res = te.lang.cce.vaxpy(x2, x1, tvm.const(alpha, dtype=dtype))
        else:
            # int32
            if alpha != 1:
                # add+muls use fp32
                to_type = "float32"
                input_x_cast = te.lang.cce.cast_to(x1, to_type)
                input_y_cast = te.lang.cce.cast_to(x2, to_type)

                if neg_1_axis_flag:
                    res_muls = te.lang.cce.vmuls(x2, alpha)
                    res_tmp = te.lang.cce.vadd(x1, res_muls)
                else:
                    res_tmp = te.lang.cce.vaxpy(input_y_cast, input_x_cast,
                                                tvm.const(alpha, dtype=to_type))

                res = te.lang.cce.cast_to(res_tmp, dtype)

            else:
                # if alpha == 1
                res = te.lang.cce.vadd(x2, x1)
    else:
        if dtype in ("float16", "float32"):
            # fp16 or fp32
            res_muls = te.lang.cce.vmuls(x2, alpha)
            res = te.lang.cce.vadd(x1, res_muls)
        else:
            # int32
            if alpha != 1:
                # add+muls use fp32
                to_type = "float32"
                input_x1_cast = te.lang.cce.cast_to(x1, to_type)
                input_x2_cast = te.lang.cce.cast_to(x2, to_type)

                res_muls = te.lang.cce.vmuls(input_x2_cast, alpha)
                res_tmp = te.lang.cce.vadd(input_x1_cast, res_muls)

                res = te.lang.cce.cast_to(res_tmp, dtype)
            else:
                # if alpha == 1
                res = te.lang.cce.vadd(x2, x1)

    return res


@register_operator("Axpy")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def axpy(x1, x2, y, alpha, kernel_name="axpy"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input_x
    x2 : dict
        shape and dtype of input_y
    y : dict
        shape and dtype of output, should be same shape and type as input
    alpha : float
        scalar apply to input_y:input_y*alpha
    kernel_name : str
        kernel name, default value is "axpy"

    Returns
    -------
    None
    """

    format_pattern = _add_check_format(x1, x2)
    shape_x1, shape_x2 = _infer_shape(format_pattern, x1, x2)

    # check shape
    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    shape_x2 = shape_util.scalar2tensor_one(shape_x2)
    x1["shape"] = shape_x1
    x2["shape"] = shape_x2

    # check dtype
    dtype_list = ("float16", "float32", "int32")
    dtype_x1 = x1.get("dtype").lower()
    para_check.check_dtype(dtype_x1, dtype_list)
    dtype_x2 = x2.get("dtype").lower()
    para_check.check_dtype(dtype_x2, dtype_list)

    # produce shapes
    ins = tbe_base.classify([x1, x2], tbe_base.Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x1, _input_x2) in ins:
        with tbe_base.compute():
            shape_x1, shape_x2 = \
                shape_util.variable_shape([_input_x1, _input_x2])
            shape_x1, shape_x2 = shape_util.refine_shapes_for_broadcast(shape_x1, shape_x2)

            data_input_x1 = tvm.placeholder(shape_x1, name="data_input_x1", dtype=dtype_x1)
            data_input_x2 = tvm.placeholder(shape_x2, name="data_input_x2", dtype=dtype_x2)
            res = axpy_compute(data_input_x1, data_input_x2, y, alpha, kernel_name)
            tensors.append((data_input_x1, data_input_x2, res))
        with tvm.target.cce():
            schedule = te.lang.cce.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}

    te.lang.cce.build(schedules, config)
