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
bias_add_grad
"""
# pylint: disable=locally-disabled,invalid-name
# pylint: disable=unnecessary-comprehension,global-statement
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te import tvm
from te import platform as tbe_platform
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.op_utils import check_op_params
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_format
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_ATTR_STR
from te.lang.base.operation import add_compile_info
from te.utils.error_manager import error_manager_vector
from te.utils import shape_util
from topi.cce import util as cce_util


REDUCE_LIST = None


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-branches
# pylint: disable=locally-disabled,too-many-locals
def _infer_axes(input_data_format, data_format, shape):
    """
    To infer sum operate axis by input_data format and data_format
    to keep compute Architecture, so use global parameter send variable
    Parameters:
    ----------
    input_data_format: str
        op's input data format
    data_format: str
        'NCHW' or 'NHWC'
    shape : tuple or list
        the input data shape

    Returns
    -------
    g_shape_list. list
    """
    g_shape_list = []
    if input_data_format == 'FRACTAL_NZ':
        if data_format == "NCHW":
            if len(shape) == 4:
                for i in range(-1 * len(shape), 0):
                    if i not in (-1, -4):
                        g_shape_list += [i + len(shape)]
            elif len(shape) == 5:
                for i in range(-1 * len(shape), 0):
                    if i not in (-2, -3):
                        g_shape_list += [i + len(shape)]
            else:
                g_shape_list.append(0)
                for i in range(2, len(shape)):
                    g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 4:
                raise RuntimeError("cce_bias_add_grad_nz_2_nhwc only support shape larger than 4D")
            for i in range(-1 * len(shape), 0):
                if i not in (-1, -4):
                    g_shape_list += [i + len(shape)]
    elif input_data_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NC1HWC0", "NDC1HWC0"):
        if input_data_format == "FRACTAL_Z":
            # mean format is FRACTAL_Z, shape is C1HWNiNoC0
            g_shape_list = [1, 2, 3, 4]
        elif input_data_format == "FRACTAL_Z_3D":
            # mean format is FRACTAL_Z_3D, shape is DC1HWNiNoC0
            g_shape_list = [0, 2, 3, 4, 5]
        elif input_data_format == "NC1HWC0":
            # mean format is NC1HWC0, shape is NC1HWC0
            g_shape_list = [0, 2, 3]
        elif input_data_format == "NDC1HWC0":
            # mean format is NDC1HWC0, shape is NDC1HWC0
            g_shape_list = [0, 1, 3, 4]
    else:
        if data_format == "NCHW":
            g_shape_list = [0]
            for i in range(2, len(shape)):
                g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 2:
                raise RuntimeError("cce_bias_add_grad only support shape larger than 2D")
            g_shape_list = [x for x in range(len(shape) - 1)]

    return g_shape_list


def bias_add_grad_compute(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x: TVM tensor
        the placeholder of y input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad

    Returns
    -------
    TVM tensor by bias add grad
    """
    dtype = x.dtype
    y_dtype = y.get("dtype").lower()

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.sum", "float32"):
        x = tbe.cast_to(x, "float32")

    result = tbe.sum(x, REDUCE_LIST)
    result = tbe.cast_to(result, y_dtype)

    return result


@tbe_base.register_operator("BiasAddGrad")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_STR, KERNEL_NAME)
def bias_add_grad(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x : dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad
    Returns
    -------
    None
    """
    shape = x.get("shape")
    if len(shape) < 2:
        error_detail = "cce_bias_add_grad only support shape larger than 2D"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    dtype = x.get("dtype").lower()
    data_format = data_format.upper()
    check_list = ("float16", "float32")
    check_dtype(dtype, check_list, param_name="x")
    data_format_tuple = ("NCHW", "NHWC")
    input_data_format = x.get("format").upper()
    check_format(data_format, data_format_tuple, param_name="x")
    x["rel_pos_to_reduce"] = "before"

    g_shape_list = _infer_axes(input_data_format, data_format, shape)
    add_compile_info("_ori_axis", g_shape_list)
    input_axis = {"shape": [len(g_shape_list), ], "value": g_shape_list, "rel_pos_to_reduce": "axis"}
    global REDUCE_LIST
    ins = classify([x, input_axis], Mode.REDUCE)
    schedules, tensors = [], []
    for (_x, axes) in ins:
        with tbe_base.compute():
            shape_x = shape_util.variable_shape([_x, axes], op_mode="reduce")[0]
            input_data = tvm.placeholder(shape_x, name="input_data", dtype=dtype)
            REDUCE_LIST = cce_util.axis_check(len(shape_x), axes.get("value"))

            res = bias_add_grad_compute(input_data, y, data_format, kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
