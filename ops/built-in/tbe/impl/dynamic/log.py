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
log
"""
import math
from functools import reduce as reduceIns
from te.utils.error_manager import error_manager_vector
import te.lang.cce as tbe
from te import tvm
from te import platform as tbe_platform
import te.lang.base as tbe_base
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_dtype
from te.utils.op_utils import check_op_params
from te.utils.op_utils import variable_shape
from te.utils.op_utils import OPTION_ATTR_FLOAT
from topi import generic
from impl.util.platform_adapter import register_operator


def _isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    Return True if the values a and b are close to each other and False otherwise
    See math.isclose for further understanding.
    Parameters
    ----------
    valuex : value x
    valuey : value y
    rel_tol : relative tolerance
    abs_tol : absolute tolerance
    Returns
    -------
    bool
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)


# pylint: disable=too-many-arguments,unused-argument,too-many-locals
def log_compute(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="log"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    base: float
        base default -1.0
    scale: float
        scale default 1.0
    shift: float
        shift default 0.0
    kernel_name : str
        kernel name, default value is "log"

    Returns
    -------
    output tensor
    """
    log_base = 1.0 if _isclose(base, -1.0) else math.log(base)
    base_scale = 1.0 / log_base

    dtype = input_x.dtype
    f322f16_support = tbe_platform.api_check_support("te.lang.cce.cast_to", "f322f16")
    if dtype == "float32" and not f322f16_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'input_x', "float16", dtype)

    if _isclose(scale, 1.0) and _isclose(shift, 0.0):
        if dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
            input_x = tbe.cast_to(input_x, "float16")
        x_log = tbe.vlog(input_x)
        if dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
            x_log = tbe.cast_to(x_log, "float32")
    else:
        x_scale_and_shift = input_x
        if not _isclose(scale, 1.0):
            x_scale_and_shift = tbe.vmuls(input_x, tvm.const(scale, dtype=dtype))

        if not _isclose(shift, 0.0):
            x_scale_and_shift = tbe.vadds(x_scale_and_shift, tvm.const(shift, dtype=dtype))

        if dtype == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
            x_scale_and_shift = tbe.cast_to(x_scale_and_shift, "float16")

        x_log = tbe.vlog(x_scale_and_shift)

        if dtype == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
            x_log = tbe.cast_to(x_log, "float32")

    if not _isclose(base_scale, 1.0):
        res = tbe.vmuls(x_log, tvm.const(base_scale, dtype=dtype))
        return res
    return x_log


@register_operator("Log")
@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, OPTION_ATTR_FLOAT,
                 OPTION_ATTR_FLOAT, OPTION_ATTR_FLOAT, KERNEL_NAME)
def log(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="log"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    base: float
        base default -1.0
    scale: float
        scale default 1.0
    shift: float
        shift default 0.0
    kernel_name : str
        kernel name, default value is "log"

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype")
    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    check_dtype(input_dtype, check_list, param_name="input_x")

    if base <= 0 and (not _isclose(base, -1.0)):
        error_manager_vector.raise_err_input_value_invalid(kernel_name, 'base', "strictly positive or -1", base)

    schedules, tensors = [], []
    ins = classify([input_x], Mode.ELEWISE)
    for (_input_x,) in ins:
        with tbe_base.compute():
            x_shape = variable_shape([_input_x])
            fuseshape = [1]
            fuseshape[0] = reduceIns(lambda x, y: x * y, x_shape[0])
            data_input = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_input")
            res = log_compute(data_input, output_y, base, scale, shift, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = generic.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "need_build": True}
    tbe.build(schedules, config)
