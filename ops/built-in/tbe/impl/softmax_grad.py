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
softmax_grad
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
import impl.dynamic as dyn_impl
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=locally-disabled,unused-argument
# 'pylint: disable=unused-variable
@tbe_platform.fusion_manager.fusion_manager.register("softmax_grad")
def softmax_grad_compute(softmax, grad_softmax, grad_x, axis,
                         kernel_name="softmax_grad"):
    """Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: TVM tensor
        the placeholder of first input data
    grad_softmax: TVM tensor
        the placeholder of second input data
    grad_x: dict
        the dict of output data
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        cce kernel name, default value is "softmax_grad"

    Returns
    -------
    res: TVM tensor
        the result of softmax_grad_compute
    """
    dtype = softmax.dtype
    shape_input1 = shape_util.shape_to_list(softmax.shape)
    shape_input2 = shape_util.shape_to_list(grad_softmax.shape)
    has_improve_precision = False
    if list(shape_input1) != list(shape_input2):
        shape_input1, shape_input2, shape =\
            shape_util.broadcast_shapes(shape_input1, shape_input2,
                                        param_name_input1="softmax",
                                        param_name_input2="grad_softmax")
        softmax = tbe.broadcast(softmax, shape, dtype)
        grad_softmax = tbe.broadcast(grad_softmax, shape, dtype)

    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        softmax = tbe.cast_to(softmax, "float32")
        grad_softmax = tbe.cast_to(grad_softmax, "float32")
        has_improve_precision = True
    data_vmul = tbe.vmul(softmax, grad_softmax)
    data_sum = tbe.sum(data_vmul, axis=axis, keepdims=True)
    if list(shape_input1) != list(shape_input2):
        data_sum_tmp = tbe.broadcast(data_sum, shape)
    else:
        data_sum_tmp = tbe.broadcast(data_sum, shape_input2)
    data_sub = tbe.vsub(grad_softmax, data_sum_tmp)
    res = tbe.vmul(softmax, data_sub)
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def update_5hd_axis(origin_format, axis, input_format):
    """
    update the axis of 5hd format
    data using for compute and schedule
    """
    if hasattr(axis, 'index'):
        axis = axis[0]

    axis_str = origin_format[axis]
    offset_6hd = 1 if input_format == "NDC1HWC0" else 0

    dict_format_axis = {
        "N": [0, ],
        "C": [-4, -1],
        "H": [2 + offset_6hd, ],
        "W": [3 + offset_6hd, ],
        "D": [1, ]
    }

    return dict_format_axis.get(axis_str)


# 'pylint: disable=variable_type_changed
# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def softmax_grad(softmax, grad_softmax, grad_x, axis=-1, kernel_name="softmax_grad"):
    """Computes softmax gradients for a softmax operation
    The calculation formula is as follows :
    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

    Parameters
    ----------
    softmax: dict
        shape and dtype of first input, only support float16, float32
    grad_softmax: dict
        shape and dtype of second input, only support float16, float32
    grad_x: dict
        shape and dtype of output data, should be same shape and type as input
    axis: int, list or tuple .
        the first axis to reduce, may be negative to index from the end
        (e.g., -1 for the last axis).
        axis may be int or list(e.g. [1,2])
        if true, retains reduced dimensions with length 1,
        default value is -1
    kernel_name: str
        kernel name, default value is "softmax_grad"

    Returns
    -------
    None
    """
    dtype_softmax = softmax.get("dtype")
    input_format = softmax.get("format")
    ori_shape = softmax.get("ori_shape")
    shape_softmax = softmax.get("shape")
    shape_grad_softmax = grad_softmax.get("shape")
    if input_format == "NC1HWC0":
        if len(ori_shape) == 2:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], 1]
            softmax["ori_shape"] = new_ori_shape
            grad_softmax["ori_shape"] = new_ori_shape
            if not isinstance(axis, int):
                axis = list(axis)
            if not hasattr(axis, 'index'):
                if axis >= 0:
                    axis = axis + 1
                else:
                    axis = axis - 1
            else:
                if axis[0] >= 0:
                    axis[0] = axis[0] + 1
                else:
                    axis[0] = axis[0] - 1
        if len(ori_shape) == 3:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], ori_shape[2]]
            softmax["ori_shape"] = new_ori_shape
            grad_softmax["ori_shape"] = new_ori_shape
            if not isinstance(axis, int):
                axis = list(axis)
            if not hasattr(axis, 'index'):
                if axis >= 0:
                    axis = axis + 1
            else:
                if axis[0] >= 0:
                    axis[0] = axis[0] + 1

    if input_format in ("NC1HWC0", "NDC1HWC0", "FRACTAL_NZ"):
        context = tbe_context.op_context.get_context()
        if context is not None:
            context.set_op_mode("static")
            dyn_impl.softmax_grad(softmax, grad_softmax, grad_x, axis, kernel_name)
        else:
            with tbe_context.op_context.OpContext("static"):
                dyn_impl.softmax_grad(softmax, grad_softmax, grad_x, axis, kernel_name)
        return

    if not isinstance(axis, int):
        axis = list(axis)

    shape_util.compare_tensor_dict_key(softmax, grad_softmax, "dtype")
    para_check.check_shape(shape_softmax, param_name="softmax")
    para_check.check_shape(shape_grad_softmax, param_name="grad_softmax")

    check_list = ("float16", "float32")
    input_dtype = dtype_softmax.lower()

    para_check.check_dtype(input_dtype, check_list, param_name="softmax")
    if list(shape_softmax) != list(shape_grad_softmax):
        shape_softmax, shape_grad_softmax, shape_max = \
            shape_util.broadcast_shapes(shape_softmax, shape_grad_softmax, param_name_input1="softmax",
                                        param_name_input2="grad_softmax")

    if input_format in ("NC1HWC0", "NDC1HWC0"):
        axis = update_5hd_axis(softmax.get("ori_format"), axis, input_format)
    axis = shape_util.axis_check(len(shape_softmax), axis)

    shape_softmax, axis = shape_util.shape_refine(list(shape_softmax), axis)
    shape_softmax, axis = shape_util.simplify_axis_shape(shape_softmax, axis)
    shape_grad_softmax = shape_softmax
    softmax = tvm.placeholder(shape_softmax, name="softmax", dtype=input_dtype)
    grad_softmaxgrad = tvm.placeholder(shape_grad_softmax,
                                       name="grad_softmaxgrad",
                                       dtype=input_dtype)

    res = softmax_grad_compute(softmax, grad_softmaxgrad, grad_x, axis,
                               kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [softmax, grad_softmaxgrad, res]}
    tbe.cce_build_code(sch, config)
