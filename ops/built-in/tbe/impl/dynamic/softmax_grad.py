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
dynamic softmaxgrad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util import util_frac_z as fz
from impl.util import util_select_op_base


# 'pylint: disable=unused-argument
def op_select_format(softmax, grad_softmax, grad_x, axis, kernel_name="softmax_grad"):
    """
    select format dynamically \n
    1.when is dynamic softmax, the formats of x and y are the same and only support ND.

        example:
        original:
        softmax's Tensor(shape=(16, 16, 16), "ND")
        grad_softmax's Tensor(shape=(16, 16, 16), "ND")
        grad_x's Tensor(shape=(16, 16, 16), "ND")
    """
    input0 = util_select_op_base.gen_param(classify="input0", name="softmax",
                                           datatype="float16,float32",
                                           format="ND,ND",
                                           unknownshape_format="ND,ND")
    input1 = util_select_op_base.gen_param(classify="input1", name="grad_softmax",
                                           datatype="float16,float32",
                                           format="ND,ND",
                                           unknownshape_format="ND,ND")
    output0 = util_select_op_base.gen_param(classify="output0", name="grad_x",
                                            datatype="float16,float32",
                                            format="ND,ND",
                                            unknownshape_format="ND,ND")
    param_list = [input0, input1, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# 'pylint: disable=locally-disabled,unused-argument
# 'pylint: disable=unused-variable,disable=too-many-lines,disable=too-many-locals
@register_operator("SoftmaxGrad")
def softmax_grad_compute(softmax, grad_softmax, grad_x, axis,
                         kernel_name="softmax_grad", impl_mode="high_precision"):
    """
    Computes softmax gradients for a softmax operation
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
    shape = shape_util.shape_to_list(grad_softmax.shape)
    list_axis = list(axis)

    attributes = softmax.op.attrs
    disable_fuse_axes = attributes["disable_fuse_axes"]
    ori_shape = shape_util.shape_to_list(attributes["ori_shape"])
    ori_format = attributes["ori_format"].value
    input_format = attributes["format"].value
    has_improve_precision = False
    is_use_value = False

    if len(list_axis) == 2:
        if input_format in ("NC1HWC0", "NDC1HWC0"):
            is_use_value = True
            idc_list = shape_util.shape_to_list(disable_fuse_axes)
            idx_c0 = idc_list[1]
            ori_format = ori_format.upper()
            c = ori_shape[ori_format.find('C')]
            c = tbe.var('c') if c == -1 else c
            pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1
        if input_format in ("FRACTAL_NZ",):
            is_use_value = True
            idc_list = shape_util.shape_to_list(disable_fuse_axes)
            idx_c1 = idc_list[0]
            idx_c0 = idc_list[1]
            c = -1
            if (idx_c0 - idx_c1) == 2:
                c = ori_shape[-1]
            else:
                c = ori_shape[-2]
            c = tbe.var('c') if c == -1 else c
            pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1

    if is_use_value:
        softmax = tbe.set_value(softmax, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                            i[list_axis[1]] > pad_c - 1), 0)

    if impl_mode == "high_performance" and dtype == "float16" and \
        tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
        grad_softmax_fp32 = tbe.cast_to(grad_softmax, "float32")
        softmax_fp32 = tbe.cast_to(softmax, "float32")
        data_vmul = tbe.vmul(softmax_fp32, grad_softmax_fp32)
        data_sum = tbe.reduce_sum(data_vmul, axis=axis, keepdims=True)
        data_sum = tbe.cast_to(data_sum, "float16")
        data_sum_tmp = tbe.broadcast(data_sum, shape)
        data_sub = tbe.vsub(grad_softmax, data_sum_tmp)
        res = tbe.vmul(softmax, data_sub)
    else:
        if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.sum", "float32"):
            grad_softmax = tbe.cast_to(grad_softmax, "float32")
            softmax = tbe.cast_to(softmax, "float32")
            has_improve_precision = True
        data_vmul = tbe.vmul(softmax, grad_softmax)

        data_sum = tbe.reduce_sum(data_vmul, axis=axis, keepdims=True)
        data_sum_tmp = tbe.broadcast(data_sum, shape)
        data_sub = tbe.vsub(grad_softmax, data_sum_tmp)
        res = tbe.vmul(softmax, data_sub)
        if has_improve_precision:
            res = tbe.cast_to(res, "float16")

    return res


def update_5hd_axis(origin_format, list_axis, input_format):
    """
    update the axis of 5hd format
    data using for compute and schedule
    """
    if hasattr(list_axis, 'index'):
        list_axis = list_axis[0]

    axis_str = origin_format[list_axis]
    offset_6hd = 1 if input_format == "NDC1HWC0" else 0

    dict_format_axis = {
        "N": [0, ],
        "C": [1 + offset_6hd, 4 + offset_6hd],
        "H": [2 + offset_6hd, ],
        "W": [3 + offset_6hd, ],
        "D": [1, ]
    }

    return dict_format_axis.get(axis_str)


# 'pylint:disable=too-many-locals,invalid-name
@register_operator("SoftmaxGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def softmax_grad(softmax, grad_softmax, grad_x, axis=-1, kernel_name="softmax_grad", impl_mode="high_precision"):
    """
    Computes softmax gradients for a softmax operation
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

    shape = softmax.get("shape")
    grad_shape = grad_softmax.get("shape")
    dtype = softmax.get("dtype").lower()
    input_format = softmax.get("format")
    ori_format = softmax.get("ori_format")
    ori_shape = softmax.get("ori_shape")

    para_check.check_shape(shape, param_name="softmax")
    para_check.check_shape(grad_shape, param_name="grad_softmax")
    para_check.check_dtype(dtype, ("float16", "float32"), param_name="softmax")
    if not isinstance(axis, int):
        list_axis = list(axis)
    else:
        list_axis = [axis]

    if input_format in ("NC1HWC0", "NDC1HWC0"):
        list_axis = update_5hd_axis(ori_format, list_axis, input_format)

    if fz.is_frac_z(softmax):
        list_axis = fz.to_frac_z_axis(ori_shape, list_axis)

    extra_params = {}
    if input_format in ("NC1HWC0", "NDC1HWC0", "FRACTAL_NZ") and len(list_axis) == 2:
        extra_params.update({"disable_fuse_axes": [list_axis[0], list_axis[1]]})

    tensors = []
    schedules = []
    ins = classify([softmax, grad_softmax, list_axis], "norm", extra_params)

    for idx, (x, grad, reduce_axis) in enumerate(ins):
        with tbe.compute():
            disable_fuse_axes = []
            if "disable_fuse_axes" in extra_params:
                disable_fuse_axes = extra_params.get("disable_fuse_axes")[idx]
            shape_var_new, grad_shape_var_new = shape_util.variable_shape([x, grad], op_mode="norm")
            softmax = tvm.placeholder(shape_var_new, dtype=dtype, name="softmax",
                                      attrs={"ori_shape": ori_shape, "ori_format": ori_format, "format": input_format,
                                             "disable_fuse_axes": disable_fuse_axes})
            grad_softmax = tvm.placeholder(grad_shape_var_new, dtype=dtype, name="grad_softmax")
            output = softmax_grad_compute(softmax, grad_softmax, grad_x, reduce_axis, kernel_name, impl_mode)
            tensors.append([softmax, grad_softmax, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
