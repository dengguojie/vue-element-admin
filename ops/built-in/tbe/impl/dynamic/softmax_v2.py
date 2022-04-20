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
dynamic softmax_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import OpPatternMode
from impl.util.norm_pattern_adapter import NormPattern
from impl.util import util_common
from impl.util import util_frac_z as fz
from impl.util import util_select_op_base


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    FP16_MAX = tvm.const(6.0e04, dtype="float16")
    FP32_MAX = tvm.const(3.4e38, dtype="float32")


# 'pylint: disable=unused-argument
def op_select_format(input_x, output_y, axis=-1, kernel_name="softmax_v2"):
    """
    select format dynamically \n
    1.when is dynamic softmax, the formats of x and y are the same and only support ND.

        example:
        original:
        x's Tensor(shape=(16, 16, 16), "ND")
        y's Tensor(shape=(16, 16, 16), "ND")
    """
    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype="float16,float32",
                                           format="ND,ND",
                                           unknownshape_format="ND,ND")
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype="float16,float32",
                                            format="ND,ND",
                                            unknownshape_format="ND,ND")
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _is_special_cases(input_shape):
    white_list_shape = [[96, 4]]
    shape_t = list(input_shape)
    if shape_t in white_list_shape:
        return True
    return False


def check_is_axes_with_last(shape, axes):
    """
    check_is_axes_with_last
    """
    if len(axes) > 1:
        for i, _ in enumerate(axes):
            if axes[i] == len(shape) - 1:
                return True
    return False


# 'pylint:disable=too-many-locals,disable=too-many-statements,too-many-branches
@register_operator("SoftmaxV2")
def softmax_v2_compute(input_x, output_y, axis=-1, kernel_name="softmax_v2", impl_mode="high_performance"):
    """
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
       the data's axis, range == [-d, d-1]
    kernel_name: str
        cce kernel name, default value is softmax_v2

    Returns
    -------
    output: TVM tensor
        the result of softmax
    """

    dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)
    list_axis = list(axis)
    last_dim = len(input_x.shape) - 1

    attributes = input_x.op.attrs
    disable_fuse_axes = attributes["disable_fuse_axes"]
    input_format = attributes["format"].value
    ori_format = attributes["ori_format"].value
    ori_shape = shape_util.shape_to_list(attributes["ori_shape"])
    max_const = Constant.FP32_MAX if dtype == "float32" else Constant.FP16_MAX
    vcmax_flag = False

    check_axis_list = [-1, last_dim]
    for i in list_axis:
        if i in check_axis_list:
            vcmax_flag = True

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
        input_x = tbe.set_value(input_x, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                            i[list_axis[1]] > pad_c - 1), -max_const)

    if dtype == "float32" and vcmax_flag and \
        not tbe_platform.api_check_support("te.lang.cce.reduce_max", "float32"):
        data_max_input = tbe.cast_to(input_x, "float16")
        data_max_output = tbe.reduce_max(data_max_input, axis=list_axis, keepdims=True)
        data_max = tbe.cast_to(data_max_output, "float32")
    else:
        data_max = tbe.reduce_max(input_x, axis=list_axis, keepdims=True)

    if check_is_axes_with_last(shape, axis):
        tmp_shape = list(data_max.shape[:-1]) + [shape[-1]]
        data_max = tbe.broadcast(data_max, tmp_shape)
        data_max = tbe.broadcast(data_max, shape)
    else:
        data_max = tbe.broadcast(data_max, shape)
    data_subtrac = tbe.vsub(input_x, data_max)

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        data_subtrac = tbe.cast_to(data_subtrac, "float32")
        has_improve_precision = True

    tbe_product = tbe_platform.get_soc_spec("SOC_VERSION")
    if data_subtrac.dtype == "float32" and tbe_product in ("Ascend310",):
        data_subtrac = tbe.cast_to(data_subtrac, "float16")
        data_exp = tbe.vexp(data_subtrac)
        data_exp = tbe.cast_to(data_exp, "float32")
    else:
        data_exp = tbe.vexp(data_subtrac)

    if data_exp.dtype == "float16" and tbe_product in ("Ascend310",):
        data_exp = tbe.cast_to(data_exp, "float32")
        has_improve_precision = True

    if is_use_value:
        data_exp = tbe.set_value(data_exp, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                              i[list_axis[1]] > pad_c - 1), 0)
    data_expsum = tbe.reduce_sum(data_exp, list_axis, keepdims=True)

    if (tbe_product in ("Ascend910", "Ascend610", "Ascend615", "Ascend710") or
            tbe_platform.api_check_support("tik.vgatherb")
       ) and output_y.get("format") == "FRACTAL_NZ" and dtype == "float16":
        if _is_special_cases(ori_shape):
            data_expsum = tbe.vrec(data_expsum, "high_precision")
        elif impl_mode == "high_precision":
            data_expsum = tbe.vrec(data_expsum, "high_precision")
        else:
            data_expsum = tbe.vrec(data_expsum)
        
        if check_is_axes_with_last(shape, axis):
            tmp_shape = list(data_expsum.shape[:-1]) + [shape[-1]]
            data_expsum = tbe.broadcast(data_expsum, tmp_shape)
            data_expsum = tbe.broadcast(data_expsum, shape)
        else:
            data_expsum = tbe.broadcast(data_expsum, shape)
        output = tbe.vmul(data_exp, data_expsum)
    else:
        if check_is_axes_with_last(shape, axis):
            tmp_shape = list(data_expsum.shape[:-1]) + [shape[-1]]
            data_expsum = tbe.broadcast(data_expsum, tmp_shape)
            data_expsum = tbe.broadcast(data_expsum, shape)
        else:
            data_expsum = tbe.broadcast(data_expsum, shape)
        output = tbe.vdiv(data_exp, data_expsum)

    if has_improve_precision and dtype == "float16":
        output = tbe.cast_to(output, "float16")

    return output


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
        "N": [0],
        "C": [1 + offset_6hd, 4 + offset_6hd],
        "H": [2 + offset_6hd],
        "W": [3 + offset_6hd],
        "D": [1]
    }

    return dict_format_axis.get(axis_str)


# 'pylint:disable=invalid-name,too-many-locals
@register_operator("SoftmaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def softmax_v2(input_x, output_y, axis=-1, kernel_name="softmax_v2", impl_mode="high_performance"):
    """
    algorithm: softmax
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x : dict
        format: FORMAT_ND , NC1HWC0
               dtype: only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
        the data's axis.
        format: FORMAT_ND, NC1HWC0
                range == [-d, d-1]
    kernel_name : str
        cce kernel name, default value is softmax_v2
    impl_mode: str.
        high_precision or high_performance for inference, default value is OpImplMode.HIGH_PERFORMANCE.
        no need to add into ops_info file.

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    input_format = input_x.get("format")
    ori_format = input_x.get("ori_format")
    ori_shape = input_x.get("ori_shape")

    para_check.check_dtype(dtype, ("float16", "float32"), param_name="x")
    para_check.check_shape(shape, param_name="x")

    extra_params = dict()
    if axis is None:
        # when axis is None, it is binary case, go unknown axis schedule
        list_axis = NormPattern.REDUCE_UNKNOWN_MODE
        extra_params.update(NormPattern.REDUCE_SINGLE_TYPE)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_IDX, 0)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_NAME, "axes")
        operation.add_compile_info(NormPattern.REDUCE_ATTR_DTYPE, "ListInt")
    elif not isinstance(axis, int):
        list_axis = list(axis)
    else:
        list_axis = [axis]

    # only static op support special format, update axis for special format
    if not util_common.is_unknown(input_x):
        if input_format in ("NC1HWC0", "NDC1HWC0"):
            list_axis = update_5hd_axis(ori_format, list_axis, input_format)

        if fz.is_frac_z(input_x):
            list_axis = fz.to_frac_z_axis(ori_shape, list_axis)

        if input_format in ("NC1HWC0", "NDC1HWC0", "FRACTAL_NZ") and len(list_axis) == 2:
            extra_params.update({"disable_fuse_axes": [list_axis[0], list_axis[1]]})

    tensors = []
    schedules = []
    ins = classify([input_x, list_axis], OpPatternMode.NORM, extra_params)

    for idx, (x, reduce_axis) in enumerate(ins):
        with tbe.compute():
            disable_fuse_axes = []
            if "disable_fuse_axes" in extra_params:
                disable_fuse_axes = extra_params.get("disable_fuse_axes")[idx]
            shape_var_new = shape_util.variable_shape([x], op_mode="norm")[0]
            input_x = tvm.placeholder(shape_var_new,
                                      dtype=dtype,
                                      name="input_x",
                                      attrs={
                                          "ori_shape": ori_shape,
                                          "ori_format": ori_format,
                                          "format": input_format,
                                          "disable_fuse_axes": disable_fuse_axes
                                      })
            output = softmax_v2_compute(input_x, output_y, reduce_axis, kernel_name, impl_mode)
            tensors.append([input_x, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
