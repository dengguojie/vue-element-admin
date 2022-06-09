# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic bn_training_update_grad
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import ReduceInput
from impl.util.util_select_op_base import ReduceOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import OpAttr


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-locals, too-many-statements,redefined-builtin
def get_op_support_info(grads,
                        x,
                        batch_mean,
                        batch_variance,
                        diff_scale,
                        diff_offset,
                        epsilon,
                        kernel_name="bn_training_update_grad"):
    """
    get_op_support_info
    """
    format_grads = grads.get("format").upper()
    if format_grads == "NC1HWC0" or format_grads == "NCHW":
        axis_split_matrix = [[SplitInput([0, [1], [-1], [-1]], [1, [1], [-1], [-1]], [2, [1], [-1], [-1]], \
                                         [3, [1], [-1], [-1]]), \
                              SplitOutput([0, [1]], [1, [1]])]]
        axis_reduce_list = [[ReduceInput([0, [0]], [1, [0]]), ReduceOutput([0, 0, True], [1, 0, True])]]
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(grads,
                     x,
                     batch_mean,
                     batch_variance,
                     diff_scale,
                     diff_offset,
                     epsilon,
                     kernel_name="bn_training_update_grad"):
    """
    1. when input(grads)'s ori_shape is [1, ? ,1, ?] and the format is NCHW
    the Op BNTrainingUpdateGrad can support NCHW.
    > for example:
    > grads : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_mean : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_variance : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingUpdateGrad can process with NC1HWC0:
    > grads : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_mean : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_variance : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    format_x = x.get("ori_format").upper()
    origin_shape = x.get("ori_shape")

    if format_x == "NCHW" and len(origin_shape) == 4 and origin_shape[0] == 1 and origin_shape[2] == 1:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="grads",
                                               datatype="float16,float,float16,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float,float16,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="batch_mean",
                                               datatype="float,float,float,float",
                                               format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="batch_variance",
                                               datatype="float,float,float,float",
                                               format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="diff_scale",
                                                datatype="float,float,float,float",
                                                format="NCHW, NCHW,NC1HWC0,NC1HWC0")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="diff_offset",
                                                datatype="float,float,float,float",
                                                format="NCHW, NCHW,NC1HWC0,NC1HWC0")
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="grads",
                                               datatype="float16,float,float16,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float,float16,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="batch_mean",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="batch_variance",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="diff_scale",
                                                datatype="float,float,float,float",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="diff_offset",
                                                datatype="float,float,float,float",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0")

    param_list = [input0, input1, input2, input3, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _check_format(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NDC1HWC0", "NCHW"):
        error_reson = "The data format only supports NC1HWC0 and NCHW and NDC1HWC0."
        error_manager_vector.raise_err_specific_reson("bn_training_update_grad", error_reson)
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_reson = "The origin format only supports NCHW when format is NCHW"
            error_manager_vector.raise_err_specific_reson("bn_training_update_grad", error_reson)


# 'pylint: disable=too-many-statements,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator_compute("BNTrainingUpdateGrad", op_mode="dynamic", support_fusion=True)
def bn_training_update_grad_compute(grads,
                                    x,
                                    batch_mean,
                                    batch_variance,
                                    diff_scale,
                                    diff_offset,
                                    epsilon,
                                    kernel_name="bn_training_update_grad",
                                    reduce_axis=None):
    """
    Compute for bn_training_update_grad_compute
    x_norm:(x-input_reserve_space_1)*
            np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_scale:np.sum(y*(x-input_reserve_space_1)*
                         np.power((reserve_space_2 + epsilon), (-0.5)))
    diff_offset: np.sum(y)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads. Must be one of the following
        type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x. Must be one of the following
        type: `float16`, `float32`.
    batch_mean: TVM tensor 5D
        the placeholder of batch_mean. Must be one of the following
        type: `float32`.
    batch_variance: TVM tensor 5D
        the placeholder of batch_variance. Must be one of the following
        type: `float32`.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_training_update_grad"
    reduce_axis: list
        reduce axis of input shape

    Returns
    -------
    res_list: list
       [diff_scale, diff_offset].
    """
    data_format = diff_scale.get("format").upper()
    if not reduce_axis and data_format in ("NC1HWC0", "NCHW"):
        axis = [0, 2, 3]
    elif not reduce_axis and data_format in ("NDC1HWC0",):
        axis = [0, 1, 3, 4]
    else:
        axis = reduce_axis

    if not isinstance(epsilon, float):
        tbe_context.get_context().add_compile_info("has_epsilon", True)
    epsilon = get_attr_by_cls(epsilon, OpAttr(0, "epsilon", "Float", 0.0000001), "float32")

    if grads.dtype == "float16":
        grads = tbe.cast_to(grads, "float32")

    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")

    shape_x = shape_util.shape_to_list(x.shape)
    batch_mean_inverse = tbe.vmuls(batch_mean, tvm.const(-1, dtype=batch_mean.dtype))
    input_mean = tbe.broadcast(batch_mean_inverse, shape_x)
    x_sub = tbe.vadd(x, input_mean)

    data_adds = tbe.vadds(batch_variance, epsilon)
    data_rsqrt = tbe.vsqrt(data_adds)
    shape_var = shape_util.shape_to_list(batch_variance.shape)
    scalar_one = 1
    data_cast = tbe.broadcast(tvm.const(scalar_one, "float32"), shape_var)
    data_rsqrts = tbe.vdiv(data_cast, data_rsqrt)
    rsqrts_broadcast = tbe.broadcast(data_rsqrts, shape_x)
    x_norm = tbe.vmul(x_sub, rsqrts_broadcast)

    scale_mul = tbe.vmul(grads, x_norm)

    diff_scale, diff_offset = tuple_sum([scale_mul, grads], axis, True)

    res_list = [diff_scale, diff_offset]
    return res_list


# 'pylint: disable=too-many-statements,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator("BNTrainingUpdateGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_training_update_grad(grads,
                            x,
                            batch_mean,
                            batch_variance,
                            diff_scale,
                            diff_offset,
                            epsilon,
                            kernel_name="bn_training_update_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_update_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
    x: dict
        dict of x, A 5D Tensor for input x.
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for output diff_scale.
    diff_offset: dict
        dict of diff_offset, A 5D Tensor for output diff_offset.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.0001`.
    kernel_name: str
        kernel name, default value is "bn_training_update_grad"

    Returns
    -------
    None
    """
    shape_grads = grads.get("shape")
    dtype_grads = grads.get("dtype")
    dtype_x = x.get("dtype")
    dtype_batch_mean = batch_mean.get("dtype")
    dtype_batch_variance = batch_variance.get("dtype")

    input_grads_dtype = dtype_grads.lower()
    input_x_dtype = dtype_x.lower()
    batch_mean_dtype = dtype_batch_mean.lower()
    batch_variance_dtype = dtype_batch_variance.lower()

    para_check.check_dtype(input_grads_dtype, ("float32", "float16"), param_name="grads")
    para_check.check_dtype(input_x_dtype, ("float32", "float16"), param_name="x")
    para_check.check_dtype(batch_mean_dtype, ("float32",), param_name="batch_mean")
    para_check.check_dtype(batch_variance_dtype, ("float32",), param_name="batch_variance")

    shape_util.compare_tensor_dict_key(grads, x, "shape")
    shape_util.compare_tensor_dict_key(batch_mean, batch_variance, "shape")

    data_format = grads.get("format")
    ori_format = grads.get("ori_format")
    _check_format(data_format, ori_format)

    if data_format in ("NCHW",):
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_grads[1]
        range_list = util_common.gen_range(shape_list)
        batch_mean["shape"] = shape_list
        batch_mean["range"] = range_list
        batch_variance["shape"] = shape_list
        batch_variance["range"] = range_list

    if data_format in ("NC1HWC0", "NCHW"):
        list_axis = [0, 2, 3]
    else:
        list_axis = [0, 1, 3, 4]

    extra_params = {"compile_broadcast_axis": {2: list_axis, 3: list_axis}}
    ins = classify([grads, x, batch_mean, batch_variance, list_axis],
                   OpPatternMode.TUPLE_REDUCE,
                   extra_params=extra_params)
    schedules, tensors = [], []
    for (_grads, _x, _batch_mean, _batch_variance, _reduce_axis) in ins:
        with tbe.compute():
            _shape_grads, _shape_x, _shape_mean, _shape_variance = shape_util.variable_shape(
                [_grads, _x, _batch_mean, _batch_variance], op_mode=OpPatternMode.TUPLE_REDUCE)
            input_grads = tvm.placeholder(_shape_grads, name="input_grads", dtype=input_grads_dtype)
            input_x = tvm.placeholder(_shape_x, name="input_x", dtype=input_x_dtype)
            input_mean = tvm.placeholder(_shape_mean, name="input_mean", dtype=batch_mean_dtype)
            input_variance = tvm.placeholder(_shape_variance, name="input_variance", dtype=batch_variance_dtype)
            res = bn_training_update_grad_compute(input_grads, input_x, input_mean, input_variance, diff_scale,
                                                  diff_offset, epsilon, kernel_name, _reduce_axis)

            tensor_list = [input_grads, input_x, input_mean, input_variance] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
