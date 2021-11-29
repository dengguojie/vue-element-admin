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
bn_training_update_grad
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tuple_sum
from impl.util.platform_adapter import para_check
from tbe.common.utils import shape_util
from tbe.common.utils.errormgr import error_manager_vector
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import add_compile_info


def _check_format_nd(data_format, origin_foramt):
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
@register_operator_compute("BNTrainingUpdateGrad", op_mode="dynamic", support_fusion=False)
def bn_training_update_grad_compute(grads, x, batch_mean, batch_variance,
                                    diff_scale, diff_offset, epsilon,
                                    kernel_name="bn_training_update_grad"):
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

    Returns
    -------
    res_list: list
       [diff_scale, diff_offset].
   """
    scalar_one = 1
    shape_x = shape_util.shape_to_list(x.shape)
    axis = [0, 2, 3]

    if grads.dtype == "float16":
        grads = tbe.cast_to(grads, "float32")

    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")

    batch_mean_inverse = tbe.vmuls(batch_mean, tvm.const(-1, dtype=batch_mean.dtype))
    input_mean = tbe.broadcast(batch_mean_inverse, shape_x)
    x_sub = tbe.vadd(x, input_mean)

    data_adds = tbe.vadds(batch_variance, epsilon)
    data_rsqrt = tbe.vsqrt(data_adds)
    shape_var = shape_util.shape_to_list(batch_variance.shape)
    data_cast = tbe.broadcast(tvm.const(scalar_one, "float32"), shape_var)
    data_rsqrts = tbe.vdiv(data_cast, data_rsqrt)
    rsqrts_broadcast = tbe.broadcast(data_rsqrts, shape_x)
    x_norm = tbe.vmul(x_sub, rsqrts_broadcast)

    scale_mul = tbe.vmul(grads, x_norm)

    [diff_scale, diff_offset] = tuple_sum([scale_mul, grads], axis, True)

    res_list = [diff_scale, diff_offset]
    return res_list


# 'pylint: disable=too-many-statements,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator("BNTrainingUpdateGrad", "BNTrainingUpdateGrad")
def bn_training_update_grad(grads, x, batch_mean, batch_variance,
                            diff_scale, diff_offset, epsilon=0.0001,
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

    dynamic_dim = -1
    shape_grads = grads.get("shape")
    shape_x = x.get("shape")
    shape_batch_mean = batch_mean.get("shape")
    shape_batch_variance = batch_variance.get("shape")

    dtype_grads = grads.get("dtype")
    dtype_x = x.get("dtype")
    dtype_batch_mean = batch_mean.get("dtype")
    dtype_batch_variance = batch_variance.get("dtype")

    range_grads = grads.get("range")
    range_x = x.get("range")
    range_batch_mean = batch_mean.get("range")
    range_batch_variance = batch_variance.get("range")

    input_grads_dtype = dtype_grads.lower()
    input_x_dtype = dtype_x.lower()
    batch_mean_dtype = dtype_batch_mean.lower()
    batch_variance_dtype = dtype_batch_variance.lower()

    data_format = grads.get("format")
    ori_format = grads.get("ori_format")
    _check_format_nd(data_format, ori_format)

    if data_format == "NDC1HWC0":
        shape_grads = [shape_grads[0] * shape_grads[1], shape_grads[2], shape_grads[3], shape_grads[4], shape_grads[5]]
        shape_x = [shape_x[0] * shape_x[1], shape_x[2], shape_x[3], shape_x[4], shape_x[5]]
        shape_batch_mean = [shape_batch_mean[0] * shape_batch_mean[1], shape_batch_mean[2], shape_batch_mean[3], \
        shape_batch_mean[4], shape_batch_mean[5]]
        shape_batch_variance = [shape_batch_variance[0] * shape_batch_variance[1], shape_batch_variance[2], \
        shape_batch_variance[3], shape_batch_variance[4], shape_batch_variance[5]]

    schedules = []
    tensors = []

    ins = [[{'shape': shape_grads, 'range': range_grads, 'const_shape': shape_grads},
            {'shape': shape_x, 'range': range_x, 'const_shape': shape_x},
            {'shape': shape_batch_mean, 'range': range_batch_mean, 'const_shape': shape_batch_mean},
            {'shape': shape_batch_variance, 'range': range_batch_variance, 'const_shape': shape_batch_variance}
            ]]

    for (_, _, _, _) in ins:
        with tbe.compute():

            if dynamic_dim in shape_x:
                mode = para_check.ORIGINAL
                dim_0_0 = operation.var("dim_0_0", range_grads[0])
                dim_0_1 = operation.var("dim_0_1", range_grads[1])
                dim_0_2 = operation.var("dim_0_2", range_grads[2])
                dim_0_3 = operation.var("dim_0_3", range_grads[3])

                shape1 = [dim_0_0, dim_0_1, dim_0_2, dim_0_3, 16]
                shape2 = [1, dim_0_1, 1, 1, 16]

                grads_input = tvm.placeholder(shape1, name="grads_input",
                                            dtype=input_grads_dtype)
                x_input = tvm.placeholder(shape1, name="x_input", dtype=input_x_dtype)
                batch_mean_input = tvm.placeholder(shape2,
                                                name="batch_mean_input",
                                                dtype=batch_mean_dtype)
                batch_variance_input = tvm.placeholder(shape2,
                                                    name="batch_variance_input",
                                                    dtype=batch_variance_dtype)
            else:
                mode = para_check.CONST
                grads_input = tvm.placeholder(shape_grads, name="grads_input",
                                            dtype=input_grads_dtype)
                x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_x_dtype)
                batch_mean_input = tvm.placeholder(shape_batch_mean,
                                                name="batch_mean_input",
                                                dtype=batch_mean_dtype)
                batch_variance_input = tvm.placeholder(shape_batch_variance,
                                                    name="batch_variance_input",
                                                    dtype=batch_variance_dtype)

            get_context().get_current_compute().add("mode", mode)
            add_compile_info("mode", mode)

            res_list = bn_training_update_grad_compute(grads_input, x_input,
                                                    batch_mean_input,
                                                    batch_variance_input, diff_scale,
                                                    diff_offset,
                                                    epsilon, kernel_name=kernel_name)
            tensors.append([grads_input, x_input, batch_mean_input, batch_variance_input] + res_list)
            with tvm.target.cce():
                sch = tbe.auto_schedule(res_list)
            schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
