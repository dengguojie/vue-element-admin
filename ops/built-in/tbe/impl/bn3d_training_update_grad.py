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
bn_3d_training_update_grad
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.bn_training_update_grad import bn_training_update_grad

SCALAR_ONE = 1


# pylint: disable = unused-argument
# pylint: disable=invalid-name,redefined-builtin,too-many-statements
@tbe_platform.fusion_manager.fusion_manager.register("bn3d_training_update_grad")
def bn3d_training_update_grad_compute(grads, x, batch_mean, batch_variance,
                                      diff_scale, diff_offset, epsilon,
                                      kernel_name="bn3d_training_update_grad"):
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
        kernel name, default value is "bn_3d_training_update_grad"

    Returns
    -------
    res_list: list
       [diff_scale, diff_offset].
   """
    shape_x = shape_util.shape_to_list(x.shape)
    axis = [0, 2, 3]

    if grads.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        grads = tbe.cast_to(grads, "float32")
    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        x = tbe.cast_to(x, "float32")
    batch_mean_inverse = tbe.vmuls(batch_mean, tvm.const(-1, dtype=batch_mean.dtype))
    input_mean = tbe.broadcast(batch_mean_inverse, shape_x)
    x_sub = tbe.vadd(x, input_mean)

    data_adds = tbe.vadds(batch_variance, epsilon)
    data_rsqrt = tbe.vsqrt(data_adds)
    shape_var = shape_util.shape_to_list(batch_variance.shape)
    data_cast = tbe.broadcast(tvm.const(SCALAR_ONE, "float32"), shape_var)
    data_rsqrts = tbe.vdiv(data_cast, data_rsqrt)
    rsqrts_broadcast = tbe.broadcast(data_rsqrts, shape_x)
    x_norm = tbe.vmul(x_sub, rsqrts_broadcast)

    scale_mul = tbe.vmul(grads, x_norm)

    diff_scale, diff_offset = tbe.tuple_sum([scale_mul, grads], axis, True)
    res_list = [diff_scale, diff_offset]
    return res_list


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn3d_training_update_grad(grads, x, batch_mean, batch_variance,
                              diff_scale, diff_offset, epsilon=0.0001,
                              kernel_name="bn3d_training_update_grad"):
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
        kernel name, default value is "bn_3d_training_update_grad"

    Returns
    -------
    None
    """

    bn_training_update_grad(grads, x, batch_mean, batch_variance, diff_scale,
                            diff_offset, epsilon, kernel_name=kernel_name)


