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
apply_proximal_adagrad
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from impl.util import util_apply_op_schedule
from impl.util import util_compute
from te.utils.error_manager import error_manager_vector

CONST_ZERO = 0
CONST_ONE = 1


def _check_shape_is_same(var, accum, grad):
    """
    Check whether var.shape accum.shape and grad.shape is same or not.

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.

    Returns:
    None
    """
    shape_var = var.get("shape")
    shape_accum = accum.get("shape")
    shape_grad = grad.get("shape")
    if shape_var != shape_accum or shape_var != shape_grad:
        error_detail = "shape of var and accum and grad should be same"
        error_manager_vector.raise_err_input_shape_invalid("apply_proximal_adagrad_d", "var or accum or grad", error_detail)

# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-locals,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("apply_proximal_adagrad_d")
def apply_proximal_adagrad_d_compute(var, accum, lr, l1, l2, grad, var_out,
                                     accum_out, use_locking=False,
                                     kernel_name="apply_proximal_adagrad"):
    """
    the operator's compute
    accum += grad * grad
    learning_rate = lr_broad * rsqrt(accum)
    prox_v = var - grad * learning_rate
    if l1 > 0 :
        var = sign(prox_v)/(1+learning_rate*l2)*max{|prox_v|-learning_rate*l1,0}
    else:
        var = prox_v / (1+l2*learning_rate)

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
        the value of out_var, accum_out, out_data
    """
    dtype = var.dtype
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vsqrt", "float32"):
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        l1 = tbe.cast_to(l1, "float32")
        l2 = tbe.cast_to(l2, "float32")
        grad = tbe.cast_to(grad, "float32")
        has_improve_precision = True

    lr_broad = tbe.broadcast(lr, var.shape)
    l1_broad = tbe.broadcast(l1, var.shape)
    l2_broad = tbe.broadcast(l2, var.shape)

    grad_2 = tbe.vmul(grad, grad)
    accum_out = tbe.vadd(accum, grad_2)
    accum_sqrt = tbe.vsqrt(accum_out)
    learning_rate = tbe.vdiv(lr_broad, accum_sqrt)
    learning_rate_grad = tbe.vmul(grad, learning_rate)
    prox_v = tbe.vsub(var, learning_rate_grad)
    l2_lr = tbe.vmul(l2_broad, learning_rate)
    l2_lr_1 = tbe.vadds(l2_lr, tvm.const(CONST_ONE, "float32"))
    prox_v_abs = tbe.vabs(prox_v)
    prox_v_sign = util_compute.sign(prox_v)
    learning_rate_l1 = tbe.vmul(learning_rate, l1_broad)
    prox_v_l1 = tbe.vsub(prox_v_abs, learning_rate_l1)
    max_value = tbe.vmax(prox_v_l1, tbe.broadcast(
        tvm.const(CONST_ZERO, "float32"), prox_v.shape))
    var_res = tbe.vmul(prox_v_sign, max_value)
    var_new = tbe.vdiv(var_res, l2_lr_1)
    output_data = tbe.vadds(var_new, tvm.const(CONST_ZERO, "float32"))
    output_accum_data = tbe.vadds(accum_out, tvm.const(CONST_ZERO, "float32"))

    if has_improve_precision:
        var_new = tbe.cast_to(var_new, "float16")
        accum_out = tbe.cast_to(accum_out, "float16")
        output_data = tbe.cast_to(output_data, "float16")
        output_accum_data = tbe.cast_to(output_accum_data, "float16")

    # this compute is for muti output
    def _compute(*index):
        return var_new(*index), accum_out(*index), output_data(*index), output_accum_data(*index)

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def apply_proximal_adagrad_d(var, accum, lr, l1, l2, grad, var_out,
                             accum_out, use_locking=False,
                             kernel_name="apply_proximal_adagrad_d"):
    """
    Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
    None
    """
    _check_shape_is_same(var, accum, grad)

    input_dict = (var, accum, lr, l1, l2, grad)
    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_proximal_adagrad_d_compute,
                                                           [var_out, accum_out], 15)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'accum', 'lr', 'l1', 'l2', 'grad'),
                                                           scalar=('lr', 'l1', 'l2'),
                                                           reuse=('var', 'accum'))
    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name), kernel_name)
