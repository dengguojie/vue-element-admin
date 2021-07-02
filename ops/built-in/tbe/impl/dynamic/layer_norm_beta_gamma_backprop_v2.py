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
layer_norm_beta_gamma_backprop_v2
"""
import operator
import tbe.common.register as tbe_register
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


@tbe_register.register_param_generalization("LayerNormBetaGammaBackpropV2")
def layer_norm_beta_gamma_backprop_v2_generalization(input_dy, res_for_gamma, output_pd_gamma, output_pd_beta,
                                                     shape_gamma, impl_mode, generalize_config=None):
    """
    for now only support dy (-1, -1, N)  and shape_gamma is (N,)
    """
    result = []
    last_dim = input_dy["shape"][-1]
    shape_in = (-1, -1, last_dim)
    range_in = [(1, -1), (1, -1), (last_dim, last_dim)]
    shape_out = (last_dim, )
    range_out = [(last_dim, last_dim)]
    input_dy["shape"], input_dy["ori_shape"] = shape_in, shape_in
    input_dy["range"], input_dy["ori_range"] = range_in, range_in
    res_for_gamma["shape"], res_for_gamma["ori_shape"] = shape_in, shape_in
    res_for_gamma["range"], res_for_gamma["ori_range"] = range_in, range_in
    output_pd_gamma["shape"], output_pd_gamma["ori_shape"] = shape_out, shape_out
    output_pd_gamma["range"], output_pd_gamma["ori_range"] = range_out, range_out
    output_pd_beta["shape"], output_pd_beta["ori_shape"] = shape_out, shape_out
    output_pd_beta["range"], output_pd_beta["ori_range"] = range_out, range_out
    result.append([input_dy, res_for_gamma, output_pd_gamma, output_pd_beta, shape_gamma])
    return result


def _update_gamma_shape(shape_x, shape_gamma):
    """
    update shape_gamma for subsequent calculation
    """
    params_axis_tmp = []
    if len(shape_x) != len(shape_gamma):
        sub = len(shape_x) - len(shape_gamma)
        shape_gamma = list(shape_gamma)
        for i in range(sub):
            shape_gamma.insert(0, 1)
            params_axis_tmp.append(i)

    shape_gamma_new = tuple(shape_gamma)
    params_axis = tuple(params_axis_tmp)

    return shape_gamma_new, params_axis


def layer_norm_beta_gamma_backprop_v2_compute(data_dy, res_for_gamma, output_pd_gamma,
                                              output_pd_beta, shape_gamma,
                                              kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    DSL description of the layernorm_grad operator's
    mathematical calculation process

    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of dy input data
    res_for_gamma: TVM tensor
        the placeholder of x input data
    input_variance: TVM tensor
        the placeholder of variance input data
    input_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma: list or tuple
        original shape of gamma

    Returns
    -------
    res_tuple: tuple
        (pd_gamma, pd_beta)
    """
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(res_for_gamma.shape)
    param_axis = _update_gamma_shape(shape_x, shape_gamma)[1]

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        dtype = "float32"

    data_dy_2 = data_dy
    if has_improve_precision:
        data_dy_2 = tbe.cast_to(data_dy, "float32")
        data_dy = tbe.cast_to(data_dy, "float32")
        res_for_gamma = tbe.cast_to(res_for_gamma, "float32")

    data_x = tbe.vmul(res_for_gamma, data_dy)
    if param_axis:
        pd_gamma = tbe.reduce_sum(data_x, param_axis, keepdims=True)
        pd_beta = tbe.reduce_sum(data_dy_2, param_axis, keepdims=True)
    else:
        pd_beta = tbe.vadds(data_dy_2, tvm.const(0, dtype=dtype))
        pd_gamma = tbe.vadds(data_x, tvm.const(0, dtype=dtype))

    if dtype == "float16" and not has_improve_precision:
        pd_gamma = tbe.cast_to(pd_gamma, "float32")
        pd_beta = tbe.cast_to(pd_beta, "float32")

    res_list = [pd_gamma, pd_beta]

    return res_list


@register_operator("LayerNormBetaGammaBackpropV2", pattern="Layer_norm_beta_gamma_backprop")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def layer_norm_beta_gamma_backprop_v2(input_dy, res_for_gamma, output_pd_gamma,
                                      output_pd_beta, shape_gamma,
                                      kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
      algorithm: layernorm_grad
      calculating: gradient of layernorm
                   compute partial derivation of x, gamma and beta
          pd_gamma = np.sum(data_dy*res_for_gamma, param_axis, keepdims=True)
          pd_beta  = np.sum(data_dy, param_axis, keepdims=True)

      Parameters
      ----------
      input_dy : dict
          shape and dtype of input dy, only support float16, float32
      res_for_gamma: dict
          shape and dtype of input res_for_gamma, only support float16, float32
      output_pd_gamma: dict
          shape and dtype of output, only support float16, float32
      output_pd_beta: dict
          shape and dtype of output, only support float16, float32
      shape_gamma: list
          shape of gamma
      kernel_name: str
          cce kernel name, default value is "layer_norm_beta_gamma_backprop_v2"

      Returns
      -------
      None
      """
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    dtype_x = res_for_gamma.get("dtype").lower()
    format_dy = input_dy.get("format")

    dim_0 = tbe.var("dim_0")
    dim_1 = tbe.var("dim_1")
    shape_data = (dim_0, dim_1, shape_dy[2])

    data_dy = tvm.placeholder(shape_data, name="data_dy_layernormgrad_beta_gamma", dtype=dtype)
    data_x = tvm.placeholder(shape_data, name="data_x", dtype=dtype_x)

    with tbe.compute():
        res_list = layer_norm_beta_gamma_backprop_v2_compute(data_dy,
                                                             data_x,
                                                             output_pd_gamma,
                                                             output_pd_beta,
                                                             shape_gamma)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res_list)

    tensor_list = [data_dy, data_x] + list(res_list)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.build(sch, config)

