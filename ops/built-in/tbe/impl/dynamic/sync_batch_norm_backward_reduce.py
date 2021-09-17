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

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpImplMode


@register_operator_compute("SyncBatchNormBackwardReduce", op_mode="dynamic", support_fusion=True)
def sync_batch_norm_backward_reduce_compute(sum_dy,
                                            sum_dy_dx_pad,
                                            mean,
                                            invert_std,
                                            sum_dy_xmu,
                                            y,
                                            kernel_name="sync_batch_norm_backward_reduce"):
    """
    sync_batch_norm_backward_reduce compute

    Parameters:
    ----------
    sum_dy: TVM tensor
        The result of pad grad_bias
    sum_dy_dx_pad: TVM tensor
        The result of pad sum_dy_dx
    mean: TVM tensor
        Mean of saved forward input
    invert_std: TVM tensor
        Reciprocal of the variance of the saved forward input
    kernel_name: str
        cce kernel nema, default value is sync_batch_norm_backward_reduce
    
    Returns
    -------
    res: TVM tensor

    """

    dy_mean = tbe.vmul(mean, sum_dy)
    sum_dy_xmu = tbe.vsub(sum_dy_dx_pad, dy_mean)
    grad_weight_res = tbe.vmul(sum_dy_xmu, invert_std)
    return [sum_dy_xmu, grad_weight_res]


def sync_batch_norm_backward_reduce(sum_dy,
                                    sum_dy_dx_pad,
                                    mean,
                                    invert_std,
                                    sum_dy_xmu,
                                    y,
                                    kernel_name="sync_batch_norm_backward_reduce"):
    """
    algorithm: batch_norm_grad_reduce

    Parameters:
    ---------
    sum_dy: dict
        The result of pad grad_bias
    sum_dy_dx_pad: dict
        The result of pad sum_dy_dx
    mean: dict
        Mean of saved forward input
    invert_std: dict
        Reciprocal of the variance of the saved forward input
    kernel_name: str
        cce kernel nema, default value is sync_batch_norm_backward_reduce
    
    Returns
    -------
    None
    """
    dtype_lower_sum_dy = sum_dy["dtype"].lower()
    dtype_lower_sum_dy_dx_pad = sum_dy_dx_pad["dtype"].lower()
    dtype_lower_mean = mean["dtype"].lower()
    dtype_lower_invert_std = invert_std["dtype"].lower()

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_lower_sum_dy, check_list, param_name="sum_dy")
    sum_dy["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_sum_dy_dx_pad, check_list, param_name="sum_dy_dx_pad")
    sum_dy_dx_pad["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_mean, check_list, param_name="mean")
    mean["rel_pos_to_reduce"] = "before"
    para_check.check_dtype(dtype_lower_invert_std, check_list, param_name="invert_std")
    invert_std["rel_pos_to_reduce"] = "before"

    tensors = []
    schedules = []
    ins = classify([sum_dy, sum_dy_dx_pad, mean, invert_std], OpPatternMode.ELEWISE)

    for (_sum_dy, _sum_dy_dx_pad, _mean, _invert_std) in ins:
        with tbe.compute():
            shape_sum_dy, shape_sum_dy_dx_pad, shape_mean, shape_invert_std = \
                shape_util.variable_shape([_sum_dy, _sum_dy_dx_pad, _mean, _invert_std])
            data_input_sum_dy = tvm.placeholder(shape_sum_dy, name="sum_dy", dtype=dtype_lower_sum_dy)
            data_input_sum_dy_dx_pad = tvm.placeholder(shape_sum_dy_dx_pad, name="data_sum_dy_dx_pad", dtype=dtype_lower_sum_dy_dx_pad)
            data_input_mean = tvm.placeholder(shape_mean, name="data_mean", dtype=dtype_lower_mean)
            data_input_invert_std = tvm.placeholder(shape_invert_std, name="data_invert_std", dtype=dtype_lower_invert_std)

            res = sync_batch_norm_backward_reduce_compute(data_input_sum_dy, data_input_sum_dy_dx_pad, data_input_mean,
                                                                 data_input_invert_std, sum_dy_xmu, y)
            tensors.append([data_input_sum_dy, data_input_sum_dy_dx_pad, data_input_mean, data_input_invert_std] + res)                                              
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)