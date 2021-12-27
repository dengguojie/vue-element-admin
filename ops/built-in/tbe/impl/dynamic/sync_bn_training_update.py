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
sync_bn_training_update
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector

@register_operator_compute("SyncBNTrainingUpdate", op_mode="dynamic", support_fusion=True)
# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,too-many-statements
# 'pylint: disable=unused-argument,invalid-name
def sync_bn_training_update_compute(mean, running_mean, momentum, kernel_name="sync_bn_training_update"):
    """
    calculating data's sync_bn_training_update, running_mean_update = mean * momentum + running_mean * (1 - momentum)
    :param mean: TVM tensor
    :param running_mean: TVM tensor
    :param momentum: float
    :param kernel_name: str
    :return: TVM tensor
    """

    input_dtype = mean.dtype

    momentum_value = tvm.const(momentum, dtype=input_dtype)
    one_momentum_value = tvm.const(1 - momentum, dtype=input_dtype)

    mean_temp = tbe.vmuls(mean, momentum_value)
    running_mean_temp = tbe.vmuls(running_mean, one_momentum_value)
    res = tbe.vadd(mean_temp, running_mean_temp)

    return res


#register op

@register_operator("SyncBNTrainingUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,too-many-statements
# 'pylint: disable=unused-argument,invalid-name
def sync_bn_training_update(mean,
                            running_mean,
                            running_mean_update,
                            momentum=0.01,
                            kernel_name="sync_bn_training_update"):
    """
    algorithm: sync_bn_training_update
    calculating data's sync_bn_training_update, running_mean_update = mean * momentum + running_mean * (1 - momentum)

    Parameters
    ----------
    mean : dict
        shape and dtype of first input, only support float16, float32
    running_mean : dict
        shape and dtype of second input, only support float16, float32
    running_mean_update : dict
        shape and dtype of output, should be broadcast shape and type as input
    momentum : float
        the update step length
    kernel_name : str
        cce kernel name, default value is sync_bn_training_update

    Returns
    -------
    None
    """
    dtype_mean = mean.get("dtype").lower()
    dtype_running_mean = running_mean.get("dtype").lower()

    check_list = ["float16", "float32"]

    para_check.check_dtype(dtype_mean, check_list)
    para_check.check_dtype(dtype_running_mean, check_list)

    if dtype_mean != dtype_running_mean:
        error_manager_vector.raise_err_two_input_dtype_invalid('sync_bn_training_update', "mean", "running_mean", \
                        "the dtype of mean, running_mean, must be the same")

    ins = classify([mean, running_mean], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_mean, _running_mean) in ins:
        with tbe.compute():
            shape_mean, shape_running_mean = shape_util.variable_shape([_mean, _running_mean])

            data_mean = tvm.placeholder(shape_mean, dtype=dtype_mean, name="data_mean")
            data_running_mean = tvm.placeholder(shape_running_mean, dtype=dtype_running_mean, name="data_running_mean")

            res = sync_bn_training_update_compute(data_mean, data_running_mean, momentum, kernel_name)
            input_list = [data_mean, data_running_mean, res]
            tensors.append(input_list)

        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
