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
batch_norm_3d
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.batch_norm import batch_norm


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin, too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_norm3d(x, scale, offset, mean, variance, y, batch_mean,
                 batch_variance, reserve_space_1, reserve_space_2,
                 epsilon=0.0001, data_format="NCDHW",
                 is_training=True, kernel_name="batch_norm3d"):
    """
    algorithm: fused_batch_norm_3d
    Batch normalization.
    Note that the size of 5D Tensors are defined by "NDC1HWC0".
    The input tensor's dimension C should be equal.

    Parameters
    ----------
    x: dict
        dict of input, A Tensor for input data.
    scale: dict
        dict of scale,
        A Tensor for scaling factor, to scale the normalized x.
    offset: dict
        dict of offset, A Tensor for offset, to shift to the normalized x.
    mean: dict
        dict of mean, A Tensor for population mean.
        Used for inference only, must be empty for training.
    variance: dict
        dict of variance, A Tensor for population variance.
        Used for inference only, must be empty for training.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`. Has the same type as `mean`.
    batch_variance: dict
        dict of batch_var, A `Tensor`. Has the same type as `variance`.
    reserve_space_1: dict
        dict of reserve_space_1, A `Tensor`.
    reserve_space_2: dict
        dict of reserve_space_2, A `Tensor`.
    epsilon: float
        A small float number added to the variance of x. Defaults to `0.001`.
    data_format: str
        The data format for x and y. Support "NC1HWC0" only.
    is_training: bool
        A bool value indicates the operation for train (default) or inference.
    kernel_name: str
        kernel name, default value is "batch_norm"

    Returns
    -------
    None
    """

    batch_norm(x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1,
               reserve_space_2, epsilon, data_format, is_training, kernel_name=kernel_name)
