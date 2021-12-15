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
soft_margin_loss_gard
"""
import te.lang.cce as tbe
from te import tvm
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_conf import api_check_support
from te.utils import shape_util
from te.utils import para_check
from te.utils.shape_util import broadcast_shapes
from te.utils.shape_util import shape_to_list


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=unused-argument,too-many-locals
@fusion_manager.register("soft_margin_loss_grad")
def soft_margin_loss_gard_compute(input_predict, input_label, input_dout,
                                  reduction, kernel_name="soft_margin_loss_grad"):
    """calculating data

    Parameters
    ----------
    :param input_predict: TVM tensor
        the placeholder of input_predict
    :param input_label: TVM tensor
        the placeholder of input_label
    :param input_dout: TVM tensor
        the placeholder of input_dout
    :param reduction: str
        the method of reduction
    :param kernel_name: str
        kernel name, default value is "soft_margin_loss_gard"

    Returns
    -------
    output tensor
    """
    predict_shape = shape_to_list(input_predict.shape)
    label_shape = shape_to_list(input_label.shape)
    _, _, shape_max = broadcast_shapes(predict_shape, label_shape)
    para_check.check_shape_size(shape_max, Constant.SHAPE_SIZE_LIMIT)

    input_predict = tbe.broadcast(input_predict, shape_max)
    input_label = tbe.broadcast(input_label, shape_max)
    input_dout = tbe.broadcast(input_dout, shape_max)

    dtype = input_predict.dtype

    predict_data = input_predict
    label_data = input_label
    dout_data = input_dout
    predict_neg = tbe.vmuls(predict_data, tvm.const(-1, dtype))

    predict_mul_label = tbe.vmul(predict_neg, label_data)
    cloud_flag = api_check_support("te.lang.cce.vexp", "float32")

    if predict_mul_label.dtype == "float16":
        if cloud_flag:
            predict_mul_label = tbe.cast_to(predict_mul_label, "float32")
            z = tbe.vexp(predict_mul_label)
        else:
            z = tbe.vexp(predict_mul_label)
    else:
        if cloud_flag:
            z = tbe.vexp(predict_mul_label)
        else:
            predict_mul_label = tbe.cast_to(predict_mul_label, "float16")
            z = tbe.vexp(predict_mul_label)

    num = 1
    if reduction == 'mean':
        for shape in predict_shape:
            num *= shape

    res1 = tbe.vmuls(tbe.vmuls(label_data, tvm.const(-1, dtype)),
                     tvm.const(1 / num, dtype))
    res2 = tbe.vdiv(z, tbe.vadds(z, tvm.const(1, dtype)))
    res3 = tbe.vmul(res1, res2)

    res = tbe.vmul(res3, dout_data)

    if dtype == "float16" and cloud_flag:
        res = tbe.cast_to(res, "float16")
    if dtype == "float32" and not cloud_flag:
        res = tbe.cast_to(res, "float32")
    return res


# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def soft_margin_loss_grad(input_predict, input_label, input_dout, output_gdient,
                          reduction="mean",
                          kernel_name="soft_margin_loss_grad"):
    """calculating data

    Parameters
    ----------
    :param input_predict: dict
        shape and dtype of predpict
    :param input_label: dict
        shape and dtype of label
    :param input_dout: dict
        shape and dtype of dout
    :param output_gdient: dict
        shape and dtype of output, should be same shape and type as predpict
    :param reduction: str
        the method of reduction
    :param kernel_name: str
        kernel name, default value is "soft_margin_loss_gard"

    Returns
    -------
    None
    """

    predict_shape = shape_util.scalar2tensor_one(input_predict.get("shape"))
    predict_dtype = input_predict.get("dtype")
    label_shape = shape_util.scalar2tensor_one(input_label.get("shape"))
    label_dtype = input_label.get("dtype")
    dout_shape = shape_util.scalar2tensor_one(input_dout.get("shape"))
    dout_dtype = input_dout.get("dtype")

    # reshape
    predict_shape = list(predict_shape)
    label_shape = list(label_shape)
    dout_shape = list(dout_shape)
    if len(predict_shape) > len(label_shape):
        times = len(predict_shape) - len(label_shape)
        cnt = 0
        while cnt < times:
            label_shape.insert(0, 1)
            cnt += 1

    if len(predict_shape) > len(dout_shape):
        times = len(predict_shape) - len(dout_shape)
        cnt = 0
        while cnt < times:
            dout_shape.insert(0, 1)
            cnt += 1

    # initialize data
    predict_data = tvm.placeholder(predict_shape, name="predict_data",
                                   dtype=predict_dtype)
    label_data = tvm.placeholder(label_shape, name="label_data",
                                 dtype=label_dtype)
    dout_data = tvm.placeholder(dout_shape, name="dout_data",
                                dtype=dout_dtype)
    res = soft_margin_loss_gard_compute(predict_data, label_data, dout_data,
                                        reduction)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [predict_data, label_data, dout_data, res]}
    tbe.build(schedule, config)
