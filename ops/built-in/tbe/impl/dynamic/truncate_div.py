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
truncate_div
"""
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm
import te.lang.cce as tbe
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
import te.lang.base as tbe_base


# pylint: disable=locally-disabled,too-many-locals,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("truncate_div")
def truncate_div_compute(input_x, input_y, output_x,
                         kernel_name="truncate_div"):
    """
    compute truncate_div
    calculating data's truncate_div, res = floor(x / y) if x/y>0 else ceil(x/y)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data x
    input_y: TVM tensor
        the placeholder of input data y
    output_x: dict
        not used yet
    kernel_name: str
        kernel name

    Returns
    -------
    res: TVM tensor
        the result of truncate_div_compute
    """
    shape_list = shape_util.broadcast_shapes(
        shape_util.shape_to_list(input_x.shape),
        shape_util.shape_to_list(input_y.shape),
        param_name_input1="input_x", param_name_input2="input_y")
    int_list = ("int32", "int8", "uint8")
    input_dtype = input_x.dtype

    if input_dtype in int_list:
        data_x_broad = tbe.cast_to(input_x, 'float32')
        data_y_broad = tbe.cast_to(input_y, 'float32')
        data_x_broad = tbe.broadcast(data_x_broad, shape_list[2])
        data_y_broad = tbe.broadcast(data_y_broad, shape_list[2])
        res_div = tbe.vdiv(data_x_broad, data_y_broad)
        if tbe_platform.api_check_support("tbe.vmins", "float32"):
            res_min_int = tbe.ceil(tbe.vmins(res_div, 0))
            res_max_int = tbe.floor(tbe.vmaxs(res_div, 0))
        else:
            data_zero = tbe.broadcast(tvm.const(0, 'float32'), shape_list[2], 'float32')
            res_min_int = tbe.ceil(tbe.vmin(res_div, data_zero))
            res_max_int = tbe.floor(tbe.vmax(res_div, data_zero))
        res_trunc = tbe.vadd(res_min_int, res_max_int)
    else:
        if tbe_platform.api_check_support("tbe.vlog", "float32"):
            input_x = tbe.cast_to(input_x, 'float32')
            input_y = tbe.cast_to(input_y, 'float32')
        data_x_broad = tbe.broadcast(input_x, shape_list[2])
        data_y_broad = tbe.broadcast(input_y, shape_list[2])
        res_trunc = tbe.vdiv(data_x_broad, data_y_broad)

    res = tbe.cast_to(res_trunc, input_dtype)

    return res


@tbe_base.register_operator("TruncateDiv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def truncate_div(input_x, input_y, output_x, kernel_name="truncate_div"):
    """
    algorithm: truncate_div
    calculating data's truncate_div, res = floor(x / y) if x/y>0 else ceil(x/y)

    Parameters
    ----------
    input_x: dict with keys(shape and dtype)
        only support {float16, float32, int8, uint8(on mini)},
        {float16, float32(on cloud)}
    input_y: dict with keys(shape and dtype)
        dict info of input_y
    output_x: dict with keys(shape and dtype)
        dict info of output_x
    kernel_name: str
        kernel name, default value is "truncate_div"

    Returns
    -------
    None
    """

    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_x")

    if x_dtype != y_dtype:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_018
        error_info['op_name'] = 'truncate_div'
        error_info['param_name1'] = 'x_dtype'
        error_info['param_name2'] = 'y_dtype'
        error_info['param1_dtype'] = str(x_dtype)
        error_info['param2_dtype'] = str(y_dtype)
        raise RuntimeError(error_info,
                           "In op[%s], the parameter[%s][%s] are not equal in "
                           "dtype with dtype[%s][%s]." % (
                               error_info['op_name'],
                               error_info['param_name1'],
                               error_info['param_name2'],
                               error_info['param1_dtype'],
                               error_info['param2_dtype']))

    ins = classify([input_x, input_y], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x, _input_y) in ins:
        with tbe_base.compute():
            x_shape, y_shape = \
                shape_util.variable_shape([_input_x, _input_y], support_broadcast=True)
            reshape_x, reshape_y = shape_util.refine_shapes_for_broadcast(x_shape,
                                                                          y_shape)
            data1 = tvm.placeholder(reshape_x, dtype=x_dtype, name="data1")
            data2 = tvm.placeholder(reshape_y, dtype=y_dtype, name="data2")
            res = truncate_div_compute(data1, data2, output_x, kernel_name)
            tensors.append([data1, data2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
