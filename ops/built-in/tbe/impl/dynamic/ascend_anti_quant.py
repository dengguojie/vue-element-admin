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
ascend_anti_quant
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl import ascend_quant_util as util


# 'pylint: disable=too-many-arguments,invalid-name,unused-argument
# 'pylint: disable=unnecessary-lambda
# 'pylint: disable=too-many-locals
def _check_params(x, y, scale, offset, sqrt_mode, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr.
    """
    shape = x.get("shape")
    x_format = x.get("format")
    x_dtype = x.get("dtype").lower()

    para_check.check_format(x_format, ("NC1HWC0",), param_name="x")
    para_check.check_shape(shape, min_rank=5, max_rank=5, param_name="x")
    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(x_dtype, ("int8",), param_name="x")


def _reform_compute_generate(tensor, in_shape, out_shape, scale_val):
    """
    generate lambda func

    Parameters
    ----------
    tensor : input tensor
    in_shape : the shape of input tensor
    out_shape :the shape of output tensor
    scale_val : the value of scale

    Returns
    -------
    res lambda_func
    """
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    n_dim = len(in_shape)

    def lambda_func(*indice):
        new_indice = [indice[0],
                      (indice[1] * out_shape[n_dim - 1]) // in_shape[n_dim - 1]] \
                     + list(indice[2:n_dim - 1]) \
                     + [(indice[1] * out_shape[n_dim - 1]) % in_shape[n_dim - 1] + indice[n_dim - 1]]

        return tensor(*new_indice) * scale_val

    return lambda_func


def _reform_by_vmuls(input_tensor, input_shape, output_shape, scale_val):
    """
    5 dim input tensor C0 change

    Parameters
    ----------
    input_tensor : input tensor
    input_shape : the shape of input tensor
    output_shape :the shape of output tensor
    scale_val : the value of scale

    Returns
    -------
    res tensor
    """
    vmuls_vector = tvm.compute(output_shape,
                               _reform_compute_generate(input_tensor, input_shape, output_shape, scale_val),
                               name="reform_by_vmuls", tag="antiquant_reform_by_vmuls")

    return vmuls_vector


@register_operator("AscendAntiQuant", pattern="AscendAntiQuant")
def ascend_anti_quant_compute(x, y, scale, offset, sqrt_mode=False, kernel_name="ascend_anti_quant"):
    """
    int8 -> float16

    Parameters:
    ----------
    x : the tensor of input
    y : the dict of output
    scale : the data of scale
    offset : the data of offset
    sqrt_mode : the sqrt mode when true the result to do sqrt
    kernel_name : cce kernel name, default value is "ascend_anti_quant"

    Returns:
    -------
    None
    """
    in_shape = shape_util.shape_to_list(x.shape)
    out_shape = (in_shape[0], in_shape[1] * 2, in_shape[2], in_shape[3] // 2)

    input_ub = tvm.compute(in_shape, lambda *i: x(*i), name="input_ub", tag="antiquant_input_ub")
    # cast int8 to fp16
    cast_f16_ub = tvm.compute(in_shape, lambda *indice: shape_util.cast(input_ub(*indice), "float16"),
                              name="cast_f16_ub", tag="antiquant_cast_f16_ub")

    # add offset
    offset_value = tvm.const(offset, "float16")
    offset_ub = tvm.compute(in_shape, lambda *indice: cast_f16_ub(*indice) + offset_value,
                            name="offset_ub", tag="antiquant_offset_ub")

    scale_value = tvm.const(scale, "float16")
    if sqrt_mode:
        scale_sqrt_ub = tvm.compute(in_shape, lambda *indice: offset_ub(*indice) * scale_value,
                                    name="scale_sqrt_ub", tag="antiquant_scale_sqrt_ub")
        scale_ub = _reform_by_vmuls(scale_sqrt_ub, in_shape, out_shape, scale_value)
    else:
        # mul scale and convert 32 to 16 of C0
        scale_ub = _reform_by_vmuls(offset_ub, in_shape, out_shape, scale_value)

    ori_shape = y.get("ori_shape")
    ori_format = y.get("ori_format")
    if ori_format == "NHWC":
        ori_shape = [ori_shape[0], ori_shape[3], ori_shape[1], ori_shape[2]]

    ori_c = ori_shape[1]
    # remove pad
    if 0 < ori_c % 32 <= 16:
        tmp_res = tvm.compute(out_shape, lambda *indice: scale_ub(*indice), name="tmp_res", tag="antiquant_tmp_res")

        align_shape = [ori_shape[0], (ori_shape[1] + util.FP16_BLOCK_VALUE - 1) // util.FP16_BLOCK_VALUE,
                       ori_shape[2] * ori_shape[3], util.FP16_BLOCK_VALUE]

        res = tvm.compute(align_shape, lambda *indice: tmp_res(*indice), name="res", tag="antiquant_res",
                          attrs={"scale": scale,
                                 "sqrt_mode": sqrt_mode,
                                 "offset": offset})
    else:
        res = tvm.compute(out_shape, lambda *indice: scale_ub(*indice), name="res", tag="antiquant_res",
                          attrs={"scale": scale,
                                 "sqrt_mode": sqrt_mode,
                                 "offset": offset})

    return res


# 'pylint: disable=too-many-arguments,invalid-name,unused-argument
@register_operator("AscendAntiQuant", pattern="anti_quant")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def ascend_anti_quant(x, y, scale, offset, sqrt_mode=False, kernel_name="ascend_anti_quant"):
    """
    int8 -> float16

    Parameters:
    ----------
    x : the dict of input, format is NC1HWC0
    y : the dict of output, format is NC1HWC0
    scale : the data of scale
    offset : the data of offset
    sqrt_mode : the sqrt mode when true the result to do sqrt
    kernel_name : cce kernel name, default value is "ascend_anti_quant"

    Returns:
    -------
    None
    """
    _check_params(x, y, scale, offset, sqrt_mode, kernel_name)
    input_dtype = x.get("dtype").lower()

    with tbe.compute():
        input_shape = []
        n = operation.var("n")
        c1 = operation.var("c1")
        hw = operation.var("hw")
        input_shape.append(n)
        input_shape.append(c1)
        input_shape.append(hw)
        input_shape.append(32)

        input_x = tvm.placeholder(input_shape, name="input_x", dtype=input_dtype)
        res = ascend_anti_quant_compute(input_x, y, scale, offset, sqrt_mode, kernel_name)

    schedules = []
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    schedules.append(sch)

    tensor_list = [input_x, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(schedules, config)
