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
ascend_quant
"""
import functools
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl import ascend_quant_util as quant_util


# pylint: disable=too-many-arguments,invalid-name,unused-argument,unnecessary-lambda,too-many-locals
@register_operator_compute("AscendQuant", op_mode="dynamic", support_fusion=True)
def ascend_quant_compute(x, y, scale, offset, sqrt_mode=False, round_mode="Round",
                         kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the tensor of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    dtype = x.dtype
    in_shape = shape_util.shape_to_list(x.shape)
    nz_format_flag = quant_util.is_nz_format(x, True)

    c1_dim = in_shape[1]
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4
        c1_dim = in_shape[c1_index]

    read_shape, out_shape = _get_shape_info(in_shape, nz_format_flag)

    input_ub = _input_compute_generate(x, in_shape, read_shape, c1_dim, c1_index)

    if dtype == "float32":
        cast_f16_ub = tvm.compute(read_shape, lambda *indice: shape_util.cast(input_ub(*indice), "float16"),
                                  name="cast_f16_ub", tag="cast_f16_ub")
        cast_i8_ub = _compute_scale(cast_f16_ub, in_shape, out_shape, (scale, offset, sqrt_mode), nz_format_flag)
    else:
        cast_i8_ub = _compute_scale(input_ub, in_shape, out_shape, (scale, offset, sqrt_mode), nz_format_flag)

    res = tvm.compute(out_shape, lambda *indice: cast_i8_ub(*indice), name="res", tag="quant",
                      attrs={"scale": scale,
                             "sqrt_mode": sqrt_mode,
                             "offset": offset,
                             "round_mode": round_mode})
    return res


def _check_params(x, y, scale, offset, sqrt_mode, round_mode, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr
    """
    shape = x.get("shape")
    x_format = x.get("format")
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_format(x_format, format_list, param_name="x")
    if x_format == "NC1HWC0":
        para_check.check_shape(shape, min_rank=5, max_rank=5, param_name="x")
    if x_format == "FRACTAL_NZ":
        para_check.check_shape(shape, min_rank=4, param_name="x")
    para_check.check_shape(shape, param_name="x")

    round_mode_list = ["Round", "Ceil", "Floor", "Trunc"]
    if round_mode not in round_mode_list:
        rule = "round_mode only support [Round, Ceil, Floor, Trunc]"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "round_mode", round_mode)


def _get_shape_info(in_shape, nz_format_flag):
    """
    the compute of scale

    Parameters
    ----------
    in_shape: the shape of input tensor
    nz_format_flag: the format of output tensor

    Returns
    -------
    read_shape, out_shape
    """
    c0_index = len(in_shape) - 1
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4

    out_shape = in_shape[:]
    read_shape = in_shape[:]
    read_shape[c1_index] = (read_shape[c1_index] + 1) // 2 * 2

    for dim, _ in enumerate(in_shape):
        if dim == c0_index:
            out_shape[dim] = in_shape[dim] * 2
        if dim == c1_index:
            out_shape[dim] = (in_shape[dim] + 1) // 2
    return read_shape, out_shape


def _input_compute_generate(x, in_shape, read_shape, c1_dim, c1_index):
    """
    generate lambda func
    """
    dtype = x.dtype
    zero = tvm.const(0, dtype=dtype)
    c1_is_var = bool(isinstance(c1_dim, tvm.expr.Var))
    if not c1_is_var and c1_dim % 2 == 0:
        res = tvm.compute(in_shape, lambda *i: x(*i), name="input_ub", tag="input_ub", attrs={"c_out": c1_dim})
    else:
        input_ub = tvm.compute(read_shape,
                               lambda *indice: tvm.select(indice[c1_index] <= in_shape[c1_index] - 1, x(*indice)),
                               name="input_ub", tag="input_ub", attrs={"c_out": c1_dim})
        padding_ub = tvm.compute(read_shape,
                                 lambda *indice: tvm.select(indice[c1_index] > in_shape[c1_index] - 1, zero),
                                 name="padding_ub", tag="quant_padding", attrs={"c_out": c1_dim})
        res = tvm.compute(read_shape, lambda *indice: padding_ub(*indice) + input_ub(*indice),
                          name="add_ub", tag="quant_add", attrs={"c_out": c1_dim})
    return res


def _reform_compute_generate(tensor, in_shape, out_shape, val_info, nz_format_flag):
    """
    generate lambda func

    Parameters
    ----------
    tensor: input tensor
    in_shape: the shape of input tensor
    out_shape: the shape of output tensor
    val_info: the val info of offset,scale
    nz_format_flag: the format of input tensor

    Returns
    -------
    res lambda_func
    """
    in_shape = list(in_shape)
    out_shape = list(out_shape)
    n_dim = len(in_shape)

    c0_index = n_dim - 1
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4

    def lambda_func(*indice):
        """
        c1,c0 reform compute
        """
        new_indice = [0] * n_dim
        for i in range(n_dim):
            if i == c0_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] + indice[c0_index]) % in_shape[c0_index]
            elif i == c1_index:
                new_indice[i] = (indice[c1_index] * out_shape[c0_index] + indice[c0_index]) // in_shape[c0_index]
            else:
                new_indice[i] = indice[i]

        if val_info[0]:
            return tensor(*new_indice) + val_info[1]

        return tensor(*new_indice) * val_info[2]

    return lambda_func


def _reform_by_vadds(input_tensor, input_shape, output_shape, offset_val, nz_format_flag):
    """
    5 dim input tensor C0 change

    Parameters
    ----------
    input_tensor: input tensor
    input_shape: the shape of input tensor
    output_shape: the shape of output tensor
    offset_val: the val of offset
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    vadds_vector = tvm.compute(output_shape,
                               _reform_compute_generate(input_tensor, input_shape, output_shape,
                                                        (True, offset_val, -1), nz_format_flag),
                               name="reform_by_vadds", tag="reform_by_vadds")

    return vadds_vector


def _reform_by_vmuls(input_tensor, input_shape, output_shape, scale_val, nz_format_flag):
    """
    5 dim input tensor C0 change

    Parameters
    ----------
    input_tensor: input tensor
    input_shape: the shape of input tensor
    output_shape: the shape of output tensor
    scale_val: the val of scale
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    vmuls_vector = tvm.compute(output_shape,
                               _reform_compute_generate(input_tensor, input_shape, output_shape,
                                                        (False, -1, scale_val), nz_format_flag),
                               name="reform_by_vmuls", tag="reform_by_vmuls")

    return vmuls_vector


def _compute_scale(in_tensor, in_shape, out_shape, attr_list, nz_format_flag):
    """
    the compute of scale

    Parameters
    ----------
    in_tensor: input tensor
    in_shape: the shape of input tensor
    out_shape: the shape of output tensor
    attr_list: the attr list
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    scale = attr_list[0]
    offset = attr_list[1]
    sqrt_mode = attr_list[2]
    if scale != 1:
        scale_value = tvm.const(scale, "float16")
        scale_ub = _reform_by_vmuls(in_tensor, in_shape, out_shape, scale_value, nz_format_flag)
        if sqrt_mode:
            scale_sqrt_ub = tvm.compute(out_shape, lambda *indice: scale_ub(*indice) * scale_value,
                                        name="scale_sqrt_ub", tag="scale_sqrt_ub")
            res = _compute_offset(scale_sqrt_ub, in_shape, out_shape, (offset, False, scale), nz_format_flag)
        else:
            res = _compute_offset(scale_ub, in_shape, out_shape, (offset, False, scale), nz_format_flag)
    else:
        res = _compute_offset(in_tensor, in_shape, out_shape, (offset, True, scale), nz_format_flag)
    return res


def _compute_offset(in_tensor, in_shape, out_shape, attr_list, nz_format_flag):
    """
    the compute of scale

    Parameters
    ----------
    in_tensor: input tensor
    in_shape: the shape of input tensor
    out_shape: the shape of output tensor
    attr_list: the attr list
    nz_format_flag: the format of input tensor

    Returns
    -------
    res tensor
    """
    offset = attr_list[0]
    reform_flag = attr_list[1]
    scale = attr_list[2]
    if offset != 0 or scale == 1:
        offset_value = tvm.const(offset, "float16")
        if reform_flag:
            offset_ub = _reform_by_vadds(in_tensor, in_shape, out_shape, offset_value, nz_format_flag)
        else:
            offset_ub = tvm.compute(out_shape, lambda *indice: in_tensor(*indice) + offset_value,
                                    name="offset_ub", tag="offset_ub")
        cast_i8_ub = tvm.compute(out_shape,
                                 lambda *indice: shape_util.cast(offset_ub(*indice), "int8"),
                                 name='cast_i8_ub', tag="cast_i8_ub")
    else:
        cast_i8_ub = tvm.compute(out_shape,
                                 lambda *indice: shape_util.cast(in_tensor(*indice), "int8"),
                                 name='cast_i8_ub', tag="cast_i8_ub")
    return cast_i8_ub


def _get_variable_shape(x):
    """
    get dynamic shape info
    """
    x_format = x.get("format")
    x_shape = x.get("shape")
    x_range = x.get("range")
    d_shape = []
    var_index_list = []

    def _get_range(start_dim, end_dim, _range, index):
        res = 1
        for _i in range(start_dim, end_dim):
            if _range[_i][index] is None:
                res = None
                return res
            res *= _range[_i][index]
        return res

    def _get_shape(start_dim, end_dim, _shape, _range, _suffix):
        flag1 = True
        flag2 = False
        for k in range(start_dim, end_dim):
            flag1 = flag1 and _shape[k] == -1 and _range[k][0] == _range[k][1]
            flag2 = flag2 or _shape[k] == -1
        if flag1:
            val = functools.reduce(lambda x, y: x * y, _range[start_dim:end_dim][0])
            d_shape.append(val)
        elif flag2:
            range_start = _get_range(start_dim, end_dim, _range, 0)
            range_end = _get_range(start_dim, end_dim, _range, 1)
            var = operation.var_inner("_dim_" + str(_suffix), (range_start, range_end))
            d_shape.append(var)
            var_index_list.append(_suffix)
        else:
            val = functools.reduce(lambda x, y: x * y, _shape[start_dim:end_dim])
            d_shape.append(val)

    suffix = 0
    if x_format == "NC1HWC0":
        for i in range(len(x_shape)):
            # n,c1
            if i < 2:
                _get_shape(i, i + 1, x_shape, x_range, suffix)
            # h*w
            elif i == 2:
                _get_shape(i, i + 2, x_shape, x_range, suffix)
            # c0
            elif i == len(x_shape) - 1:
                d_shape.append(16)
            suffix += 1
    else:
        # n
        if len(x_shape) > 4:
            _get_shape(0, len(x_shape) - 4, x_shape, x_range, suffix)
        else:
            d_shape.append(1)
        suffix += 1
        for i in range(len(x_shape)):
            # c1
            if i == len(x_shape) - 4:
                _get_shape(i, i + 1, x_shape, x_range, suffix)
            # h*w
            elif i == len(x_shape) - 3:
                _get_shape(i, i + 2, x_shape, x_range, suffix)
            # c0
            elif i == len(x_shape) - 1:
                d_shape.append(16)
            suffix += 1
    return d_shape, var_index_list


@register_operator("AscendQuant", pattern="quant")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def ascend_quant(x, y, scale, offset, sqrt_mode=False, round_mode="Round", kernel_name="ascend_quant"):
    """
    float16/float32 -> int8

    Parameters:
    ----------
    x : the dict of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    _check_params(x, y, scale, offset, sqrt_mode, round_mode, kernel_name)
    input_dtype = x.get("dtype").lower()

    with tbe.compute():
        input_shape, var_index_list = _get_variable_shape(x)
        tbe_context.get_context().add_compile_info("var_index_list", var_index_list)
        input_x = tvm.placeholder(input_shape, name="input_x", dtype=input_dtype)
        res = ascend_quant_compute(input_x, y, scale, offset, sqrt_mode, round_mode, kernel_name)

    schedules = []
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    schedules.append(sch)

    tensor_list = [input_x, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(schedules, config)
