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
ascend_quant
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import shape_util
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl import ascend_quant_util as util


# pylint: disable=too-many-arguments,invalid-name,unused-argument,unnecessary-lambda,too-many-locals
def _check_params(x, y, scale, offset, sqrt_mode, round_mode, dst_type, kernel_name):
    """
    check the parameters including shape, dtype, kernel_name, attr
    """
    shape = x.get("shape")
    x_format = x.get("format")
    dtype = x.get("dtype").lower()
    format_list = ["NC1HWC0", "FRACTAL_NZ"]
    para_check.check_format(x_format, format_list, param_name="x")
    if x_format == "NC1HWC0":
        para_check.check_shape(shape, min_rank=5, max_rank=5, param_name="x")
    if x_format == "FRACTAL_NZ":
        para_check.check_shape(shape, min_rank=4, param_name="x")

    para_check.check_shape(shape, param_name="x")

    if tbe_platform.cce_conf.is_lhisi_version():
        check_list = ["float16"]
    else:
        check_list = ["float16", "float32"]
    para_check.check_dtype(dtype, check_list, param_name="x")

    round_mode_list = ["Round", "Ceil", "Floor", "Trunc"]
    if round_mode not in round_mode_list:
        rule = "round_mode only support [Round, Ceil, Floor, Trunc]"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "round_mode", round_mode)

    y_dtype = _dst_type_conversion(dst_type)
    y_check_list = ["int8", "int4"]
    para_check.check_dtype(y_dtype, y_check_list, param_name="dst_type")


def _check_l1_fusion(x, y):
    """
    check the l1 fusion parameters
    """
    x_addr_type = x.get("addr_type", 0)
    x_l1_fusion_type = x.get("L1_fusion_type", -1)
    y_l1_fusion_type = y.get("L1_fusion_type", -1)

    if x_l1_fusion_type not in (-1, 0):
        rule = "quant L1_fusion_type only  support (-1, 0)"
        error_manager_vector.raise_err_check_params_rules("ascend_quant", rule, "L1_fusion_type", x_l1_fusion_type)

    if y_l1_fusion_type not in (-1, 0):
        rule = "quant L1_fusion_type only  support (-1, 0)"
        error_manager_vector.raise_err_check_params_rules("ascend_quant", rule, "L1_fusion_type", y_l1_fusion_type)

    attr = {"addr_type": x_addr_type,
            "L1_fusion_type": x_l1_fusion_type}

    return x_l1_fusion_type, y_l1_fusion_type, attr


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


def _input_compute_generate(x, in_shape, read_shape, c1_dim, c1_index, c1_transform):
    """
    generate lambda func
    """
    dtype = x.dtype
    if c1_dim % c1_transform == 0:
        input_ub = tvm.compute(
            in_shape, lambda *i: x(*i), name="input_ub", attrs={"c_out": c1_dim, "c1_transform": c1_transform})
    else:
        zero = tvm.const(0, dtype=dtype)
        input_ub = tvm.compute(read_shape,
                               lambda *indice: tvm.select(indice[c1_index] <= in_shape[c1_index] - 1, x(*indice), zero),
                               name='input_ub', attrs={"c_out": c1_dim, "c1_transform": c1_transform})
    return input_ub


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
                               name='reform_by_vadds')

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
                               name='reform_by_vmuls')

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
    y_dtype = attr_list[3]
    if scale != 1:
        scale_value = tvm.const(scale, "float16")
        scale_ub = _reform_by_vmuls(in_tensor, in_shape, out_shape, scale_value, nz_format_flag)
        if sqrt_mode:
            scale_sqrt_ub = tvm.compute(out_shape, lambda *indice: scale_ub(*indice) * scale_value,
                                        name="scale_sqrt_ub")
            res = _compute_offset(scale_sqrt_ub, in_shape, out_shape, (offset, False, scale, y_dtype), nz_format_flag)
        else:
            res = _compute_offset(scale_ub, in_shape, out_shape, (offset, False, scale, y_dtype), nz_format_flag)
    else:
        res = _compute_offset(in_tensor, in_shape, out_shape, (offset, True, scale, y_dtype), nz_format_flag)
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
    y_dtype = attr_list[3]
    if offset != 0 or scale == 1:
        offset_value = tvm.const(offset, "float16")
        if reform_flag:
            offset_ub = _reform_by_vadds(in_tensor, in_shape, out_shape, offset_value, nz_format_flag)
        else:
            offset_ub = tvm.compute(out_shape, lambda *indice: in_tensor(*indice) + offset_value, name="offset_ub")
        if y_dtype == "int8":
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(offset_ub(*indice), "int8"), name='cast_i8_ub')
        else:
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(offset_ub(*indice), "int4"), name='cast_i4_ub')
    else:
        if y_dtype == "int8":
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(in_tensor(*indice), "int8"), name='cast_i8_ub')
        else:
            res = tvm.compute(out_shape,
                              lambda *indice: shape_util.cast(in_tensor(*indice), "int4"), name='cast_i4_ub')
    return res


def _get_shape_info(in_shape, nz_format_flag, c1_transform):
    """
    the compute of scale
    """
    c0_index = len(in_shape) - 1
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4
    out_shape = in_shape[:]
    read_shape = in_shape[:]
    read_shape[c1_index] = (read_shape[c1_index] + c1_transform - 1) // c1_transform * c1_transform
    for dim, _ in enumerate(in_shape):
        if dim == c0_index:
            out_shape[dim] = in_shape[dim] * c1_transform
        if dim == c1_index:
            out_shape[dim] = (in_shape[dim] + c1_transform - 1) // c1_transform
    return read_shape, out_shape


def _get_input_attr(x, attr_name, default_value, is_list):
    """
    get the attrs of input tensor
    """
    value = default_value
    if x.op.attrs:
        if attr_name in x.op.attrs:
            if is_list:
                value = x.op.attrs[attr_name]
            else:
                value = x.op.attrs[attr_name].value
    return value


def _get_input_l1_info(x):
    """
    get the l1 fusion info from input tensor
    """
    l1_fusion_flag = _get_input_attr(x, "l1_fusion_flag", -1, False)
    in_shape = shape_util.shape_to_list(x.shape)
    return in_shape, l1_fusion_flag


def _get_out_l1_info(y):
    """
    get the l1 fusion info from output tensor
    """
    y_addr_type = 0
    if isinstance(y, dict):
        y_addr_type = y.get("addr_type", 0)
    elif isinstance(y, tvm.tensor.Tensor):
        y_addr_type = _get_input_attr(y, "addr_type", 0, False)
    return y_addr_type


@tbe_platform.fusion_manager.fusion_manager.register("ascend_quant")
def ascend_quant_compute(x, y, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2,
                         kernel_name="ascend_quant"):
    """
    float16/float32 -> int8/int4

    Parameters:
    ----------
    x : the tensor of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    dst_type : the output data type

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    x_dtype = x.dtype
    in_shape, l1_fusion_flag = _get_input_l1_info(x)
    y_addr_type = _get_out_l1_info(y)

    nz_format_flag = util.is_nz_format(x, True)

    tensor_format = "NC1HWC0"
    if x.op.attrs:
        if "format" in x.op.attrs:
            tensor_format = x.op.attrs["format"]

    c1_dim = in_shape[1]
    c1_index = 1
    if nz_format_flag:
        c1_index = len(in_shape) - 4
        c1_dim = in_shape[c1_index]

    y_dtype = _dst_type_conversion(dst_type)
    c1_transform = 2
    if y_dtype == "int4":
        c1_transform = 4

    read_shape, out_shape = _get_shape_info(in_shape, nz_format_flag, c1_transform)

    input_ub = _input_compute_generate(x, in_shape, read_shape, c1_dim, c1_index, c1_transform)
    attr_list = (scale, offset, sqrt_mode, y_dtype)
    if x_dtype == "float32":
        cast_f16_ub = tvm.compute(read_shape, lambda *indice: shape_util.cast(input_ub(*indice), "float16"),
                                  name="cast_f16_ub")
        cast_i8_or_i4_ub = _compute_scale(cast_f16_ub, in_shape, out_shape, attr_list, nz_format_flag)
    else:
        cast_i8_or_i4_ub = _compute_scale(input_ub, in_shape, out_shape, attr_list, nz_format_flag)
    res = tvm.compute(out_shape, lambda *indice: cast_i8_or_i4_ub(*indice), name="res", tag="quant",
                      attrs={"scale": scale,
                             "sqrt_mode": sqrt_mode,
                             "offset": offset,
                             "round_mode": round_mode,
                             "input_format": tensor_format,
                             "c1_dim": c1_dim,
                             "l1_fusion_flag": l1_fusion_flag,
                             "c1_transform": c1_transform,
                             "addr_type": y_addr_type})
    return res


def get_op_support_info(x, y, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2,
                        kernel_name="ascend_quant"):
    """
    get split info
    """
    return util.get_quant_support_info(x, l1_fusion_enable=1)


def _dst_type_conversion(dst_type):
    """
    convert dst_type from int to string
    """
    dst_type_str = ""
    if dst_type == 2:
        dst_type_str = "int8"
    elif dst_type == 29:
        dst_type_str = "int4"
    return dst_type_str


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def ascend_quant(x, y, scale, offset, sqrt_mode=False, round_mode="Round", dst_type=2, kernel_name="ascend_quant"):
    """
    float16/float32 -> int8/int4

    Parameters:
    ----------
    x : the dict of input

    y : the dict of output

    scale : the data of scale

    offset : the data of offset

    sqrt_mode : the sqrt mode when true the result to do sqrt

    round_mode : the data conversion mode

    dst_type : the output data type

    kernel_name : cce kernel name, default value is "ascend_quant"

    Returns:
    -------
    None
    """
    _check_params(x, y, scale, offset, sqrt_mode, round_mode, dst_type, kernel_name)
    shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    input_format = x.get("format")

    x_l1_fusion_type, y_l1_fusion_type, attr = _check_l1_fusion(x, y)

    if input_format == "NC1HWC0":
        if x_l1_fusion_type != -1:
            attr["l1_fusion_flag"] = x_l1_fusion_type
        # change to N,C1,H*W,C0
        input_shape = (shape[0], shape[1], shape[2] * shape[3], shape[4])
    else:
        if x_l1_fusion_type != -1:
            rule = "FRACTAL_NZ not support L1 fusion"
            error_manager_vector.raise_err_check_params_rules(kernel_name, rule, "L1_fusion_type", x_l1_fusion_type)
        batch = 1
        if len(shape) > 4:
            batch = functools.reduce(lambda x, y: x * y, shape[:-4])
        input_shape = (batch, shape[-4], shape[-3] * shape[-2], shape[-1])
    input_x = tvm.placeholder(input_shape, name="input_x", dtype=input_dtype, attrs=attr)

    res = ascend_quant_compute(input_x, y, scale, offset, sqrt_mode, round_mode, dst_type, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_x, res]}

    tbe.cce_build_code(sch, config)
