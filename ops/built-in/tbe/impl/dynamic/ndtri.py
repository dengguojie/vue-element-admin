"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ndtri
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CENTRAL_RANGE = 0.7
    PI = 3.1415926535
    TWODIVPI = 1.1283791670955

    # The ratio needed for numerical calculation.
    # The detailed calculation process will be given in the code comments below.
    SCALAR_P = 0.47047
    SCALAR_A = 0.3480242
    SCALAR_B = -0.0958798
    SCALAR_C = 0.7478556
    SCALER_FP16_MAX = 32768
    SCALER_FP16_MIN = 2**(-15)
    SCALAR_FP32_MIN = 2**(-126)

    LIST_A = (0.886226899, -1.645349621, 0.914624893, -0.140543331)
    LIST_B = (-2.118377725, 1.442710462, -0.329097515, 0.012229801)
    LIST_C = (-1.970840454, -1.624906493, 3.429567803, 1.641345311)
    LIST_D = (3.543889200, 1.637067800)


# 'pylint: disable=unused-argument,too-many-locals
def erfinv_compute(input_x):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype

    x_abs = tbe.vabs(input_x)

    # yl is the value of y when input_x is less equal CENTRAL_RANGE
    res_yl = cal_yl(input_x)

    # yg is the value of y when input_x is greater than CENTRAL_RANGE
    res_yg = cal_yg(input_x)

    res_y = tbe.vcmpsel(x_abs, Constant.CENTRAL_RANGE, 'le', res_yl, res_yg)

    # Two steps of Newton-Raphson correction
    for _ in range(0, 2):
        erf_result = erf(res_y)

        num = tbe.vsub(erf_result, input_x)
        res_mul = tbe.vmul(res_y, res_y)
        res_muls = tbe.vmuls(res_mul, -1)
        res_exp = tbe.vexp(res_muls)
        dem = tbe.vmuls(res_exp, Constant.TWODIVPI)
        crt = tbe.vdiv(num, dem)
        res = tbe.vsub(res_y, crt)

    return res


# 'pylint: disable=too-many-locals
def cal_yl(input_x):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    const_one = tvm.const(1, dtype="float32")
    res_zl = tbe.vmul(input_x, input_x)

    # `num = ((a[3]*z + a[2])*z + a[1])*z + a[0]`
    erfinv_vmuls_a = tbe.vmuls(res_zl, Constant.LIST_A[3])
    erfinv_vadds_a = tbe.vadds(erfinv_vmuls_a, Constant.LIST_A[2])
    erfinv_square_vmul_a = tbe.vmul(erfinv_vadds_a, res_zl)
    erfinv_square_vadds_a = tbe.vadds(erfinv_square_vmul_a, Constant.LIST_A[1])
    erfinv_cube_vmul_a = tbe.vmul(erfinv_square_vadds_a, res_zl)
    num = tbe.vadds(erfinv_cube_vmul_a, Constant.LIST_A[0])

    # `dem = (((b[3]*z + b[2])*z + b[1])*z + b[0])*z + 1`
    erfinv_vmuls_b = tbe.vmuls(res_zl, Constant.LIST_B[3])
    erfinv_vadds_b = tbe.vadds(erfinv_vmuls_b, Constant.LIST_B[2])
    erfinv_square_vmul_b = tbe.vmul(erfinv_vadds_b, res_zl)
    erfinv_square_vadds_b = tbe.vadds(erfinv_square_vmul_b, Constant.LIST_B[1])
    erfinv_cube_vmul_b = tbe.vmul(erfinv_square_vadds_b, res_zl)
    erfinv_cube_vadds_b = tbe.vadds(erfinv_cube_vmul_b, Constant.LIST_B[0])
    erfinv_power4_vmul_b = tbe.vmul(erfinv_cube_vadds_b, res_zl)
    dem = tbe.vadds(erfinv_power4_vmul_b, const_one)

    # `yl = input_x * numl / deml`
    xnum = tbe.vmul(input_x, num)
    res_yl = tbe.vdiv(xnum, dem)

    return res_yl


# 'pylint: disable=too-many-locals
def cal_sign(input_x, dtype):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    fp16_max = tvm.const(Constant.SCALER_FP16_MAX, dtype=dtype)
    fp16_min = tvm.const(Constant.SCALER_FP16_MIN, dtype=dtype)

    data_vmuls = tbe.vmuls(input_x, fp16_max)
    data_abs = tbe.vabs(data_vmuls)
    data_vadds = tbe.vadds(data_abs, fp16_min)
    data_div = tbe.vdiv(data_vmuls, data_vadds)
    data_round = tbe.round(data_div)
    tensor_sign = tbe.cast_to(data_round, dtype)

    return tensor_sign


# 'pylint: disable=too-many-locals
def cal_yg(input_x):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    dtype = "float32"
    const_one = tvm.const(1, dtype="float32")
    const_negative_one = tvm.const(-1, dtype="float32")
    x_abs = tbe.vabs(input_x)

    # `zg = sqrt(-log((1-|x|)/2))`
    x_abs_minus_one = tbe.vadds(x_abs, const_negative_one)
    data_neg = tbe.vmuls(x_abs_minus_one, -1)
    mul_data = tbe.vmuls(data_neg, 0.5)
    data_vlog = tbe.vlog(mul_data, 1)
    zg_square = tbe.vabs(data_vlog)
    res_zg = tbe.vsqrt(zg_square, 1)

    # `numg = ((c[3]*z + c[2])*z + c[1])*z + c[0]`
    zg_vmuls_c3 = tbe.vmuls(res_zg, Constant.LIST_C[3])
    lr_vadds_c2 = tbe.vadds(zg_vmuls_c3, Constant.LIST_C[2])
    lr_vmul_zg = tbe.vmul(lr_vadds_c2, res_zg)
    lr_vadds_c1 = tbe.vadds(lr_vmul_zg, Constant.LIST_C[1])
    lr_vmul_zg = tbe.vmul(lr_vadds_c1, res_zg)
    numg = tbe.vadds(lr_vmul_zg, Constant.LIST_C[0])

    # `demg = (d[1]*z + d[0])*z + 1`
    zg_vmuls_d1 = tbe.vmuls(res_zg, Constant.LIST_D[1])
    lr_vadds_d0 = tbe.vadds(zg_vmuls_d1, Constant.LIST_D[0])
    lr_vmul_zg = tbe.vmul(lr_vadds_d0, res_zg)
    demg = tbe.vadds(lr_vmul_zg, const_one)
    tensor_sign = cal_sign(input_x, dtype)

    # `yg = copysign(numg, input_x) / demg`
    numg_sign = tbe.vmul(numg, tensor_sign)
    res_yg = tbe.vdiv(numg_sign, demg)

    return res_yg


# 'pylint: disable=too-many-locals
def erf(input_x):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    dtype = "float32"
    shape = shape_util.shape_to_list(input_x.shape)
    const_one = tvm.const(1, dtype="float32")
    const_negative_one = tvm.const(-1, dtype="float32")
    const_p = tvm.const(Constant.SCALAR_P, dtype="float32")
    const_a = tvm.const(Constant.SCALAR_A, dtype="float32")
    const_b = tvm.const(Constant.SCALAR_B, dtype="float32")
    const_c = tvm.const(Constant.SCALAR_C, dtype="float32")

    tensor_sign = cal_sign(input_x, dtype)
    tensor_one = tbe.broadcast(const_one, shape, "float32")
    tensor_abs = tbe.vabs(input_x)
    erf_t_vmuls = tbe.vmuls(tensor_abs, const_p)
    erf_t_vadds = tbe.vadds(erf_t_vmuls, const_one)
    erf_data_t = tbe.vdiv(tensor_one, erf_t_vadds)

    erf_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erf_data_vmuls = tbe.vmuls(erf_abs_square, const_negative_one)
    erf_data_exp = tbe.vexp(erf_data_vmuls)

    erf_data_t_square = tbe.vmul(erf_data_t, erf_data_t)
    erf_data_t_cube = tbe.vmul(erf_data_t, erf_data_t_square)

    erf_t_vmuls = tbe.vmuls(erf_data_t, const_a)
    erf_t_square_vmuls = tbe.vmuls(erf_data_t_square, const_b)
    erf_t_cube_vmuls = tbe.vmuls(erf_data_t_cube, const_c)

    erf_square_vadd = tbe.vadd(erf_t_vmuls, erf_t_square_vmuls)
    erf_cube_vadd_ = tbe.vadd(erf_square_vadd, erf_t_cube_vmuls)
    erf_cube_vmuls = tbe.vmuls(erf_cube_vadd_, const_negative_one)
    erf_exp_vmul = tbe.vmul(erf_cube_vmuls, erf_data_exp)
    erf_exp_vadds = tbe.vadds(erf_exp_vmul, const_one)
    erf_result = tbe.vmul(tensor_sign, erf_exp_vadds)

    return erf_result


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
@register_operator_compute("Ndtri", op_mode="dynamic", support_fusion=True)
def ndtri_compute(input_x, output_y, kernel_name="ndtri"):
    """
    compute ndtri, `y = sqrt(2) * erfinv(2 * x - 1)`.

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "ndtri"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype.lower()
    shape = shape_util.shape_to_list(input_x.shape)

    has_improve_precision = False
    # Change dtype to float32
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vsqrt", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        dtype = "float32"

    # `sqrt(2)`
    res_two = tbe.broadcast(tvm.const(2, dtype), shape)
    res_sqrt = tbe.vsqrt(res_two)

    # `sqrt(2) * erfinv(2 * x - 1)`
    res_muls = tbe.vmuls(input_x, 2)
    res_one = tbe.broadcast(tvm.const(1, dtype), shape)
    res_sub = tbe.vsub(res_muls, res_one)
    res_erfinv = erfinv_compute(res_sub)
    res_ndtri = tbe.vmul(res_sqrt, res_erfinv)

    # x belongs to (-inf, -1] and [1, inf] equl to maxnum
    zeros = tbe.vmuls(input_x, 0)
    res_maxnum = tbe.vrec(zeros)
    tmp_one = tbe.vcmpsel(input_x, tvm.const(1, dtype), 'ge', res_maxnum, res_ndtri)
    res = tbe.vcmpsel(input_x, tvm.const(0, dtype), 'le', res_maxnum, tmp_one)

    # Restore dtype
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("Ndtri")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ndtri(input_x, output_y, kernel_name="ndtri"):
    """
    Computes ndtri element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same type as input
    kernel_name: str
        kernel name, default value is "ndtri"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x_,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([x_])
            data_input = tvm.placeholder(x_shape[0], name="data_input", dtype=dtype_input)
            res = ndtri_compute(data_input, output_y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
