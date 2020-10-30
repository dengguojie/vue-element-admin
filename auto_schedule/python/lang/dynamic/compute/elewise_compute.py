# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
elewise compute
"""
from te import platform as cceconf
from te.platform.cce_conf import CceProductParams as product_params
from te import tvm

from .broadcast_compute import broadcast
from . import util
from . import cast_compute

NAME_INDEX = [0]


@util.dtype_check_decorator
def vmuls(raw_tensor, scalar):
    """
    multiply a tensor by a scalar, dtype of raw_tensor
    and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor*scalar
    """
    dtype = raw_tensor.dtype
    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_mul',
                               args=[scalar])


@util.dtype_check_decorator
def vadds(raw_tensor, scalar):
    """
    add a tensor by a scalar, dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : raw_tensor + scalar
    """
    dtype = raw_tensor.dtype
    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_add',
                               args=[scalar])


@util.dtype_check_decorator
def vmaxs(raw_tensor, scalar):
    """
    Calculate elewise compare,
    return the max one of scalar or tensor's element,
    dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : max(raw_tensor, scalar)
    """
    dtype = raw_tensor.dtype
    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_max',
                               args=[scalar])


@util.dtype_check_decorator
def vmins(raw_tensor, scalar):
    """
    Calculate elewise compare,
    return the min one of scalar or tensor's element,
     dtype of raw_tensor and scalar must be the same

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    scalar : float, int, or tvm const

    Returns
    -------
    wrapped_tensor : min(raw_tensor, scalar)
    """
    dtype = raw_tensor.dtype
    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be scalar")
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_min',
                               args=[scalar])


def __vlog_calculate_by_taylor(data_x):
    """
    calculate ln(raw_tensor), use taylor expansion to calculate log
    """
    # 'pylint: disable=too-many-locals, too-many-statements

    # Log threshold
    const_log_threshold_1 = 0.6666666666666667
    const_log_threshold_2 = 0.3333333333333333
    # Log value
    log_four_three = 0.28768207245178085
    log_five_three = 0.5108256237659907
    log_five_two = 0.916290731874155
    # const value
    const_neg_one = -1
    const_one = 1
    const_two = 2
    const_one_three = 0.3333333333333333
    const_half = 1.0 / 2
    const_three_four = 0.75
    const_one_five = 0.2
    const_one_four_neg = -0.25
    const_five_two = 0.4
    const_dot_six = 0.6
    float_16_max = 32768
    const_half_neg = -0.5

    const_one_nine = 0.111111111111111
    const_one_eight_neg = -0.125
    const_one_seven = 0.142857142857143
    const_one_six_neg = -0.166666666666667

    dtype = data_x.dtype
    shape = data_x.shape

    def _taylor_compute(data):
        # 'pylint: disable=too-many-locals
        taylor_nine = vmuls(data, tvm.const(const_one_nine, dtype))
        taylor_eight_1 = vadds(taylor_nine,
                               tvm.const(const_one_eight_neg, dtype))
        taylor_eight_2 = vmul(taylor_eight_1, data)
        taylor_seven_1 = vadds(taylor_eight_2,
                               tvm.const(const_one_seven, dtype))
        taylor_seven_2 = vmul(taylor_seven_1, data)
        taylor_six_1 = vadds(taylor_seven_2,
                             tvm.const(const_one_six_neg, dtype))
        taylor_six_2 = vmul(taylor_six_1, data)
        taylor_five_1 = vadds(taylor_six_2, tvm.const(const_one_five, dtype))
        taylor_five_2 = vmul(taylor_five_1, data)
        taylor_four_1 = vadds(taylor_five_2,
                              tvm.const(const_one_four_neg, dtype))
        taylor_four_2 = vmul(taylor_four_1, data)
        taylor_three_1 = vadds(taylor_four_2,
                               tvm.const(const_one_three, dtype))
        taylor_three_2 = vmul(taylor_three_1, data)
        taylor_two_1 = vadds(taylor_three_2,
                             tvm.const(const_half_neg, dtype))
        taylor_two_2 = vmul(taylor_two_1, data)
        taylor_one = vadds(taylor_two_2, tvm.const(const_one, dtype))
        taylor = vmul(taylor_one, data)

        return taylor

    def _log_compute_block_gt_2(data_x, res, shape):
        """
        when data > 2, use vlog directly
        when data > 32768, float16 will overflow, use log(x/2.5)+log(2.5)

        Parameters
        ----------
        data: input tensor that we want to calculate log

        Returns
        -------
        res : return of log

        """
        # 'pylint: disable=too-many-locals
        # if data > 2, use vlog
        threshold_3 = broadcast(tvm.const(const_two, dtype), shape)
        res = vcmpsel(data_x, threshold_3, 'ge', vlog(data_x), res)
        # if data > 32768, use log(x/2.5)+log(2.5)
        float_16_max_tensor = broadcast(tvm.const(float_16_max, dtype), shape)
        overflow_value = vmuls(data_x, const_five_two)
        res_overflow = vadds(vlog(overflow_value), log_five_two)
        res = vcmpsel(data_x, float_16_max_tensor, 'ge', res_overflow, res)
        res = cast_compute.cast_to(res, dtype)

        return res

    def _log_compute_block_lt_2_gt_1(data_x, shape):
        # 'pylint: disable=too-many-locals
        # phase1: index_1:data>(5/3)&&data<2
        data = vadds(data_x, tvm.const(const_neg_one, dtype))
        threshold_1 = broadcast(tvm.const(const_log_threshold_1, dtype), shape)
        data_1 = vadds(data,
                       tvm.const(const_neg_one * const_log_threshold_1, dtype))
        data1_vmuls = vmuls(data_1, tvm.const(const_dot_six, dtype))
        data_sel = vcmpsel(data, threshold_1, 'ge', data1_vmuls, data)
        data_sel = cast_compute.cast_to(data_sel, dtype)

        # phase2:index_2:data>(4/3)&&data<(5/3)
        threshold_2 = broadcast(tvm.const(const_log_threshold_2, dtype), shape)
        data_2 = vadds(data_sel,
                       tvm.const(const_neg_one * const_log_threshold_2, dtype))
        data2_vmuls = vmuls(data_2, tvm.const(const_three_four, dtype))
        data_sel = vcmpsel(data_sel, threshold_2, 'ge', data2_vmuls, data_sel)
        data_sel = cast_compute.cast_to(data_sel, dtype)

        # 'phase3:Taylor
        taylor = _taylor_compute(data_sel)

        # phase4:return back to original data
        # add log(4/3)
        res = vcmpsel(data_sel, threshold_2, 'ge',
                      vadds(taylor, tvm.const(log_four_three, dtype)), taylor)
        res = cast_compute.cast_to(res, dtype)
        # add log(5/3)
        res = vcmpsel(data, threshold_1, 'ge',
                      vadds(taylor, tvm.const(log_five_three, dtype)), res)
        res = cast_compute._cast(res, dtype)

        return res

    def _log_compute_block_gt_1(data_x, shape):
        res = _log_compute_block_lt_2_gt_1(data_x, shape)
        res = _log_compute_block_gt_2(data_x, res, shape)

        return res

    def _log_compute_block_gt_half_lt_1(data_x, res, shape):
        threshold_5 = broadcast(tvm.const(const_one, dtype), shape)
        data = vadds(data_x, tvm.const(const_neg_one, dtype))
        taylor = _taylor_compute(data)
        res = vcmpsel(data_x, threshold_5, 'le', taylor, res)
        res = cast_compute.cast_to(res, dtype)

        return res

    def _log_compute_block_lt_half(data_x, res, shape):
        threshold_4 = broadcast(tvm.const(const_half, dtype), shape)
        res = vcmpsel(data_x, threshold_4, 'le',
                      vmuls(_log_compute_block_gt_1(vrec(data_x), shape),
                            const_neg_one), res)
        res = cast_compute.cast_to(res, dtype)

        return res

    res = _log_compute_block_gt_1(data_x, shape)

    res = _log_compute_block_gt_half_lt_1(data_x, res, shape)

    res = _log_compute_block_lt_half(data_x, res, shape)

    return res


@util.dtype_check_decorator
def vlog(raw_tensor, priority_flag=0):
    """
    calculate ln(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : log(raw_tensor)
    """

    if not cceconf.intrinsic_check_support("Intrinsic_vln", "float32") \
            and priority_flag == 1:
        return __vlog_calculate_by_taylor(raw_tensor)

    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_log')


@util.dtype_check_decorator
def vexp(raw_tensor):
    """
    calculate exp(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : exp(raw_tensor)
    """
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_exp')


@util.dtype_check_decorator
def vabs(raw_tensor):
    """
    calculate abs(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : abs(raw_tensor)
    """
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_abs')


@util.dtype_check_decorator
def vrec(raw_tensor):
    """
    calculate vrec(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrec(raw_tensor)
    """
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_rec')


@util.dtype_check_decorator
def vrelu(raw_tensor):
    """
    calculate vrelu(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrelu(raw_tensor)
    """
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_relu')


@util.dtype_check_decorator
def vnot(raw_tensor):
    """
    calculate vnot(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vnot(raw_tensor)
    """
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_not')


def __vsqrt_calculate_by_newton(raw_tensor):
    """
    calculate vsqrt(raw_tensor), use newton iteration to calcuate sqrt

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """

    const_half = 0.5
    sqrt_const_iter = 3

    dtype = raw_tensor.dtype

    if not cceconf.intrinsic_check_support("Intrinsic_vlog", dtype):
        raw_tensor = cast_compute.cast_to(raw_tensor, "float16")
    init_res = vlog(raw_tensor)
    init_res = vmuls(init_res, tvm.const(const_half))
    init_res = vexp(init_res)

    for _ in range(sqrt_const_iter):
        res = vdiv(raw_tensor, init_res)
        res = vadd(res, init_res)
        res = vmuls(res, tvm.const(const_half, dtype))
        init_res = res
    return res


@util.dtype_check_decorator
def vsqrt(raw_tensor, priority_flag=0):
    """
    calculate vsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """
    if not cceconf.intrinsic_check_support("Intrinsic_vsqrt"):
        if priority_flag == 1:
            return __vsqrt_calculate_by_newton(raw_tensor)
        dtype = raw_tensor.dtype
        res = __single_elewise_op(raw_tensor, dtype, 'elewise_single_rsqrt')
        return vrec(res)
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_sqrt')


def __vrsqrt_calculate_by_newton(raw_tensor):
    """
    calculate vrsqrt(raw_tensor), use newton iteration to calcuate sqrt

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """

    const_half = 0.5
    sqrt_const_iter = 3

    dtype = raw_tensor.dtype

    if not cceconf.intrinsic_check_support("Intrinsic_vlog", dtype):
        raw_tensor = cast_compute.cast_to(raw_tensor, "float16")
    init_res = vlog(raw_tensor)
    init_res = vmuls(init_res, tvm.const(const_half))
    init_res = vexp(init_res)

    for _ in range(sqrt_const_iter):
        res = vdiv(raw_tensor, init_res)
        res = vadd(res, init_res)
        res = vmuls(res, tvm.const(const_half, dtype))
        init_res = res
    return vrec(res)


@util.dtype_check_decorator
def vrsqrt(raw_tensor, priority_flag=0):
    """
    calculate vrsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """
    if not cceconf.intrinsic_check_support("Intrinsic_vsqrt") \
            and priority_flag == 1:
        return __vrsqrt_calculate_by_newton(raw_tensor)
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_rsqrt')


@util.dtype_check_decorator
def vadd(lhs, rhs):
    """
    calculate elewise add

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs + rhs
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_add")


@util.dtype_check_decorator
def vsub(lhs, rhs):
    """
    calculate elewise sub

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : lhs - rhs
    """

    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_sub")


@util.dtype_check_decorator
def vmul(lhs, rhs):
    """
    calculate elewise multiply

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    Returns
    -------
    wrapped_tensor : lhs*rhs
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_mul")


@util.dtype_check_decorator
def vdiv(lhs, rhs):
    """
    calculate elewise div

    Parameters
    -----
    lhs: wrapped_tensor or tvm.tensor
         divisor tensor
    rhs: wrapped_tensor or tvm.tensor
         divided tensor

    returns
    -----
    wrapped_tensor: lhs / rhs
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    if not cceconf.intrinsic_check_support("Intrinsic_vdiv"):
        dtype = rhs.dtype
        reciprocal_rhs = __single_elewise_op(rhs, dtype, 'elewise_single_rec')
        vdiv_value = __binary_elewise_op(lhs, reciprocal_rhs,
                                         "elewise_binary_mul")
        return vdiv_value

    return __binary_elewise_op(lhs, rhs, "elewise_binary_div")


@util.dtype_check_decorator
def vmin(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : min(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")
    return __binary_elewise_op(lhs, rhs, "elewise_binary_min")


@util.dtype_check_decorator
def vmax(lhs, rhs):
    """
    calculate elewise compare, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")
    return __binary_elewise_op(lhs, rhs, "elewise_binary_max")


@util.dtype_check_decorator
def vor(lhs, rhs):
    """
    calculate bitwise or op, return the or value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : or(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")
    return __binary_elewise_op(lhs, rhs, "elewise_binary_or")


@util.dtype_check_decorator
def vand(lhs, rhs):
    """
    calculate bitwise and op, return the and value
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")
    return __binary_elewise_op(lhs, rhs, "elewise_binary_and")


@util.dtype_check_decorator
def vaxpy(lhs, rhs, scalar):
    """
    calculate elewise scalar*lhs + rhs, return the min one
    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor
    rhs : wrapped_tensor or tvm.tensor
        left hand tensor
    scalar:
    Returns
    -------
    wrapped_tensor : max(lhs , rhs)
    """
    if isinstance(scalar, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be scalar")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_scalar_axpy",
                               args=[scalar])


@util.dtype_check_decorator
def vmla(tensor_0, tensor_1, tensor_2):
    """
    calculate x*tensor_1 + tensor_2,  only support float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : X*tensor_1 + tensor_2
    """
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be tvm.tensor")
    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_mla")


@util.dtype_check_decorator
def vmadd(tensor_0, tensor_1, tensor_2):
    """
    calculate tensor_0*tensor_2 + tensor_1,  only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : tensor_0*tensor_2 + tensor_1
    """
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be tvm.tensor")
    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_madd")


@util.dtype_check_decorator
def vmaddrelu(tensor_0, tensor_1, tensor_2):
    """
    calculate relu(tensor_0*tensor_2 + tensor_1),
    only support  float16, float32
    Parameters
    ----------
    tensor_0 : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : relu(tensor_0*tensor_2 + tensor_1)
    """
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        raise RuntimeError("The third input type must be tvm.tensor")
    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_maddrelu")


# 'pylint: disable=R0914
def __binary_elewise_op(tensor_l, tensor_r, op_name, args=None):
    _check_elewise_binary_shape(tensor_l, tensor_r)

    if tensor_l.dtype != tensor_r.dtype:
        raise RuntimeError(
            "dtype must be the same while lhType is %s, rhType is %s" %
            (tensor_l.dtype, tensor_r.dtype))

    fun_map1 = {
        "elewise_binary_add": lambda *i: tensor_l(*i) + tensor_r(*i),
        "elewise_binary_sub": lambda *i: tensor_l(*i) - tensor_r(*i),
        "elewise_binary_div": lambda *i: tensor_l(*i) / tensor_r(*i),
        "elewise_binary_mul": lambda *i: tensor_l(*i) * tensor_r(*i),
        "elewise_binary_min": lambda *i: tvm.min(tensor_l(*i), tensor_r(*i)),
        "elewise_binary_max": lambda *i: tvm.max(tensor_l(*i), tensor_r(*i)),
        "elewise_binary_and": lambda *i: tensor_l(*i) & tensor_r(*i),
        "elewise_binary_or": lambda *i: tensor_l(*i) | tensor_r(*i),
    }
    if op_name in fun_map1:
        lambda_func = fun_map1[op_name]
    elif op_name == "emit_insn_elewise_binary_cmp":
        fun_map2 = {
            "lt": lambda *i: tensor_l(*i) < tensor_r(*i),
            "gt": lambda *i: tensor_l(*i) > tensor_r(*i),
            "le": lambda *i: tensor_l(*i) <= tensor_r(*i),
            "ge": lambda *i: tensor_l(*i) >= tensor_r(*i),
            "eq": lambda *i: tvm.expr.EQ(tensor_l(*i), tensor_r(*i)),
            "ne": lambda *i: tvm.expr.NE(tensor_l(*i), tensor_r(*i)),
        }
        operation = args[0]
        if operation not in fun_map2:
            raise RuntimeError("vcmp do not support the input op_name: %s" %
                               operation)
        lambda_func = fun_map2[operation]
    else:
        raise RuntimeError("operation %s not support yet" % op_name)

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1
    shape = tensor_l.shape
    if op_name == "emit_insn_elewise_binary_cmp" and args[1] == 'bit':
        shape = util.shape_to_list(shape)
        if shape[-1] % 8 != 0:
            raise RuntimeError(
                "The input shape's last axis must be multiple of 8")

        k = tvm.reduce_axis((0, 8), name='k')
        res_shape = shape
        res_shape[-1] = res_shape[-1] // 8

        def _compute(*index):
            res_index = []
            for i, value in enumerate(index):
                if i == len(index) - 1:
                    res_index.append(value * 8 + k)
                else:
                    res_index.append(value)
            tensor = tvm.bit(lambda_func(*res_index).astype('uint8'), axis=k)
            return tensor

        op_name = op_name + "|" + args[0] + "|" + args[1]

        with tvm.tag_scope(op_name):
            output = tvm.compute(res_shape, _compute, name='output')
        return output

    if op_name in ("emit_insn_elewise_binary_cmp",):
        for arg in args:
            op_name = op_name + "|" + arg

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


# 'pylint: disable=R0912
def __single_elewise_op(input_tensor, dtype, op_name, args=None):
    """
    factory method of single elewise operations
    """
    fun_map = {
        "elewise_single_log": lambda *i: tvm.log(input_tensor(*i)),
        "elewise_single_exp": lambda *i: tvm.exp(input_tensor(*i)),
        "elewise_single_rec": lambda *i: 1 / input_tensor(*i),
        "elewise_single_VS_add": lambda *i: input_tensor(*i) + util.astype(args[0], dtype),
        "elewise_single_VS_mul": lambda *i: input_tensor(*i) * util.astype(args[0], dtype),
        "elewise_single_VS_max": lambda *i: tvm.max(input_tensor(*i),
                                                    util.astype(args[0], dtype)),
        "elewise_single_VS_min": lambda *i: tvm.min(input_tensor(*i),
                                                    util.astype(args[0], dtype)),
        "elewise_single_abs": lambda *i: tvm.select(input_tensor(*i) >= 0,
                                                    input_tensor(*i),
                                                    - input_tensor(*i)),
        "elewise_single_relu": lambda *i: tvm.select(input_tensor(*i) >= 0,
                                                     input_tensor(*i),
                                                     tvm.const(0, dtype=dtype)),
        "elewise_single_not": lambda *i: ~input_tensor(*i),
        "elewise_single_sqrt": lambda *i: tvm.sqrt(input_tensor(*i)),
        "elewise_single_rsqrt": lambda *i: tvm.rsqrt(input_tensor(*i)),
    }
    if op_name in fun_map:
        lambda_func = fun_map[op_name]
    else:
        raise RuntimeError("operation %s not support yet" % op_name)

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1
    shape = input_tensor.shape

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    if op_name == "elewise_single_rec":
        def __get_newton_iter_num():
            _newton_iter_num = 2
            if product_params().is_mini_version():
                _newton_iter_num = 1
            return _newton_iter_num

        newton_iter_num = __get_newton_iter_num()
        name_pre = op_name.split("_")[-1] + "_"
        const_num_neg_one = tvm.const(-1, dtype=dtype)
        const_num_two = tvm.const(2, dtype=dtype)

        # newton iteration formula is x(n) = x(n-1)(2 - ax(n-1))
        for _ in range(newton_iter_num):
            # the name of each compute
            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            with tvm.tag_scope("elewise_binary_mul"):
                tmp_mul = tvm.compute(
                    shape,
                    lambda *indice: input_tensor(*indice) * tmp(*indice),
                    name=name)

            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # 'pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_single_VS_mul"):
                tmp_negative = tvm.compute(
                    shape,
                    lambda *indice: tmp_mul(*indice) * const_num_neg_one,
                    name=name)

            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # 'pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_single_VS_add"):
                tmp_plus = tvm.compute(
                    shape,
                    lambda *indice: tmp_negative(*indice) + const_num_two,
                    name=name)
            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # 'pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_binary_mul"):
                tmp = tvm.compute(
                    shape,
                    lambda *indice: tmp_plus(*indice) * tmp(*indice),
                    name=name)

    return tmp


def __multiple_elewise_op(tensor_0, tensor_1, tensor_2, op_name):
    """
    factory method of binary multiple operations
    """
    intrin = "v" + op_name.split("_")[-1]
    is_support_dtype = cceconf.intrinsic_check_support("Intrinsic_" + intrin,
                                                       tensor_0.dtype)

    _check_multiple_elewise_op_shape(tensor_0, tensor_1, tensor_2)
    dtype_cond1 = tensor_0.dtype != tensor_1.dtype or \
        tensor_0.dtype != tensor_2.dtype or \
        tensor_2.dtype != tensor_1.dtype
    if dtype_cond1:
        dtype_cond2 = op_name != "elewise_multiple_mla" or \
            tensor_0.dtype != tensor_1.dtype or \
            tensor_0.dtype != "float16" or \
            tensor_2.dtype != "float32"
        if dtype_cond2:
            raise RuntimeError(
                "dtype error, vmla not support mixed data type auto cast")
    elif not is_support_dtype:
        raise RuntimeError(
            "dtype error, vmla not support mixed data type auto cast")

    shape = tensor_0.shape
    if op_name == "elewise_multiple_mla":
        type2 = tensor_2.dtype
        lambda_func = lambda *i: \
            tvm.expr.Cast(type2, tensor_0(*i) * tensor_1(*i)) + tensor_2(*i)
    elif op_name == "elewise_multiple_madd":
        lambda_func = lambda *i: tensor_0(*i) * tensor_2(*i) + tensor_1(*i)
    elif op_name == "elewise_multiple_maddrelu":
        lambda_func = lambda *i: tvm.relu(tensor_0(*i) * tensor_2(*i) +
                                          tensor_1(*i))
    else:
        raise RuntimeError("operation %s not support yet" % op_name)

    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    # list cases of same input, "1"s in same_list means same input
    if tensor_0 == tensor_1 and tensor_0 == tensor_2:
        same_list = [1, 1, 1]
    elif tensor_0 == tensor_1:
        same_list = [1, 1, 0]
    elif tensor_0 == tensor_2:
        same_list = [1, 0, 1]
    elif tensor_1 == tensor_2:
        same_list = [0, 1, 1]
    else:
        same_list = [0, 0, 0]

    str_same_list = ",".join([str(i) for i in same_list])
    with tvm.tag_scope(op_name + '|' + str_same_list):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


def _check_elewise_binary_shape(lhs, rhs):
    if len(lhs.shape) != len(rhs.shape):
        raise RuntimeError(
            "The lhs shape ndim %d must be equal to the rhs %d" %
            (len(lhs.shape), len(rhs.shape)))

    for _l, _r in zip(lhs.shape, rhs.shape):
        if not util.equal(_l, _r):
            raise RuntimeError("The lhs shape must be equal to the rhs")


def _check_multiple_elewise_op_shape(tensor_0, tensor_1, tensor_2):
    """
    check multiple elewise op's shape
    :param tensor_0:
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    if len(tensor_0.shape) != len(tensor_1.shape) \
            or len(tensor_0.shape) != len(tensor_2.shape) \
            or len(tensor_2.shape) != len(tensor_1.shape):
        raise RuntimeError(
            "The input shape ndim must be equal to the each other")

    for _a, _b, _c in zip(tensor_0.shape, tensor_1.shape, tensor_2.shape):
        if not (util.equal(_a, _b) and util.equal(_b, _c)):
            raise RuntimeError(
                "The input shape must be equal to the each other")


def vcmpsel_data_shape_check(*args):
    """
    check vcmpsel's data shape
    :param args:
    :return:
    """
    arg_temp = args[0]

    for sh_value in arg_temp.shape:
        if not isinstance(sh_value, tvm.expr.Expr):
            if (sh_value.value == 0 or sh_value.value < -1) \
                    or not isinstance(sh_value.value, int):
                raise RuntimeError("The input shape value \
                                   must be a positive integer or -1!")

    for arg in args:
        if len(arg.shape) != len(arg_temp.shape):
            raise RuntimeError("The input shape ndim \
                               must be equal to the each other!")

    for i in range(len(arg_temp.shape)):
        for arg in args:
            if not isinstance(arg_temp.shape[i], tvm.expr.Expr) \
                    and not isinstance(arg.shape[i], tvm.expr.Expr):
                if arg_temp.shape[i].value != arg.shape[i].value:
                    raise RuntimeError("The input shape must be "
                                       "equal to the each other!")


def vcmpsel_data_dtype_check(*args):
    """
    check vcmpsel's data type
    :param args:
    :return:
    """
    arg_temp = args[0]

    for arg in args:
        if arg.dtype != arg_temp.dtype:
            raise RuntimeError("The input dtype \
                               must be the same to the each other!")


# 'pylint: disable=too-many-branches, too-many-statements
@util.dtype_check_decorator
def vcmpsel(lhs, rhs=None, operation='lt', slhs=None, srhs=None):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        compare left hand tensor
    rhs : wrapped_tensor or tvm.tensor or scalar
        compare right hand tensor or scalar
    operation : operator type, eq, ne, lt, gt, ge, le
    slhs : wrapped_tensor or tvm.tensor or scalar
        select left hand tensor or scalar
    srhs : wrapped_tensor or tvm.tensor or scalar
        select right hand tensor or scalar

    Returns
    -------
    wrapped_tensor
    """
    def check_input(lhs, operation):
        if not isinstance(lhs, tvm.tensor.Tensor):
            raise RuntimeError("The first input type must be tvm.tensor!")

        if operation not in ['eq', 'ne', 'lt', 'gt', 'ge', 'le']:
            raise RuntimeError("The op's value must be "
                               "eq, ne, lt, gt, ge, le!")

    check_input(lhs, operation)

    if rhs is None:
        rhs = 2.0

    if slhs is None:
        slhs = lhs

    if srhs is None:
        if isinstance(rhs, tvm.tensor.Tensor):
            srhs = rhs
        else:
            srhs = 0.0

    shape = lhs.shape
    lhs_dtype = lhs.dtype
    util.dsl_check_support("vcmpsel", lhs_dtype)

    def _get_cmpvs_lambda_func(operation, lhs, rhs):
        if operation == 'lt':
            lambda_func = lambda *indice: lhs(*indice) < rhs
        elif operation == 'gt':
            lambda_func = lambda *indice: lhs(*indice) > rhs
        elif operation == 'le':
            lambda_func = lambda *indice: lhs(*indice) <= rhs
        elif operation == 'ge':
            lambda_func = lambda *indice: lhs(*indice) >= rhs
        elif operation == 'eq':
            lambda_func = lambda *indice: tvm.expr.EQ(lhs(*indice), rhs)
        elif operation == 'ne':
            lambda_func = lambda *indice: tvm.expr.NE(lhs(*indice), rhs)
        else:
            raise RuntimeError("vcmp do not support the input op")
        return lambda_func

    def _get_cmpv_lambda_func(operation, lhs, rhs):
        if operation == 'lt':
            lambda_func = lambda *indice: lhs(*indice) < rhs(*indice)
        elif operation == 'gt':
            lambda_func = lambda *indice: lhs(*indice) > rhs(*indice)
        elif operation == 'le':
            lambda_func = lambda *indice: lhs(*indice) <= rhs(*indice)
        elif operation == 'ge':
            lambda_func = lambda *indice: lhs(*indice) >= rhs(*indice)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.expr.EQ(lhs(*indice), rhs(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.expr.NE(lhs(*indice), rhs(*indice))
        else:
            raise RuntimeError("vcmp do not support the input op")
        return lambda_func

    cmp_op = "elewise_binary_vcmpv_" + operation
    sel_op = "elewise_multiple_sel"
    cmpsel_op = "elewise_binary_cmpsel_" + operation

    input_type_str = util.get_vcmpsel_input_type(rhs, slhs, srhs)

    if input_type_str == "SCALAR_SCALAR_SCALAR":
        # lhs tensor, rhs, slhs, srhs scalar
        rhs = util.get_tvm_scalar(rhs, lhs_dtype)
        slhs = util.get_tvm_scalar(slhs, lhs_dtype)
        srhs = util.get_tvm_scalar(srhs, lhs_dtype)

        def _vcmpsel_tsss_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpvs_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs, srhs)
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_tsss_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs, slhs, srhs)
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs, slhs, srhs)
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs, slhs, srhs)
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs, slhs, srhs)
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs, slhs, srhs)
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs, slhs, srhs)
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_tsss_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

            return tmp
        return _vcmpsel_tsss_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)

    elif input_type_str == "TENSOR_SCALAR_SCALAR":
        vcmpsel_data_shape_check(lhs, rhs)
        vcmpsel_data_dtype_check(lhs, rhs)
        slhs = util.get_tvm_scalar(slhs, lhs_dtype)
        srhs = util.get_tvm_scalar(srhs, lhs_dtype)

        def _vcmpsel_ttss_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpv_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs, srhs)
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_ttss_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs(*indice), slhs, srhs)
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs(*indice), slhs, srhs)
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs(*indice), slhs, srhs)
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs(*indice), slhs, srhs)
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs(*indice), slhs, srhs)
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs(*indice), slhs, srhs)
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_ttss_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

            return tmp

        return _vcmpsel_ttss_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)


    elif input_type_str == "SCALAR_TENSOR_SCALAR":
        vcmpsel_data_shape_check(lhs, slhs)
        vcmpsel_data_dtype_check(lhs, slhs)
        rhs = util.get_tvm_scalar(rhs, lhs.dtype)
        srhs = util.get_tvm_scalar(srhs, lhs.dtype)

        def _vcmpsel_tsts_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpvs_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs(*indice), srhs)
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_tsts_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs, slhs(*indice), srhs)
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs, slhs(*indice), srhs)
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs, slhs(*indice), srhs)
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs, slhs(*indice), srhs)
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs, slhs(*indice), srhs)
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs, slhs(*indice), srhs)
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_tsts_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

            return tmp
        return _vcmpsel_tsts_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)


    elif input_type_str == "SCALAR_SCALAR_TENSOR":
        vcmpsel_data_shape_check(lhs, srhs)
        vcmpsel_data_dtype_check(lhs, srhs)
        rhs = util.get_tvm_scalar(rhs, lhs.dtype)
        slhs = util.get_tvm_scalar(slhs, lhs.dtype)

        def _vcmpsel_tsst_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpvs_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs, srhs(*indice))
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_tsst_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs, slhs, srhs(*indice))
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs, slhs, srhs(*indice))
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs, slhs, srhs(*indice))
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs, slhs, srhs(*indice))
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs, slhs, srhs(*indice))
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs, slhs, srhs(*indice))
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_tsst_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

        return _vcmpsel_tsst_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)

    elif input_type_str == "TENSOR_TENSOR_SCALAR":
        vcmpsel_data_shape_check(lhs, rhs, slhs)
        vcmpsel_data_dtype_check(lhs, rhs, slhs)
        srhs = util.get_tvm_scalar(srhs, lhs.dtype)

        def _vcmpsel_ttts_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpv_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs(*indice), srhs)
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_ttts_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs(*indice),
                                slhs(*indice), srhs)
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs(*indice),
                                slhs(*indice), srhs)
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs(*indice),
                                slhs(*indice), srhs)
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs(*indice),
                                slhs(*indice), srhs)
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs(*indice),
                                slhs(*indice), srhs)
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs(*indice),
                                slhs(*indice), srhs)
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_ttts_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

            return tmp
        return _vcmpsel_ttts_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)


    elif input_type_str == "TENSOR_SCALAR_TENSOR":
        vcmpsel_data_shape_check(lhs, rhs, srhs)
        vcmpsel_data_dtype_check(lhs, rhs, srhs)
        slhs = util.get_tvm_scalar(slhs, lhs.dtype)

        def _vcmpsel_ttst_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpv_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs, srhs(*indice))
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_ttst_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs(*indice),
                                slhs, srhs(*indice))
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs(*indice),
                                slhs, srhs(*indice))
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs(*indice),
                                slhs, srhs(*indice))
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs(*indice),
                                slhs, srhs(*indice))
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs(*indice),
                                slhs, srhs(*indice))
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs(*indice),
                                slhs, srhs(*indice))
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_ttst_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

            return tmp
        return _vcmpsel_ttst_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)


    elif input_type_str == "SCALAR_TENSOR_TENSOR":
        vcmpsel_data_shape_check(lhs, slhs, srhs)
        vcmpsel_data_dtype_check(lhs, slhs, srhs)
        rhs = util.get_tvm_scalar(rhs, lhs.dtype)

        def _vcmpsel_tstt_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpvs_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs(*indice), srhs(*indice))
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_tstt_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs,
                                slhs(*indice), srhs(*indice))
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs,
                                slhs(*indice), srhs(*indice))
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs,
                                slhs(*indice), srhs(*indice))
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs,
                                slhs(*indice), srhs(*indice))
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs,
                                slhs(*indice), srhs(*indice))
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs,
                                slhs(*indice), srhs(*indice))
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_tstt_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

            return tmp
        return _vcmpsel_tstt_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)


    else:
        vcmpsel_data_shape_check(lhs, rhs, slhs, srhs)
        vcmpsel_data_dtype_check(lhs, rhs, slhs, srhs)

        def _vcmpsel_tttt_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16" and not product_params().is_ng1_version():
                lambda_func_cmp = _get_cmpv_lambda_func(operation, lhs, rhs)
                name = "vcmpv_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(cmp_op):
                    condition = tvm.compute(shape, lambda_func_cmp, name=name)

                lambda_func_sel = lambda *indice: \
                    tvm.select(condition(*indice), slhs(*indice), srhs(*indice))
                name = "vsel_" + str(NAME_INDEX[0])
                NAME_INDEX[0] += 1
                with tvm.tag_scope(sel_op):
                    tmp = tvm.compute(shape, lambda_func_sel, name=name)

                return tmp

            def _get_cmpsel_tttt_lambda_func(operation, lhs, rhs, slhs, srhs):
                if operation == 'lt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) < rhs(*indice),
                                slhs(*indice), srhs(*indice))
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs(*indice),
                                slhs(*indice), srhs(*indice))
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs(*indice),
                                slhs(*indice), srhs(*indice))
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs(*indice),
                                slhs(*indice), srhs(*indice))
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs(*indice),
                                slhs(*indice), srhs(*indice))
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs(*indice),
                                slhs(*indice), srhs(*indice))
                else:
                    raise RuntimeError("vcmpsel do not support the input op")
                return lambda_func

            lambda_func = _get_cmpsel_tttt_lambda_func(operation,
                                                    lhs, rhs, slhs, srhs)

            name = cmpsel_op.split("_")[-1] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)
            return tmp
        return _vcmpsel_tttt_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)


def __vmod_cloud(lhs, rhs):
    from .cast_compute import floor
    # cloud and v200
    dtype = lhs.dtype
    lhs = cast_compute._cast(lhs, "float32")
    rhs = cast_compute._cast(rhs, "float32")
    res_div = vdiv(lhs, rhs)
    res_floor = floor(res_div)
    res_floor = cast_compute._cast(res_floor, "float32")
    res_mul = vmul(rhs, res_floor)
    res = vsub(lhs, res_mul)

    return cast_compute._cast(res, dtype)


# 'pylint: disable=too-many-locals
def __vmod_mini(lhs, rhs):
    from .cast_compute import floor
    dtype = rhs.dtype
    rhs_f16 = rhs
    # 1. calculate result for testing, using float32 for better precision
    lhs = cast_compute._cast(lhs, "float32")
    rhs = cast_compute._cast(rhs, "float32")
    test_div = vmul(lhs, vrec(rhs))
    test_div = cast_compute._cast(test_div, "float16")
    test_floor = cast_compute._cast(floor(test_div), "float32")
    test_res = vsub(lhs, vmul(rhs, test_floor))

    # 2. correct the floor result, using float16
    test_res = cast_compute._cast(test_res, dtype)
    test_floor = cast_compute._cast(test_floor, dtype)
    zero = broadcast(0.0, lhs.shape, dtype)

    # rhs positive: 0 <= res < rhs
    prhs_floor = vcmpsel(test_res, zero, 'lt',
                         vadds(test_floor, -1.0), test_floor)

    # rhs negative: rhs < res <= 0
    nrhs_floor = vcmpsel(test_res, zero, 'gt',
                         vadds(test_floor, -1.0), test_floor)

    # according to positive and negative rhs to choose p_floor or n_floor
    result_floor = vcmpsel(rhs_f16, zero, 'gt', prhs_floor, nrhs_floor)

    # 3. calculate the final result, using float32 for better precision
    result_floor = cast_compute._cast(result_floor, "float32")
    result = vsub(lhs, vmul(rhs, result_floor))

    return cast_compute._cast(result, dtype)


@util.dtype_check_decorator
def vmod(lhs, rhs):
    """
    calculate element-wise remainder of division

    Parameters
    -----
    lhs : wrapped_tensor or tvm.tensor
          left hand tensor

    rhs : wrapped_tensor or tvm.tensor
          right hand tensor

    Returns
    -----
    wrapped_tensor : lhs - floor(lhs/rhs) * rhs
    """
    if not isinstance(lhs, tvm.tensor.Tensor):
        raise RuntimeError("The first input type must be tvm.tensor")
    if not isinstance(rhs, tvm.tensor.Tensor):
        raise RuntimeError("The second input type must be tvm.tensor")

    _check_elewise_binary_shape(lhs, rhs)
    if lhs.dtype != rhs.dtype:
        raise RuntimeError("dtype must be the same while lhType is %s, "
                           "rhType is %s" % (lhs.dtype, rhs.dtype))

    # cloud using vdiv. mini using vrec for division calculation,
    # and mini should improve vmod calculation accuracy.
    if (not cceconf.intrinsic_check_support("Intrinsic_vdiv")) and \
            (not cceconf.intrinsic_check_support("Intrinsic_vconv",
                                                 "f322s32f")):
        if lhs.dtype not in ("float16", ):
            raise RuntimeError("dtype must be float16.")
        res = __vmod_mini(lhs, rhs)
    else:
        if lhs.dtype not in ("float16", "float32"):
            raise RuntimeError("dtype must be float16 or float32.")
        res = __vmod_cloud(lhs, rhs)

    return res
