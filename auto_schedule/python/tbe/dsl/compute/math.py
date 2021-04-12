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
math
"""
import tbe.dsl
from decorator import decorator
from tbe import tvm
from tbe.common.platform import ASCEND_310
from tbe.common.platform import SOC_VERSION
from tbe.common.platform import intrinsic_check_support
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.testing.dsl_source_info import source_info_decorator
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation as operation_context
from tbe.dsl.base.expr_compare import expr_equal as equal

from .cast import _cast
from .util import auto_cast_tensor
from .util import dsl_check_support
from .util import dtype_check_decorator
from .util import get_tvm_scalar
from .util import in_dynamic_and_static_unify
from .util import is_cast_support
from .util import judge_var
from .util import shape_to_list
from .util import util_astype

NAME_INDEX = [0]


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
@decorator
def _auto_cast_of_elewise(func, *args, **kwargs):
    """
    auto cast dectorator.
    Before calling elewise api, check the input tensor is supported by the intr.
    If not supported, casting the input tensor to supported dtype.
    (On condition that the cast type is supported.
    If the cast type is not supported,raising a RuntimeError).
    """
    # dynamic not support auto_cast
    if in_dynamic_and_static_unify():
        return func(*args, **kwargs)

    def _check_args_type(args):
        if len(args) in (1, 2, 3):
            if not isinstance(args[0], tvm.tensor.Tensor):
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The first input type must be [%s]" \
                                              ", while type is [%s]" \
                                              % ('tvm.tensor', type(args[0]))
                raise RuntimeError(dict_args, get_error_message(dict_args))
            if len(args) == 3:
                if not isinstance(args[1], tvm.tensor.Tensor):
                    dict_args = dict()
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                                  "while type is [%s]" \
                                                  % ('tvm.tensor', type(args[0]))
                    raise RuntimeError(dict_args, get_error_message(dict_args))

    _check_args_type(args)

    intr = func.__name__
    intr = _intrinsic_check(intr)

    is_support_fp32 = intrinsic_check_support("Intrinsic_"+intr, "float32")
    if len(args) == 1:
        def _cast_one_input_tensor(args, intr, is_support_fp32):
            temp_tensor = args[0]
            dtype = temp_tensor.dtype
            is_support_dtype = intrinsic_check_support("Intrinsic_"+intr, dtype)
            if not is_support_dtype:
                if is_support_fp32 and is_cast_support(dtype, "float32"):
                    temp_tensor = _cast(temp_tensor, "float32")
                else:
                    temp_tensor = _cast(temp_tensor, "float16")

            return temp_tensor

        temp_tensor = _cast_one_input_tensor(args, intr, is_support_fp32)
        return func(temp_tensor)
    if len(args) == 2:
        if isinstance(args[1], tvm.tensor.Tensor):
            def _cast_two_input_tensor(args, intr, is_support_fp32):
                    lhs = args[0]
                    rhs = args[1]
                    dtype_l = lhs.dtype
                    dtype_r = rhs.dtype

                    lhs_t = lhs
                    rhs_t = rhs
                    is_support_ldtype = intrinsic_check_support("Intrinsic_"+intr,
                                                                dtype_l)
                    is_support_rdtype = intrinsic_check_support("Intrinsic_"+intr,
                                                                dtype_r)
                    if not is_support_ldtype \
                            or not is_support_rdtype or dtype_l != dtype_r:
                        if is_support_fp32 \
                                and is_cast_support(dtype_l, "float32") \
                                and is_cast_support(dtype_r, "float32"):
                            lhs_t = _cast(lhs, "float32")
                            rhs_t = _cast(rhs, "float32")
                        else:
                            lhs_t = _cast(lhs, "float16")
                            rhs_t = _cast(rhs, "float16")

                    return lhs_t, rhs_t

            lhs_t, rhs_t = _cast_two_input_tensor(args, intr, is_support_fp32)
            return func(lhs_t, rhs_t)

        def _cast_tensor_scalar_two_input(args, intr, is_support_fp32):
            temp_tensor = args[0]
            scalar = args[1]
            dtype = temp_tensor.dtype
            is_support_dtype = intrinsic_check_support("Intrinsic_"+intr, dtype)
            if not is_support_dtype:
                if is_support_fp32 \
                        and is_cast_support(dtype, "float32"):
                    temp_tensor = _cast(temp_tensor, "float32")
                    dtype = "float32"
                else:
                    temp_tensor = _cast(temp_tensor, "float16")
                    dtype = "float16"

            tmp_arg = scalar
            scalar_type = judge_var(scalar)
            if scalar_type == "tvm_const" and scalar.dtype != dtype:
                tmp_arg = tvm.const(scalar.value, dtype=dtype)

            if scalar_type == "python_const":
                tmp_arg = tvm.const(scalar, dtype=dtype)

            return temp_tensor, tmp_arg

        temp_tensor, tmp_arg = _cast_tensor_scalar_two_input(args, intr, is_support_fp32)
        return func(temp_tensor, tmp_arg)
    if len(args) == 3:
        if isinstance(args[2], tvm.tensor.Tensor):
            def _cast_three_input_tensor(args, intr, is_support_fp32):
                tensor_0 = args[0]
                tensor_1 = args[1]
                tensor_2 = args[2]

                dtype_0 = tensor_0.dtype
                dtype_1 = tensor_1.dtype
                dtype_2 = tensor_2.dtype

                tensor_0_t = tensor_0
                tensor_1_t = tensor_1
                tensor_2_t = tensor_2

                if dtype_0 != dtype_1 or dtype_0 != dtype_2 or dtype_2 != dtype_1:
                    dict_args = dict()
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "Input tensors must has same dtype! " \
                                                  "while dtype_0 is [%s], " \
                                                  "dtype_1 is [%s], " \
                                                  "dtype_2 is [%s]" \
                                                  % (dtype_0, dtype_1, dtype_2)
                    raise RuntimeError(dict_args, get_error_message(dict_args))

                is_support_dtype0 = intrinsic_check_support("Intrinsic_"+intr,
                                                            dtype_0)
                if not is_support_dtype0:
                    if is_support_fp32 \
                            and is_cast_support(dtype_0, "float32"):
                        tensor_0_t = _cast(tensor_0, "float32")
                        tensor_1_t = _cast(tensor_1, "float32")
                        tensor_2_t = _cast(tensor_2, "float32")
                    else:
                        tensor_0_t = _cast(tensor_0, "float16")
                        tensor_1_t = _cast(tensor_1, "float16")
                        tensor_2_t = _cast(tensor_2, "float16")

                return tensor_0_t, tensor_1_t, tensor_2_t

            tensor_0_t, tensor_1_t, tensor_2_t = \
                _cast_three_input_tensor(args, intr, is_support_fp32)
            return func(tensor_0_t, tensor_1_t, tensor_2_t)

        def _cast_tensors_scalar_in_three_input(args, intr, is_support_fp32):
            lhs = args[0]
            rhs = args[1]
            scalar = args[2]
            dtype_l = lhs.dtype
            dtype_r = rhs.dtype

            lhs_t = lhs
            rhs_t = rhs
            is_support_ldtype = intrinsic_check_support("Intrinsic_"+intr, dtype_l)
            is_support_rdtype = intrinsic_check_support("Intrinsic_"+intr, dtype_r)
            if not is_support_ldtype \
                    or not is_support_rdtype or dtype_l != dtype_r:
                if is_support_fp32 \
                        and is_cast_support(dtype_l, "float32") \
                        and is_cast_support(dtype_r, "float32"):
                    lhs_t = _cast(lhs, "float32")
                    rhs_t = _cast(rhs, "float32")
                    dtype_l = "float32"
                else:
                    lhs_t = _cast(lhs, "float16")
                    rhs_t = _cast(rhs, "float16")
                    dtype_l = "float16"

            tmp_arg = scalar
            if not isinstance(tmp_arg, str):
                scalar_type = judge_var(scalar)
                if scalar_type == "tvm_const" and scalar.dtype != dtype_l:
                    tmp_arg = tvm.const(scalar.value, dtype=dtype_l)

                if scalar_type == "python_const":
                    tmp_arg = tvm.const(scalar, dtype=dtype_l)

            return lhs_t, rhs_t, tmp_arg

        lhs_t, rhs_t, tmp_arg = \
            _cast_tensors_scalar_in_three_input(args, intr, is_support_fp32)
        return func(lhs_t, rhs_t, tmp_arg)
    return func(*args, **kwargs)


def _cast_tensors_for_instr(instr, input_tensors):

    def _process_scalar():
        """
        process when second input is not a tensor
        """
        temp_tensor = input_tensors[0]
        scalar = input_tensors[1]
        dtype = temp_tensor.dtype
        is_support_dtype = intrinsic_check_support("Intrinsic_"+instr, dtype)
        if not is_support_dtype:
            if is_support_fp32 \
                    and is_cast_support(dtype, "float32"):
                temp_tensor = _cast(temp_tensor, "float32")
                dtype = "float32"
            else:
                temp_tensor = _cast(temp_tensor, "float16")
                dtype = "float16"

        tmp_arg = scalar
        scalar_type = judge_var(scalar)
        if scalar_type == "tvm_const" and scalar.dtype != dtype:
            tmp_arg = tvm.const(scalar.value, dtype=dtype)

        if scalar_type == "python_const":
            tmp_arg = tvm.const(scalar, dtype=dtype)
        return [temp_tensor, tmp_arg]

    instr = _intrinsic_check(instr)
    is_support_fp32 = intrinsic_check_support("Intrinsic_"+instr, "float32")

    if len(input_tensors) == 1:
        input_tensor = input_tensors[0]
        if not intrinsic_check_support("Intrinsic_"+instr, input_tensor.dtype):
            if is_support_fp32:
                input_tensor_new = _cast(input_tensor, "float32")
            else:
                input_tensor_new = _cast(input_tensor, "float16")

            return [input_tensor_new, ]

    if len(input_tensors) == 2:
        if isinstance(input_tensors[1], tvm.tensor.Tensor):
            lhs = input_tensors[0]
            rhs = input_tensors[1]
            dtype_l = lhs.dtype
            dtype_r = rhs.dtype

            lhs_t = lhs
            rhs_t = rhs
            is_support_ldtype = intrinsic_check_support("Intrinsic_"+instr,
                                                        dtype_l)
            is_support_rdtype = intrinsic_check_support("Intrinsic_"+instr,
                                                        dtype_r)
            if not is_support_ldtype \
                    or not is_support_rdtype or dtype_l != dtype_r:
                if is_support_fp32 \
                        and is_cast_support(dtype_l, "float32") \
                        and is_cast_support(dtype_r, "float32"):
                    lhs_t = _cast(lhs, "float32")
                    rhs_t = _cast(rhs, "float32")
                else:
                    lhs_t = _cast(lhs, "float16")
                    rhs_t = _cast(rhs, "float16")

            return [lhs_t, rhs_t]
        return _process_scalar()

    return input_tensors


def _intrinsic_check(intr):
    ret_intr = intr
    if not intrinsic_check_support("Intrinsic_" + intr):
        if intr == "vdiv":
            ret_intr = "vrec"
        elif intr == "vsqrt":
            ret_intr = "vrsqrt"
        elif intr == "vlog":
            ret_intr = "vln"
        elif intr == "vmaxs":
            ret_intr = "vmax"
        elif intr == "vmins":
            ret_intr = "vmin"

    return ret_intr


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % ('scalar', type(scalar))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_mul',
                               args=[scalar])


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % (
                                      'scalar', type(scalar))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_add',
                               args=[scalar])


@source_info_decorator()
@dtype_check_decorator
def vmaxs(raw_tensor, scalar):
    """
    Calculate elewise compare, return the max one of scalar or tensor's element,
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % (
                                          'scalar', type(scalar))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_max',
                               args=[scalar])


@source_info_decorator()
@dtype_check_decorator
def vmins(raw_tensor, scalar):
    """
    Calculate elewise compare, return the min one of scalar or tensor's element,
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % (
                                          'scalar', type(scalar))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_VS_min',
                               args=[scalar])


def __vlog_calculate_by_taylor(data_x):
    """
    calculate ln(raw_tensor), use taylor expansion to calculate log
    """
    # pylint: disable=too-many-locals, too-many-statements
    from .cast import cast_to
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
        # pylint: disable=too-many-locals
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
        taylor_three_1 = vadds(taylor_four_2, tvm.const(const_one_three, dtype))
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
        # pylint: disable=too-many-locals
        # if data > 2, use vlog
        threshold_3 = tbe.dsl.broadcast(tvm.const(const_two, dtype), shape)
        if in_dynamic_and_static_unify():
            res = vcmpsel(data_x, threshold_3, 'ge', vlog(data_x), res)
        else:
            index_3 = vcmp(data_x, threshold_3, 'ge')
            res = vsel(index_3, vlog(data_x), res)
        # if data > 32768, use log(x/2.5)+log(2.5)
        float_16_max_tensor = tbe.dsl.broadcast(tvm.const(float_16_max, dtype), shape)
        overflow_value = vmuls(data_x, const_five_two)
        res_overflow = vadds(vlog(overflow_value), log_five_two)
        if in_dynamic_and_static_unify():
            res = vcmpsel(data_x, float_16_max_tensor, 'ge', res_overflow, res)
        else:
            index_4 = vcmp(data_x, float_16_max_tensor, 'ge')
            res = vsel(index_4, res_overflow, res)
        res = cast_to(res, dtype)

        return res

    def _log_compute_block_lt_2_gt_1(data_x, shape):
        # pylint: disable=too-many-locals
        # phase1: index_1:data>(5/3)&&data<2
        data = vadds(data_x, tvm.const(const_neg_one, dtype))
        threshold_1 = tbe.dsl.broadcast(tvm.const(const_log_threshold_1, dtype), shape)
        data_1 = vadds(data,
                       tvm.const(const_neg_one * const_log_threshold_1, dtype))
        data1_vmuls = vmuls(data_1, tvm.const(const_dot_six, dtype))
        if in_dynamic_and_static_unify():
            data_sel = vcmpsel(data, threshold_1, 'ge', data1_vmuls, data)
        else:
            index_1 = vcmp(data, threshold_1, 'ge')
            data_sel = vsel(index_1, data1_vmuls, data)
        data_sel = cast_to(data_sel, dtype)

        # phase2:index_2:data>(4/3)&&data<(5/3)
        threshold_2 = tbe.dsl.broadcast(tvm.const(const_log_threshold_2, dtype), shape)
        data_2 = vadds(data_sel,
                       tvm.const(const_neg_one * const_log_threshold_2, dtype))
        data2_vmuls = vmuls(data_2, tvm.const(const_three_four, dtype))

        if in_dynamic_and_static_unify():
            data_sel = vcmpsel(data_sel, threshold_2, 'ge', data2_vmuls, data_sel)
        else:
            index_2 = vcmp(data_sel, threshold_2, 'ge')
            data_sel = vsel(index_2, data2_vmuls, data_sel)
        data_sel = cast_to(data_sel, dtype)

        # phase3: taylor expands
        taylor = _taylor_compute(data_sel)

        # phase4:return back to original data
        # add log(4/3)
        if in_dynamic_and_static_unify():
            res = vcmpsel(data_sel, threshold_2, 'ge',
                          vadds(taylor, tvm.const(log_four_three, dtype)), taylor)
            res = cast_to(res, dtype)
            # add log(5/3)
            res = vcmpsel(data, threshold_1, 'ge',
                          vadds(taylor, tvm.const(log_five_three, dtype)), res)
        else:
            res = vsel(index_2, vadds(taylor, tvm.const(log_four_three, dtype)),
                       taylor)
            res = cast_to(res, dtype)
            # add log(5/3)
            res = vsel(index_1, vadds(taylor, tvm.const(log_five_three, dtype)),
                       res)
        res = _cast(res, dtype)
        # d: vlog:

        return res

    def _log_compute_block_gt_1(data_x, shape):
        res = _log_compute_block_lt_2_gt_1(data_x, shape)
        res = _log_compute_block_gt_2(data_x, res, shape)

        return res

    def _log_compute_block_gt_half_lt_1(data_x, res, shape):
        threshold_5 = tbe.dsl.broadcast(tvm.const(const_one, dtype), shape)
        data = vadds(data_x, tvm.const(const_neg_one, dtype))
        taylor = _taylor_compute(data)
        if in_dynamic_and_static_unify():
            res = vcmpsel(data_x, threshold_5, 'le', taylor, res)
        else:
            index_6 = vcmp(data_x, threshold_5, 'le')
            res = vsel(index_6, taylor, res)
        res = cast_to(res, dtype)

        return res

    def _log_compute_block_lt_half(data_x, res, shape):
        threshold_4 = tbe.dsl.broadcast(tvm.const(const_half, dtype), shape)
        if in_dynamic_and_static_unify():
            res = vcmpsel(data_x, threshold_4, 'le',
                          vmuls(_log_compute_block_gt_1(vrec(data_x, "high_precision"), shape), const_neg_one), res)
        else:
            index_5 = vcmp(data_x, threshold_4, 'le')
            res = vsel(index_5, vmuls(_log_compute_block_gt_1(vrec(data_x, "high_precision"), shape),
                                      const_neg_one), res)
        res = cast_to(res, dtype)

        return res

    res = _log_compute_block_gt_1(data_x, shape)

    res = _log_compute_block_gt_half_lt_1(data_x, res, shape)

    res = _log_compute_block_lt_half(data_x, res, shape)

    return res


@source_info_decorator()
@dtype_check_decorator
def vlog(raw_tensor, impl_mode="high_performance"):
    """
    calculate ln(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : log(raw_tensor)
    """
    if not intrinsic_check_support("Intrinsic_vln", "float32") \
            and impl_mode == "high_precision":
        return __vlog_calculate_by_taylor(raw_tensor)

    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_log')


@source_info_decorator()
@dtype_check_decorator
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


@source_info_decorator()
@dtype_check_decorator
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


@source_info_decorator()
@dtype_check_decorator
def vrec(raw_tensor, impl_mode="high_performance"):
    """
    calculate vrec(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vrec(raw_tensor)
    """
    dtype = raw_tensor.dtype

    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_rec', args=[impl_mode])


def _check_multi_compute_pattern(pattern, *tensors):
    """
    check tensors.op is matched with pattern or not.
    """
    for pat in pattern:
        if isinstance(pat, tuple):
            if len(tensors) != len(pat):
                return False
            tmp_tensors = ()
            for idx, tag in enumerate(pat):
                try:
                    tensor_tag = tensors[idx].op.tag
                except Exception:       # 'pylint: disable=broad-except
                    tensor_tag = "unknown"
                if tag not in tensor_tag:
                    return False
                if not isinstance(tensors[idx].op, tvm.tensor.PlaceholderOp):
                    tmp_tensors += tuple(tensors[idx].op.input_tensors)
            tensors = tmp_tensors
        elif isinstance(pat, str):
            if len(tensors) != 1:
                return False
            try:
                tensor_tag = tensors[0].op.tag
            except Exception:       # 'pylint: disable=broad-except
                tensor_tag = "unknown"
            if pat not in tensor_tag:
                return False
            if isinstance(tensors[0].op, tvm.tensor.PlaceholderOp):
                tensors = []
            else:
                tensors = tuple(tensors[0].op.input_tensors)
        else:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "A valid pattern list should be a " \
                                          "string or tuple, while is [%s]" % type(pat)
            raise RuntimeError(dict_args, get_error_message(dict_args))
    return True


@source_info_decorator()
@dtype_check_decorator
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

    const_half = 1.0 / 2
    sqrt_const_iter = 3

    dtype = raw_tensor.dtype

    raw_tensor_news = _cast_tensors_for_instr("vlog", [raw_tensor, ])
    init_res = vlog(raw_tensor_news[0])
    init_res = vmuls(init_res, tvm.const(const_half))
    init_res = vexp(init_res)

    for _ in range(sqrt_const_iter):
        vdiv_inputs = _cast_tensors_for_instr("vdiv", [raw_tensor, init_res])
        res = vdiv(vdiv_inputs[0], vdiv_inputs[1])
        vadd_inputs = _cast_tensors_for_instr("vadd", [res, init_res])
        res = vadd(vadd_inputs[0], vadd_inputs[1],)
        res = vmuls(res, tvm.const(const_half, dtype))
        init_res = res
    return res


@source_info_decorator()
@dtype_check_decorator
def vsqrt(raw_tensor, impl_mode="high_performance"):
    """
    calculate vsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vsqrt(raw_tensor)
    """
    if not intrinsic_check_support("Intrinsic_vsqrt"):
        if impl_mode == "high_precision":
            return __vsqrt_calculate_by_newton(raw_tensor)
        dtype = raw_tensor.dtype
        res = __single_elewise_op(raw_tensor, dtype, 'elewise_single_rsqrt')
        return vrec(res, "high_precision")
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
    const_half = 1.0 / 2
    sqrt_const_iter = 3

    dtype = raw_tensor.dtype
    raw_tensor_news = _cast_tensors_for_instr("vlog", [raw_tensor, ])
    init_res = vlog(raw_tensor_news[0])
    init_res = vmuls(init_res, tvm.const(const_half))
    init_res = vexp(init_res)

    for _ in range(sqrt_const_iter):
        vdiv_inputs = _cast_tensors_for_instr("vdiv", [raw_tensor, init_res])
        res = vdiv(vdiv_inputs[0], vdiv_inputs[1])
        vadd_inputs = _cast_tensors_for_instr("vadd", [res, init_res])
        res = vadd(vadd_inputs[0], vadd_inputs[1],)
        res = vmuls(res, tvm.const(const_half, dtype))
        init_res = res
    return vrec(res, "high_precision")


@source_info_decorator()
@dtype_check_decorator
def vrsqrt(raw_tensor, impl_mode="high_performance"):
    """
    calculate vrsqrt(raw_tensor)

    Parameters
    ----------
    raw_tensor : wrapped_tensor or tvm.tensor
    impl_mode : only support high_performance and high_precision

    Returns
    -------
    wrapped_tensor : vrsqrt(raw_tensor)
    """
    if not intrinsic_check_support("Intrinsic_vsqrt") \
            and impl_mode == "high_precision":
        return __vrsqrt_calculate_by_newton(raw_tensor)
    dtype = raw_tensor.dtype
    return __single_elewise_op(raw_tensor, dtype, 'elewise_single_rsqrt')


def _check_elewise_single_shape(input_tensor):
    """
    check the input_tensor's shape whether is positive integer
    :param input_tensor
    """
    for i in range(len(input_tensor.shape)):
        if input_tensor.shape[i].value <= 0 \
                or isinstance(input_tensor.shape[i].value, int) is False:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The input shape value [%s] " \
                                          "must be a positive integer or -1!" \
                                          % input_tensor.shape[i].value
            raise RuntimeError(dict_args, get_error_message(dict_args))

# pylint: disable=too-many-locals, too-many-branches, too-many-statements


def __single_elewise_op(input_tensor, dtype, op_name, args=None):
    """
    factory method of single elewise operations
    """
    if not operation_context.in_dynamic():
        _check_elewise_single_shape(input_tensor)
    shape = shape_to_list(input_tensor.shape)
    if op_name == "elewise_single_log":
        lambda_func = lambda *indice: tvm.log(input_tensor(*indice))
    elif op_name == "elewise_single_exp":
        lambda_func = lambda *indice: tvm.exp(input_tensor(*indice))
    elif op_name == "elewise_single_rec":
        lambda_func = lambda *indice: 1 / input_tensor(*indice)
    elif op_name == "elewise_single_VS_add":
        lambda_func = lambda *indice: input_tensor(*indice) + util_astype(args[0], dtype)
    elif op_name == "elewise_single_VS_mul":
        lambda_func = lambda *indice: input_tensor(*indice) * util_astype(args[0], dtype)
    elif op_name == "elewise_single_VS_max":
        lambda_func = lambda *indice: tvm.max(input_tensor(*indice), util_astype(args[0], dtype))
    elif op_name == "elewise_single_VS_min":
        lambda_func = lambda *indice: tvm.min(input_tensor(*indice), util_astype(args[0], dtype))
    elif op_name == "elewise_single_abs":
        lambda_func = lambda *indice: tvm.select(input_tensor(*indice) >= 0, input_tensor(*indice),
                                                 - input_tensor(*indice))
    elif op_name == "elewise_single_relu":
        lambda_func = lambda *indice: tvm.select(input_tensor(*indice) >= 0, input_tensor(*indice),
                                                 tvm.const(0, dtype=dtype))
    elif op_name == "elewise_single_not":
        lambda_func = lambda *indice: ~input_tensor(*indice)
    elif op_name == "elewise_single_sqrt":
        lambda_func = lambda *indice: tvm.sqrt(input_tensor(*indice))
    elif op_name == "elewise_single_rsqrt":
        lambda_func = lambda *indice: tvm.rsqrt(input_tensor(*indice))
    else:
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "operation %s not support yet" % op_name
        raise RuntimeError(dict_args, get_error_message(dict_args))
    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    is_use_newton_iter = False
    if op_name == "elewise_single_rec" and args[0] == "high_precision":
        is_use_newton_iter = True

    if is_use_newton_iter:
        def __get_newton_iter_num():
            newton_iter_num = 2
            if get_soc_spec(SOC_VERSION) == ASCEND_310:
                newton_iter_num = 1
            return newton_iter_num

        newton_iter_num = __get_newton_iter_num()
        name_pre = op_name.split("_")[-1] + "_"
        const_num_neg_one = tvm.const(-1, dtype=dtype)
        const_num_two = tvm.const(2, dtype=dtype)

        # newton iteration formula is x(n) = x(n-1)(2 - ax(n-1))
        for _ in range(newton_iter_num):
            # the name of each compute
            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp_mul = a*x(n-1)
            with tvm.tag_scope("elewise_binary_mul"):
                tmp_mul = tvm.compute(
                    shape,
                    lambda *indice: input_tensor(*indice) * tmp(*indice),
                    name=name)

            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp_negative = -1*temp_mul
            # pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_single_VS_mul"):
                tmp_negative = tvm.compute(
                    shape,
                    lambda *indice: tmp_mul(*indice) * const_num_neg_one,
                    name=name)

            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp_plus = 2 + tmp_negative
            # pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_single_VS_add"):
                tmp_plus = tvm.compute(
                    shape,
                    lambda *indice: tmp_negative(*indice) + const_num_two,
                    name=name)
            name = name_pre + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            # compute tmp = x(n-1)*tmp_plus
            # pylint: disable=cell-var-from-loop
            with tvm.tag_scope("elewise_binary_mul"):
                tmp = tvm.compute(shape,
                                  lambda *indice: tmp_plus(*indice) * tmp(*indice),
                                  name=name)

    return tmp


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = \
            "The second input type must be [%s], while type is [%s]" % (
            'tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __binary_elewise_op(lhs, rhs, "elewise_binary_mul")


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = \
            "The second input type must be [%s], while type is [%s]" % (
                'tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if not intrinsic_check_support("Intrinsic_vdiv"):
        dtype = rhs.dtype
        reciprocal_rhs = __single_elewise_op(rhs, dtype, 'elewise_single_rec', ['high_precision'])
        vdiv_value = __binary_elewise_op(lhs, reciprocal_rhs, "elewise_binary_mul")
        return vdiv_value

    return __binary_elewise_op(lhs, rhs, "elewise_binary_div")


def __vmod_small_hisi(lhs, rhs):
    from .cast import floor
    # small hisi
    dtype = lhs.dtype
    res_div = vdiv(lhs, rhs)
    res_floor = floor(res_div)
    res_floor = _cast(res_floor, dtype)
    res_mul = vmul(rhs, res_floor)
    res = vsub(lhs, res_mul)

    return _cast(res, dtype)


def __vmod_cloud(lhs, rhs):
    from .cast import floor
    # cloud
    dtype = lhs.dtype
    lhs = _cast(lhs, "float32")
    rhs = _cast(rhs, "float32")
    res_div = vdiv(lhs, rhs)
    res_floor = floor(res_div)
    res_floor = _cast(res_floor, "float32")
    res_mul = vmul(rhs, res_floor)
    res = vsub(lhs, res_mul)

    return _cast(res, dtype)


# pylint: disable=too-many-locals
def __vmod_mini(lhs, rhs):
    from .cast import floor
    dtype = rhs.dtype
    rhs_f16 = rhs
    # 1. calculate result for testing, using float32 for better precision
    lhs = _cast(lhs, "float32")
    rhs = _cast(rhs, "float32")
    test_div = vmul(lhs, vrec(rhs, "high_precision"))
    test_div = _cast(test_div, "float16")
    test_floor = _cast(floor(test_div), "float32")
    test_res = vsub(lhs, vmul(rhs, test_floor))

    # 2. correct the floor result, using float16
    test_res = _cast(test_res, dtype)
    test_floor = _cast(test_floor, dtype)
    zero = tbe.dsl.broadcast(0.0, lhs.shape, dtype)

    if in_dynamic_and_static_unify():
        # rhs positive: 0 <= res < rhs
        prhs_floor = vcmpsel(test_res, zero, 'lt', vadds(test_floor, -1.0), test_floor)
        # rhs negative: rhs < res <= 0
        nrhs_floor = vcmpsel(test_res, zero, 'gt', vadds(test_floor, -1.0), test_floor)

        # according to positive and negative rhs to choose p_floor or n_floor
        result_floor = vcmpsel(rhs_f16, zero, 'gt', prhs_floor, nrhs_floor)
    else:
        # rhs positive: 0 <= res < rhs
        prhs = vcmp(test_res, zero, 'lt', mode='bool')
        prhs_floor = vsel(prhs, vadds(test_floor, -1.0), test_floor)
        # rhs negative: rhs < res <= 0
        nrhs = vcmp(test_res, zero, 'gt', mode='bool')
        nrhs_floor = vsel(nrhs, vadds(test_floor, -1.0), test_floor)

        # according to positive and negative rhs to choose p_floor or n_floor
        rhs_f16_gt_zero = vcmp(rhs_f16, zero, 'gt', mode='bool')
        result_floor = vsel(rhs_f16_gt_zero, prhs_floor, nrhs_floor)

    # 3. calculate the final result, using float32 for better precision
    result_floor = _cast(result_floor, "float32")
    result = vsub(lhs, vmul(rhs, result_floor))

    return _cast(result, dtype)


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = \
            "The first input type must be [%s], while type is [%s]" % (
                'tvm.tensor', type(lhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if not isinstance(rhs, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = \
            "The second input type must be [%s], while type is [%s]" % (
                'tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    _check_elewise_binary_shape(lhs, rhs)
    if lhs.dtype != rhs.dtype:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = \
            "dtype must be the same while lhType is [%s], rhType is [%s]" \
            % (lhs.dtype, rhs.dtype)
        raise RuntimeError(dict_args, get_error_message(dict_args))

    # cloud using vdiv. mini using vrec for division calculation,
    # and mini should improve vmod calculation accuracy.
    if (not intrinsic_check_support("Intrinsic_vdiv")) and \
            (not intrinsic_check_support("Intrinsic_vconv", "f322s32f")):
        if lhs.dtype not in ("float16", ):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = \
                "dtype must be float16, while dtype is [%s]" % lhs.dtype
            raise RuntimeError(dict_args, get_error_message(dict_args))
        res = __vmod_mini(lhs, rhs)
    elif not intrinsic_check_support("Intrinsic_vconv", "f322s32f"):
        if lhs.dtype not in ("float16", ):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = \
                "dtype must be float16, while dtype is [%s]" % lhs.dtype
            raise RuntimeError(dict_args, get_error_message(dict_args))
        res = __vmod_small_hisi(lhs, rhs)
    else:
        if lhs.dtype not in ("float16", "float32"):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = \
                "dtype must be float16 or float32, while dtype is [%s]" % lhs.dtype
            raise RuntimeError(dict_args, get_error_message(dict_args))
        res = __vmod_cloud(lhs, rhs)

    return res


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def is_conv_oper(tensor):
        if hasattr(tensor.op, "reduce_axis") and len(tensor.op.reduce_axis) == 2 and \
                hasattr(tensor.op, "tag") and "conv" in tensor.op.tag:
            return True
        if tensor.op.input_tensors:
            for input_tensor in tensor.op.input_tensors:
                return is_conv_oper(input_tensor)
        else:
            return False

    if is_conv_oper(rhs):
        return __binary_elewise_op(rhs, lhs, "elewise_binary_add")

    return __binary_elewise_op(lhs, rhs, "elewise_binary_add")


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __binary_elewise_op(lhs, rhs, "elewise_binary_sub")


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __binary_elewise_op(lhs, rhs, "elewise_binary_min")


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __binary_elewise_op(lhs, rhs, "elewise_binary_max")


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __binary_elewise_op(lhs, rhs, "elewise_binary_or")


@source_info_decorator()
@dtype_check_decorator
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __binary_elewise_op(lhs, rhs, "elewise_binary_and")


@source_info_decorator()
@dtype_check_decorator
def vaxpy(lhs, rhs, scalar):
    """
    calculate elewise scalar*lhs + rhs, return the min one
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
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if isinstance(scalar, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The third input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('scalar', type(scalar))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __binary_elewise_op(lhs, rhs, "elewise_binary_scalar_axpy",
                               args=[scalar])


def _vcmp_supported_types(mode):
    supported_types = None
    # the get_cmpmask need 16b aligned, so if is float32, should cast to float16
    # bit model using vcmpv. v200 support float32. v100 only support float16
    if mode == 'bit':
        supported_types = ['float16']

    return supported_types


# pylint: disable=too-many-branches, too-many-statements
@source_info_decorator()
@dtype_check_decorator
def vcmp(lhs, rhs, operation='lt', mode='bool'):
    """
    calculate elewise compare

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    operation : operator type, eq, ne, lt, gt, ge, le

    mode : bool, the dtype of return value is bool
           bit, the dtype of return value is uint8(dynamic is uint1)

    Returns
    -------
    wrapped_tensor
    """
    def __vcmp_input_check(lhs, operation, mode, shape):

        if not isinstance(lhs, tvm.tensor.Tensor):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The input type must be [%s], " \
                                          "while type is [%s]" \
                                          % ('tvm.tensor', type(lhs))
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if operation not in ['eq', 'ne', 'lt', 'gt', 'ge', 'le']:
            dict_args = dict()
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "vcmp does not support the " \
                                          "operation: %s, The operation's " \
                                          "value must be eq, ne, lt, gt, ge, le!" % operation
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if mode not in ['bool', 'bit']:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The op's mode must be bit and bool," \
                                          " while mode is [%s]" % mode
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if mode == 'bit' and isinstance(shape[-1], int) and shape[-1] % 8 != 0:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "in bit mode the last dim must be " \
                                          "mutiply of 8, while last dim is [%s]" % shape[-1]
            raise RuntimeError(dict_args, get_error_message(dict_args))

    shape = shape_to_list(lhs.shape)
    __vcmp_input_check(lhs, operation, mode, shape)

    # dynamic realize
    if in_dynamic_and_static_unify():

        # check rhs dtype and if rhs not tensor, change to lhs type
        def __check_and_change_rhs_dtype(lhs, rhs):
            if isinstance(rhs, tvm.tensor.Tensor):
                if lhs.dtype != rhs.dtype:
                    dict_args = dict()
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "dtype must be the same, " \
                                                  "while lhs is %s, rhs is %s" % (lhs.dtype, rhs.dtype)
                    raise RuntimeError(dict_args, get_error_message(dict_args))
            else:
                rhs = get_tvm_scalar(rhs, lhs.dtype)

            return rhs

        rhs = __check_and_change_rhs_dtype(lhs, rhs)

        # generate lambada function
        def __generate_dynamic_lambda_func(lhs, rhs, operation):
            if isinstance(rhs, tvm.tensor.Tensor):
                if operation == 'lt':
                    lambda_func = lambda *indice: (lhs(*indice) < rhs(*indice)).astype("uint1")
                elif operation == 'gt':
                    lambda_func = lambda *indice: (lhs(*indice) > rhs(*indice)).astype("uint1")
                elif operation == 'le':
                    lambda_func = lambda *indice: (lhs(*indice) <= rhs(*indice)).astype("uint1")
                elif operation == 'ge':
                    lambda_func = lambda *indice: (lhs(*indice) >= rhs(*indice)).astype("uint1")
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        (tvm.expr.EQ(lhs(*indice), rhs(*indice))).astype("uint1")
                else:
                    lambda_func = lambda *indice: \
                        (tvm.expr.NE(lhs(*indice), rhs(*indice))).astype("uint1")
            else:
                if operation == 'lt':
                    lambda_func = lambda *indice: (lhs(*indice) < rhs).astype("uint1")
                elif operation == 'gt':
                    lambda_func = lambda *indice: (lhs(*indice) > rhs).astype("uint1")
                elif operation == 'le':
                    lambda_func = lambda *indice: (lhs(*indice) <= rhs).astype("uint1")
                elif operation == 'ge':
                    lambda_func = lambda *indice: (lhs(*indice) >= rhs).astype("uint1")
                elif operation == 'eq':
                    lambda_func = lambda *indice: (tvm.expr.EQ(lhs(*indice), rhs)).astype("uint1")
                else:
                    lambda_func = lambda *indice: (tvm.expr.NE(lhs(*indice), rhs)).astype("uint1")

            return lambda_func

        dynamic_lambda_func = __generate_dynamic_lambda_func(lhs, rhs, operation)

        # mode bit compute
        cmp_op = "elewise_binary_vcmpv_" + operation
        cmp_name = "vcmp_bit_result_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1
        with tvm.tag_scope(cmp_op):
            vcmp_bit_result = tvm.compute(shape, dynamic_lambda_func, name=cmp_name)

        if mode == "bit":
            return vcmp_bit_result

        sel_op = "elewise_multiple_sel"
        const_one_float16 = tvm.const(1, "float16")
        const_zero_float16 = tvm.const(0, "float16")
        sel_lambda_func = lambda *indice: \
            tvm.select(vcmp_bit_result(*indice), const_one_float16, const_zero_float16)
        sel_name = "vcmp_bool_sel_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1
        with tvm.tag_scope(sel_op):
            vcmp_bool_float16_result = tvm.compute(shape, sel_lambda_func, name=sel_name)

        # float16 convert to bool
        convert_lambda_func = lambda *indice: vcmp_bool_float16_result(*indice).astype("bool")
        convert_op = "elewise_single_cast"
        convert_name = "vcmp_bool_cast_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1
        with tvm.tag_scope(convert_op):
            vcmp_bool_result = tvm.compute(shape, convert_lambda_func, name=convert_name)

        return vcmp_bool_result

    supported_types = _vcmp_supported_types(mode)

    # the output is bool or uint8, is not the same as input,
    # no need to cast to back in auto schedule
    if isinstance(rhs, tvm.tensor.Tensor):
        if lhs.dtype != rhs.dtype:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "dtype must be the same, " \
                                          "while lhs is %s, rhs is %s" % (lhs.dtype, rhs.dtype)
            raise RuntimeError(dict_args, get_error_message(dict_args))
        lhs = auto_cast_tensor(lhs, 'vcmp', supported_types, is_auto_cast=False)
        rhs = auto_cast_tensor(rhs, 'vcmp', supported_types, is_auto_cast=False)
    else:
        lhs = auto_cast_tensor(lhs, 'vcmp', supported_types, is_auto_cast=False)
        rhs = get_tvm_scalar(rhs, lhs.dtype)

    cmp_op = "emit_insn_elewise_binary_cmp"

    # generate lambda function
    def __generate_lambda_func(lhs, rhs, operation):
        if isinstance(rhs, tvm.tensor.Tensor):
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
            else:
                lambda_func = lambda *indice: \
                    tvm.expr.NE(lhs(*indice), rhs(*indice))
        else:
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
            else:
                lambda_func = lambda *indice: tvm.expr.NE(lhs(*indice), rhs)

        return lambda_func

    lambda_func = __generate_lambda_func(lhs, rhs, operation)
    name = cmp_op.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    if mode == 'bit':
        shape = shape_to_list(lhs.shape)
        if shape[-1] % 8 != 0:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "in bit mode the last dim must be " \
                                          "mutiply of 8, while last dim is [%s]" % \
                                          shape[-1]
            raise RuntimeError(dict_args, get_error_message(dict_args))

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

        cmp_op = cmp_op + "|" + operation + "|" + mode

        with tvm.tag_scope(cmp_op):
            output = tvm.compute(res_shape, _compute, name='output')
        return output

    cmp_op = cmp_op + "|" + operation + "|" + mode

    with tvm.tag_scope(cmp_op):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


@source_info_decorator()
@dtype_check_decorator
def vlogic(lhs, rhs=None, operation='logic_and'):
    """
    calculate elewise logic operation

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    rhs : wrapped_tensor or tvm.tensor
        right hand tensor

    operation : operator type, logic_and, logic_or, logic_not

    Returns
    -------
    wrapped_tensor
    """
    if in_dynamic_and_static_unify():
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "Dynamic shape not support vlogic"
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if operation not in ['logic_and', 'logic_or', 'logic_not']:
        dict_args = dict()
        dict_args["errCode"] = "E90002"
        dict_args["detailed_cause"] = "vlogic does not support the " \
                                      "operation: %s, The operation's " \
                                      "value must be logic_and, logic_or, logic_not!" % operation
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if not isinstance(lhs, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The lhs input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(lhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if operation != "logic_not" and not isinstance(rhs, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The rhs input type must be [%s], " \
                                      "while type is [%s]" \
                                      % ('tvm.tensor', type(rhs))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if operation == "logic_not":
        rhs = tvm.placeholder(lhs.shape, name="rhs", dtype=lhs.dtype)
    # the output is bool is not the same as input,
    # no need to cast to back in auto schedule
    return __binary_elewise_op(lhs, rhs, "elewise_binary_logic", args=[operation[6:]])


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def __binary_elewise_op(tensor_l, tensor_r, op_name, args=None):
    """
    factory method of binary elewise operations
    """
    _check_elewise_binary_shape(tensor_l, tensor_r)
    if tensor_l.dtype != tensor_r.dtype and op_name != "elewise_binary_scalar_axpy":
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "dtype must be the same, while lhType " \
                                      "is [%s], rhType is [%s]" \
                                      % (tensor_l.dtype, tensor_r.dtype)
        raise RuntimeError(dict_args, get_error_message(dict_args))
    shape = tensor_l.shape
    name = op_name.split("_")[-1] + "_" + str(NAME_INDEX[0])
    NAME_INDEX[0] += 1

    if op_name == "elewise_binary_add":
        lambda_func = lambda *indice: tensor_l(*indice) + tensor_r(*indice)
    elif op_name == "elewise_binary_sub":
        lambda_func = lambda *indice: tensor_l(*indice) - tensor_r(*indice)
    elif op_name == "elewise_binary_div":
        lambda_func = lambda *indice: tensor_l(*indice) / tensor_r(*indice)
    elif op_name == "elewise_binary_mul":
        lambda_func = lambda *indice: tensor_l(*indice) * tensor_r(*indice)
    elif op_name == "elewise_binary_min":
        lambda_func = lambda *indice: \
            tvm.min(tensor_l(*indice), tensor_r(*indice))
    elif op_name == "elewise_binary_max":
        lambda_func = lambda *indice: \
            tvm.max(tensor_l(*indice), tensor_r(*indice))
    elif op_name == "elewise_binary_and":
        lambda_func = lambda *indice: tensor_l(*indice) & tensor_r(*indice)
    elif op_name == "elewise_binary_or":
        lambda_func = lambda *indice: tensor_l(*indice) | tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_le":
        lambda_func = lambda *indice: tensor_l(*indice) <= tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_lt":
        lambda_func = lambda *indice: tensor_l(*indice) < tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_ge":
        lambda_func = lambda *indice: tensor_l(*indice) >= tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_gt":
        lambda_func = lambda *indice: tensor_l(*indice) > tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_ne":
        lambda_func = lambda *indice: tensor_l(*indice) != tensor_r(*indice)
    elif op_name == "elewise_binary_vcmpv_eq":
        lambda_func = lambda *indice: tensor_l(*indice) == tensor_r(*indice)
    elif op_name == "elewise_binary_scalar_axpy":
        intr = "v" + op_name.split("_")[-1]
        is_support_dtype = intrinsic_check_support("Intrinsic_"+intr,
                                                   tensor_l.dtype)
        if tensor_l.dtype != tensor_r.dtype:
            if tensor_l.dtype != "float16" or tensor_r.dtype != "float32":
                dict_args = dict()
                dict_args["errCode"] = "E90002"
                dict_args["detailed_cause"] = "dtype error, vaxpy not support " \
                                              "mixed data type auto cast, " \
                                              "while tensor_l is [%s], " \
                                              "tensor_r is [%s]" \
                                              % (tensor_l.dtype, tensor_r.dtype)
                raise RuntimeError(dict_args, get_error_message(dict_args))
        elif not is_support_dtype:
            dict_args = dict()
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "dtype error, vaxpy not support " \
                                          "mixed data type auto cast, " \
                                          "while tensor_l is [%s], " \
                                          "tensor_r is [%s]" \
                                          % (tensor_l.dtype, tensor_r.dtype)
            raise RuntimeError(dict_args, get_error_message(dict_args))
        rtype = tensor_r.dtype
        lambda_func = lambda *indice: \
            tvm.expr.Cast(rtype, tensor_l(*indice))*util_astype(args[0], rtype) + tensor_r(*indice)
        op_name = op_name + ("|1,1" if tensor_l == tensor_r else "|0,0")
    elif op_name == "emit_insn_elewise_binary_cmp":
        operation = args[0]
        if operation == 'lt':
            lambda_func = lambda *indice: tensor_l(*indice) < tensor_r(*indice)
        elif operation == 'gt':
            lambda_func = lambda *indice: tensor_l(*indice) > tensor_r(*indice)
        elif operation == 'le':
            lambda_func = lambda *indice: tensor_l(*indice) <= tensor_r(*indice)
        elif operation == 'ge':
            lambda_func = lambda *indice: tensor_l(*indice) >= tensor_r(*indice)
        elif operation == 'eq':
            lambda_func = lambda *indice: \
                tvm.expr.EQ(tensor_l(*indice), tensor_r(*indice))
        elif operation == 'ne':
            lambda_func = lambda *indice: \
                tvm.expr.NE(tensor_l(*indice), tensor_r(*indice))
        else:
            dict_args = dict()
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "vcmp do not support the " \
                                          "input op_name: %s" % operation
            raise RuntimeError(dict_args, get_error_message(dict_args))
    elif op_name == "elewise_binary_logic":
        operation = args[0]
        if operation == 'and':
            lambda_func = lambda *indice: tensor_l(*indice) & tensor_r(*indice)
        elif operation == 'or':
            lambda_func = lambda *indice: tensor_l(*indice) | tensor_r(*indice)
        elif operation == 'not':
            lambda_func = lambda *indice: ~tensor_l(*indice)
        else:
            dict_args = dict()
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "vlogic do not support the " \
                                          "input op_name: %s" % operation
            raise RuntimeError(dict_args, get_error_message(dict_args))

    else:
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "operation %s not support yet" % op_name
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if op_name == "emit_insn_elewise_binary_cmp" and args[1] == 'bit':
        shape = shape_to_list(shape)
        if shape[-1] % 8 != 0:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The input shape's last axis must be " \
                                          "mutiply of 8, while last dim is [%s]" % \
                                          shape[-1]
            raise RuntimeError(dict_args, get_error_message(dict_args))

        k = tvm.reduce_axis((0, 8), name='k')
        res_shape = shape
        res_shape[-1] = res_shape[-1] // 8

        def _compute(*index):
            """
            elewise compare for bit
            """
            res_index = []
            for i, value in enumerate(index):
                if i == len(index) - 1:
                    res_index.append(value*8 + k)
                else:
                    res_index.append(value)
            tensor = tvm.bit(lambda_func(*res_index).astype('uint8'), axis=k)
            return tensor

        op_name = op_name + "|" + args[0] + "|" + args[1]

        with tvm.tag_scope(op_name):
            output = tvm.compute(res_shape, _compute, name='output')
        return output

    if op_name in ("emit_insn_elewise_binary_cmp",
                   "elewise_binary_logic"):
        for arg in args:
            op_name = op_name + "|" + arg

    with tvm.tag_scope(op_name):
        tmp = tvm.compute(shape, lambda_func, name=name)

    return tmp


def _check_elewise_binary_shape(lhs, rhs):
    """
    check elewise binary shape
    :param lhs: left tensor
    :param rhs: right tensor
    :return:
    """
    if len(lhs.shape) != len(rhs.shape):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The lhs ndim [%s] must be equal" \
                                      " to the rhs [%s]" % (len(lhs.shape), len(rhs.shape))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    for _l, _r in zip(lhs.shape, rhs.shape):
        if not equal(_l, _r):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The lhs shape [%s] must be equal" \
                                    " to the rhs [%s]" % (lhs.shape, rhs.shape)
            raise RuntimeError(dict_args, get_error_message(dict_args))

    if not operation_context.in_dynamic():
        for sh_value in lhs.shape:
            if sh_value.value <= 0 \
                    or not isinstance(sh_value.value, int):
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The input shape value [%s] must be a positive integer" % sh_value.value
                raise RuntimeError(dict_args, get_error_message(dict_args))


def _check_is_equal(lhs, rhs):
    """
    check lhs and rhs value is equal
    :param lhs: left tensor
    :param rhs: right tensor
    :return:
    """
    if lhs.value == rhs.value:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "when lhs and rhs are all scalar, " \
                                      "lhs should unequal to rhs"
        raise RuntimeError(dict_args, get_error_message(dict_args))


@source_info_decorator()
@dtype_check_decorator
def vmla(tensor_0, tensor_1, tensor_2):
    """
    calculate x*tensor_1 + tensor_2,  only support float16, float32
    Parameters
    ----------
    x : wrapped_tensor or tvm.tensor
    tensor_1 : wrapped_tensor or tvm.tensor
    tensor_2 : wrapped_tensor or tvm.tensor
    Returns
    -------
    wrapped_tensor : X*tensor_1 + tensor_2
    """
    if not isinstance(tensor_1, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % ('tvm.tensor', type(tensor_1))
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The third input type must be [%s], " \
                                      "while type is [%s]" % ('tvm.tensor', type(tensor_2))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_mla")


@source_info_decorator()
@dtype_check_decorator
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
    if not isinstance(tensor_1, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The second input type must be [%s], " \
                                      "while type is [%s]" % (
                                      'tvm.tensor', type(tensor_1))
        raise RuntimeError(dict_args, get_error_message(dict_args))
    if not isinstance(tensor_2, tvm.tensor.Tensor):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The third input type must be [%s], " \
                                      "while type is [%s]" % (
                                      'tvm.tensor', type(tensor_2))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    return __multiple_elewise_op(tensor_0, tensor_1, tensor_2,
                                 "elewise_multiple_madd")


def __multiple_elewise_op(tensor_0, tensor_1, tensor_2, op_name):
    """
    factory method of binary multiple operations
    """
    intr = "v" + op_name.split("_")[-1]
    is_support_dtype = intrinsic_check_support("Intrinsic_"+intr,
                                               tensor_0.dtype)

    _check_multiple_elewise_op_shape(tensor_0, tensor_1, tensor_2)
    if tensor_0.dtype != tensor_1.dtype or tensor_0.dtype != tensor_2.dtype \
            or tensor_2.dtype != tensor_1.dtype:
        if op_name != "elewise_multiple_mla" \
                or tensor_0.dtype != tensor_1.dtype \
                or tensor_0.dtype != "float16" \
                or tensor_2.dtype != "float32":
            dict_args = dict()
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "dtype error, vmla not support " \
                                          "mixed data type auto cast, " \
                                          "while tensor_0 is [%s], " \
                                          "tensor_1 is [%s]" \
                                          "tensor_2 is [%s]" \
                                          % (tensor_0.dtype, tensor_1.dtype, tensor_2.dtype)
            raise RuntimeError(dict_args, get_error_message(dict_args))
    elif not is_support_dtype:
        dict_args = dict()
        dict_args["errCode"] = "E90002"
        dict_args["detailed_cause"] = "dtype error, vmla not support " \
                                      "mixed data type auto cast, " \
                                      "while tensor_0 is [%s], " \
                                      "tensor_1 is [%s]" \
                                      "tensor_2 is [%s]" \
                                      % (tensor_0.dtype, tensor_1.dtype,
                                         tensor_2.dtype)
        raise RuntimeError(dict_args, get_error_message(dict_args))

    shape = tensor_0.shape
    if op_name == "elewise_multiple_mla":
        ztype = tensor_2.dtype
        lambda_func = lambda *indice: tvm.expr.Cast(ztype,
                                                    tensor_0(*indice) * tensor_1(*indice)) + tensor_2(*indice)
    elif op_name == "elewise_multiple_madd":
        lambda_func = lambda *indice: tensor_0(*indice) * tensor_2(*indice) + tensor_1(*indice)
    elif op_name == "elewise_multiple_maddrelu":
        lambda_func = lambda *indice: \
            tvm.relu(tensor_0(*indice) * tensor_2(*indice) + tensor_1(*indice))
    else:
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "operation %s not support yet" % op_name
        raise RuntimeError(dict_args, get_error_message(dict_args))

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


def _check_multiple_elewise_op_shape(tensor_0, tensor_1, tensor_2):
    """
    check multiple elewise op's shape
    :param tensor_0:
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    if len(tensor_0.shape) != len(tensor_1.shape) or len(tensor_0.shape) != len(tensor_2.shape) \
            or len(tensor_2.shape) != len(tensor_1.shape):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The input shape ndim must be equal to" \
                                      " the each other, " \
                                      "while tensor_0 ndim is [%s], " \
                                      "tensor_1 ndim is [%s], " \
                                      "tensor_3 ndim is [%s]" \
                                      % (len(tensor_0.shape), len(tensor_1.shape), len(tensor_2.shape))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    for _a, _b, _c in zip(tensor_0.shape, tensor_1.shape, tensor_2.shape):
        if not (equal(_a, _b) and equal(_b, _c)):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The input shape must be equal to the each other, " \
                                          "while tensor_0 shape is [%s], " \
                                          "tensor_1 shape is [%s], " \
                                          "tensor_3 shape is [%s]" \
                                          % (tensor_0.shape, tensor_1.shape, tensor_2.shape)
            raise RuntimeError(dict_args, get_error_message(dict_args))

    if not operation_context.in_dynamic():
        for i in range(len(tensor_0.shape)):
            if tensor_0.shape[i].value <= 0 or isinstance(tensor_0.shape[i].value, int) is False:
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The input shape value must " \
                                              "be a positive integer, " \
                                              "while shape is [%s]" % tensor_0.shape[i].value
                raise RuntimeError(dict_args, get_error_message(dict_args))


def _vsel_bit_shape_check(condition, input_tensor):
    """
    check vsel_bit's shape
    :param condition:
    :param input_tensor:
    :return:
    """
    if len(condition.shape) != len(input_tensor.shape):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "The condition ndim [%s] must " \
                                      "be equal to the input_tensor [%s]" \
                                      % (len(condition.shape), len(input_tensor.shape))
        raise RuntimeError(dict_args, get_error_message(dict_args))

    for i in range(len(condition.shape)):
        if i == len(condition.shape) - 1:
            if (input_tensor.shape[i].value % 8 != 0) \
                    or (input_tensor.shape[i].value // 8 != condition.shape[i].value):
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "the sel tensor's last dim [%s] " \
                                              "must be multiple of 8 " \
                                              "and div the last dim of " \
                                              "condition shape [%s] is 8" \
                                              % (input_tensor.shape[i].value,
                                                 input_tensor.shape[i].value // condition.shape[i].value)
                raise RuntimeError(dict_args, get_error_message(dict_args))
        else:
            if condition.shape[i].value != input_tensor.shape[i].value:
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The lhs shape [%s] must be " \
                                              "equal to the rhs [%s]" \
                                              % (condition.shape[i].value, input_tensor.shape[i].value)
                raise RuntimeError(dict_args, get_error_message(dict_args))

    for i in range(len(input_tensor.shape)):
        if input_tensor.shape[i].value <= 0 \
                or isinstance(input_tensor.shape[i].value, int) is False:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The input shape value [%s] must " \
                                          "be a positive integer" % input_tensor.shape[i].value
            raise RuntimeError(dict_args, get_error_message(dict_args))


# pylint: disable=too-many-branches, too-many-statements
@source_info_decorator()
@dtype_check_decorator
def vsel(condition, lhs, rhs):
    """
    if condition = ture, the result is lhs,
        select

    Parameters
    ----------
    condition : wrapped_tensor or tvm.tensor, the dtype is bool or uint8(dynamic is uint1)

    lhs : wrapped_tensor or tvm.tensor or scalar

    rhs : wrapped_tensor or tvm.tensor or scalar

    Returns
    -------
    wrapped_tensor :
    """

    def __vsel_input_check(condition):

        if not isinstance(condition, tvm.tensor.Tensor):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = \
                "The condition type must be [%s], while type is [%s]" % (
                'tvm.tensor', type(condition))
            raise RuntimeError(dict_args, get_error_message(dict_args))
    __vsel_input_check(condition)

    src_dtype = "float16"

    def _get_vsel_input_type(inputs):
        type_strs = []
        for one in inputs:
            if isinstance(one, tvm.tensor.Tensor):
                type_strs.append("TENSOR")
            else:
                type_strs.append("SCALAR")

        return "_".join(type_strs)

    input_type_str = _get_vsel_input_type([lhs, rhs])

    # two const input check
    def _check_two_scalar_inputs(lhs, rhs):
        # if lhs,rhs are all scalar, only support float16
        if judge_var(lhs) == "tvm_const" and lhs.dtype != "float16":
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "when lhs and rhs are all scalar, " \
                                          "only support float16, while lhs.dtype is [%s]" % lhs.dtype
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if judge_var(rhs) == "tvm_const" and rhs.dtype != "float16":
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "when lhs and rhs are all scalar, " \
                                          "only support float16, while rhs.dtype is [%s]" % rhs.dtype
            raise RuntimeError(dict_args, get_error_message(dict_args))

    # dynamic realize
    def _dynamic_check_shape(condition_shape):
        if isinstance(condition_shape[-1], int) and condition_shape[-1] % 8 != 0:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "vsel last dim must be " \
                                          "mutiply of 8, while last dim is [%s]" % condition_shape[-1]
            raise RuntimeError(dict_args, get_error_message(dict_args))

    if in_dynamic_and_static_unify():
        shape = condition.shape
        _dynamic_check_shape(shape)

        def _check_dynamic_condition_dtype(condition_dtype):
            if condition_dtype not in ("uint1", "bool"):
                dict_args = dict()
                dict_args["errCode"] = "E90003"
                dict_args["detailed_cause"] = "Dynamic shape vsel only support uint1 and bool, " \
                                              "but condtion dtype is [%s]" % condition_dtype
                raise RuntimeError(dict_args, get_error_message(dict_args))

        condition_dtype = condition.dtype
        _check_dynamic_condition_dtype(condition_dtype)

        if condition.dtype == "bool":
            cast_lambda_func = lambda *index: condition(*index).astype("float16")
            vector_cast_op = "elewise_single_cast"
            cast_name = "vsel_bool_cast_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            with tvm.tag_scope(vector_cast_op):
                cast_float16 = tvm.compute(shape, cast_lambda_func, name=cast_name)

            cmp_lambda_func = lambda *indice: (cast_float16(*indice) > tvm.const(0, "float16")).astype("uint1")
            cmp_op = "elewise_binary_vcmpv_gt"
            cmp_gt_name = "vsel_bool_vcmpv_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            with tvm.tag_scope(cmp_op):
                condition = tvm.compute(shape, cmp_lambda_func, name=cmp_gt_name)

        # mode bit dtype = "uint1"
        op_name = "elewise_multiple_sel"

        def _get_vsel_dynamic_lambda_func(input_type_str, condition, lhs, rhs):
            if input_type_str == "TENSOR_TENSOR":
                _check_multiple_elewise_op_shape(condition, lhs, rhs)
                dynamic_lambda_func = lambda *indice: \
                    tvm.select(condition(*indice), lhs(*indice), rhs(*indice))
            elif input_type_str == "SCALAR_TENSOR":
                _check_elewise_binary_shape(condition, rhs)
                lhs = get_tvm_scalar(lhs, rhs.dtype)
                dynamic_lambda_func = lambda *indice: \
                    tvm.select(condition(*indice), lhs, rhs(*indice))
            elif input_type_str == "TENSOR_SCALAR":
                _check_elewise_binary_shape(condition, lhs)
                rhs = get_tvm_scalar(rhs, lhs.dtype)
                dynamic_lambda_func = lambda *indice: \
                    tvm.select(condition(*indice), lhs(*indice), rhs)
            else:
                # if lhs,rhs are all scalar, only support float16
                _check_two_scalar_inputs(lhs, rhs)
                lhs = get_tvm_scalar(lhs, "float16")
                rhs = get_tvm_scalar(rhs, "float16")
                dynamic_lambda_func = lambda *indice: \
                    tvm.select(condition(*indice), lhs, rhs)

            return dynamic_lambda_func

        dynamic_lambda_func = _get_vsel_dynamic_lambda_func(input_type_str, condition, lhs, rhs)
        name = "sel_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1
        with tvm.tag_scope(op_name):
            tmp = tvm.compute(shape, dynamic_lambda_func, name=name)
        return tmp

    op_name = "emit_insn_elewise_multiple_sel"
    if condition.dtype == "bool":
        mode = 'bool'
        shape = condition.shape
        if input_type_str == "TENSOR_TENSOR":
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel')
            rhs = auto_cast_tensor(rhs, 'vsel')
            _check_multiple_elewise_op_shape(condition, lhs, rhs)
            lambda_func = lambda *indice: \
                tvm.select(condition(*indice), lhs(*indice), rhs(*indice))
        elif input_type_str == "SCALAR_TENSOR":
            _check_elewise_binary_shape(condition, rhs)
            src_dtype = rhs.dtype
            rhs = auto_cast_tensor(rhs, 'vsel')
            lhs = get_tvm_scalar(lhs, rhs.dtype)
            lambda_func = lambda *indice: \
                tvm.select(condition(*indice), lhs, rhs(*indice))
        elif input_type_str == "TENSOR_SCALAR":
            _check_elewise_binary_shape(condition, lhs)
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel')
            rhs = get_tvm_scalar(rhs, lhs.dtype)
            lambda_func = lambda *indice: \
                tvm.select(condition(*indice), lhs(*indice), rhs)
        else:
            # if lhs,rhs are all scalar, only support float16
            _check_two_scalar_inputs(lhs, rhs)

            lhs = get_tvm_scalar(lhs, "float16")
            rhs = get_tvm_scalar(rhs, "float16")
            _check_is_equal(lhs, rhs)

            lambda_func = lambda *indice: tvm.select(condition(*indice), lhs, rhs)

        name = "sel" + "_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1
        op_name = op_name + '|' + mode
        with tvm.tag_scope(op_name):
            tmp = tvm.compute(shape, lambda_func, name=name)
    elif condition.dtype == "uint8":
        mode = 'bit'
        shape_condition = shape_to_list(condition.shape)
        shape = shape_condition
        shape[-1] = shape[-1] * 8

        supported_type = ["float16"]

        def get_indice(indice):
            """
            get indice
            """
            res_index = []
            for i, value in enumerate(indice):
                if i == len(indice) - 1:
                    res_index.append(value // 8)
                else:
                    res_index.append(value)
            return res_index

        if input_type_str == "TENSOR_TENSOR":
            _check_elewise_binary_shape(lhs, rhs)
            _vsel_bit_shape_check(condition, lhs)
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel', supported_type)
            rhs = auto_cast_tensor(rhs, 'vsel', supported_type)

            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'),
                                  lhs(*indice), rhs(*indice))
        elif input_type_str == "SCALAR_TENSOR":
            _vsel_bit_shape_check(condition, rhs)
            src_dtype = rhs.dtype
            rhs = auto_cast_tensor(rhs, 'vsel', supported_type)
            lhs = get_tvm_scalar(lhs, rhs.dtype)

            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'),
                                  lhs, rhs(*indice))
        elif input_type_str == "TENSOR_SCALAR":
            _vsel_bit_shape_check(condition, lhs)
            src_dtype = lhs.dtype
            lhs = auto_cast_tensor(lhs, 'vsel', supported_type)
            rhs = get_tvm_scalar(rhs, lhs.dtype)

            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'),
                                  lhs(*indice), rhs)
        else:
            # if lhs,rhs are all scalar, only support float16
            _check_two_scalar_inputs(lhs, rhs)

            lhs = get_tvm_scalar(lhs, "float16")
            rhs = get_tvm_scalar(rhs, "float16")
            _check_is_equal(lhs, rhs)

            def _compute(*indice):
                res_index = get_indice(indice)
                return tvm.select(condition(*res_index).astype('bool'), lhs, rhs)

        name = "sel" + "_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1

        op_name = op_name + '|' + mode
        with tvm.tag_scope(op_name):
            tmp = tvm.compute(shape, _compute, name=name)
    else:
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "condition only support bool and " \
                                      "uint8, but condition dtype is [%s]" % condition.dtype
        raise RuntimeError(dict_args, get_error_message(dict_args))
 
    res_dtype = tmp.dtype
    if src_dtype != res_dtype:
        tmp = _cast(tmp, src_dtype, is_auto_cast=False)
    return tmp


def _vcmpsel_data_shape_check(*args):
    """
    check vcmpsel's data shape
    :param args:
    :return:
    """
    arg_temp = args[0]

    for sh_value in arg_temp.shape:
        if operation_context.in_dynamic():
            if not isinstance(sh_value, tvm.expr.Expr):
                if sh_value.value == 0 or not isinstance(sh_value.value, int):
                    dict_args = dict()
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "dynamic input shape value [%s]" \
                                                  " must be a nonzero integer or variable!" % sh_value.value
                    raise RuntimeError(dict_args, get_error_message(dict_args))
        else:
            if sh_value.value <= 0 \
                    or not isinstance(sh_value.value, int):
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The input shape value [%d] " \
                                              "must be a positive integer!" % sh_value.value
                raise RuntimeError(dict_args, get_error_message(dict_args))

    for arg in args:
        if len(arg.shape) != len(arg_temp.shape):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The input shape ndim must be equal" \
                                          " to the each other! " \
                                          "while arg dims is [%s], " \
                                          "arg_temp dims is [%s]" % (len(arg.shape), len(arg_temp.shape))
            raise RuntimeError(dict_args, get_error_message(dict_args))

    for i in range(len(arg_temp.shape)):
        for arg in args:
            if not equal(arg_temp.shape[i], arg.shape[i]):
                dict_args = dict()
                dict_args["errCode"] = "E90001"
                dict_args["detailed_cause"] = "The lhs shape [%s] must be " \
                                              "equal to the rhs [%s]" % (arg_temp.shape, arg.shape)
                raise RuntimeError(dict_args, get_error_message(dict_args))


def _vcmpsel_data_dtype_check(*args):
    """
    check vcmpsel's data type
    :param args:
    :return:
    """
    arg_temp = args[0]

    for arg in args:
        if arg.dtype != arg_temp.dtype:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The input dtype must be the same " \
                                          "to the each other! while arg.dtype is [%s], " \
                                          "arg_temp.dtype is [%s]" % (arg.dtype, arg_temp.dtype)
            raise RuntimeError(dict_args, get_error_message(dict_args))


@source_info_decorator()
@dtype_check_decorator
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
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = \
                "The second input type must be [%s], while type is [%s]" % (
                'tvm.tensor', type(lhs))
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if operation not in ['eq', 'ne', 'lt', 'gt', 'ge', 'le']:
            dict_args = dict()
            dict_args["errCode"] = "E90002"
            dict_args["detailed_cause"] = "vcmpsel does not support the " \
                                          "operation: %s, The operation's " \
                                          "value must be eq, ne, lt, gt, ge, le!" % operation
            raise RuntimeError(dict_args, get_error_message(dict_args))

        if in_dynamic_and_static_unify():
            if not dsl_check_support("tbe.dsl.vcmpsel", lhs.dtype):
                dict_args = dict()
                dict_args["errCode"] = "E90002"
                dict_args["detailed_cause"] = "dynamic tbe.dsl.vcmpsel is not" \
                                              " supported [%s]!" % (lhs.dtype,)
                raise RuntimeError(dict_args, get_error_message(dict_args))

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

    def _get_cmpvs_lambda_func(operation, lhs, rhs):
        if in_dynamic_and_static_unify():
            dynamic_cmpvs_lambda_func_dict = {
                'lt': lambda *indice: (lhs(*indice) < rhs).astype("uint1"),
                'gt': lambda *indice: (lhs(*indice) > rhs).astype("uint1"),
                'le': lambda *indice: (lhs(*indice) <= rhs).astype("uint1"),
                'ge': lambda *indice: (lhs(*indice) >= rhs).astype("uint1"),
                'eq': lambda *indice: (tvm.expr.EQ(lhs(*indice), rhs)).astype("uint1"),
                'ne': lambda *indice: (tvm.expr.NE(lhs(*indice), rhs)).astype("uint1")
            }
            lambda_func = dynamic_cmpvs_lambda_func_dict[operation]
        else:
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
                dict_args = dict()
                dict_args["errCode"] = "E90002"
                dict_args["detailed_cause"] = "vcmp do not support the " \
                                              "input op_name: %s" % operation
                raise RuntimeError(dict_args, get_error_message(dict_args))
        return lambda_func

    def _get_cmpv_lambda_func(operation, lhs, rhs):
        if in_dynamic_and_static_unify():
            dynamic_cmpv_lambda_func_dict = {
                'lt': lambda *indice: (lhs(*indice) < rhs(*indice)).astype("uint1"),
                'gt': lambda *indice: (lhs(*indice) > rhs(*indice)).astype("uint1"),
                'le': lambda *indice: (lhs(*indice) <= rhs(*indice)).astype("uint1"),
                'ge': lambda *indice: (lhs(*indice) >= rhs(*indice)).astype("uint1"),
                'eq': lambda *indice: (tvm.expr.EQ(lhs(*indice), rhs(*indice))).astype("uint1"),
                'ne': lambda *indice: (tvm.expr.NE(lhs(*indice), rhs(*indice))).astype("uint1")
            }
            lambda_func = dynamic_cmpv_lambda_func_dict[operation]
        else:
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
                dict_args = dict()
                dict_args["errCode"] = "E90002"
                dict_args["detailed_cause"] = "vcmp do not support the " \
                                              "input op_name: %s" % operation
                raise RuntimeError(dict_args, get_error_message(dict_args))
        return lambda_func

    cmp_op = "elewise_binary_vcmpv_" + operation
    sel_op = "elewise_multiple_sel"
    cmpsel_op = "elewise_binary_cmpsel_" + operation

    def get_vcmpsel_input_type(rhs, slhs, srhs):
        type_strs = []
        if isinstance(rhs, tvm.tensor.Tensor):
            type_strs.append("TENSOR")
        else:
            type_strs.append("SCALAR")

        if isinstance(slhs, tvm.tensor.Tensor):
            type_strs.append("TENSOR")
        else:
            type_strs.append("SCALAR")

        if isinstance(srhs, tvm.tensor.Tensor):
            type_strs.append("TENSOR")
        else:
            type_strs.append("SCALAR")

        return "_".join(type_strs)

    input_type_str = get_vcmpsel_input_type(rhs, slhs, srhs)

    if input_type_str == "SCALAR_SCALAR_SCALAR":
        if not in_dynamic_and_static_unify():
            lhs = auto_cast_tensor(lhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)
        slhs = get_tvm_scalar(slhs, lhs.dtype)
        srhs = get_tvm_scalar(srhs, lhs.dtype)

        def _vcmpsel_tsss_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16":
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
                    dict_args = dict()
                    dict_args["errCode"] = "E90002"
                    dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                                  "input op_name: %s" % operation
                    raise RuntimeError(dict_args, get_error_message(dict_args))
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

    if input_type_str == "TENSOR_SCALAR_SCALAR":
        _vcmpsel_data_shape_check(lhs, rhs)
        _vcmpsel_data_dtype_check(lhs, rhs)
        if not in_dynamic_and_static_unify():
            lhs = auto_cast_tensor(lhs, "vsel")
            rhs = auto_cast_tensor(rhs, "vsel")
        slhs = get_tvm_scalar(slhs, lhs.dtype)
        srhs = get_tvm_scalar(srhs, lhs.dtype)

        def _vcmpsel_ttss_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16":
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
                    dict_args = dict()
                    dict_args["errCode"] = "E90002"
                    dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                                  "input op_name: %s" % operation
                    raise RuntimeError(dict_args, get_error_message(dict_args))
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

    if input_type_str == "SCALAR_TENSOR_SCALAR":
        _vcmpsel_data_shape_check(lhs, slhs)
        _vcmpsel_data_dtype_check(lhs, slhs)
        if not in_dynamic_and_static_unify():
            lhs = auto_cast_tensor(lhs, "vsel")
            slhs = auto_cast_tensor(slhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)
        srhs = get_tvm_scalar(srhs, lhs.dtype)

        def _vcmpsel_tsts_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16":
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
                    dict_args = dict()
                    dict_args["errCode"] = "E90002"
                    dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                                  "input op_name: %s" % operation
                    raise RuntimeError(dict_args, get_error_message(dict_args))
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

    if input_type_str == "SCALAR_SCALAR_TENSOR":
        _vcmpsel_data_shape_check(lhs, srhs)
        _vcmpsel_data_dtype_check(lhs, srhs)
        if not in_dynamic_and_static_unify():
            srhs = auto_cast_tensor(srhs, "vsel")
            lhs = auto_cast_tensor(lhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)
        slhs = get_tvm_scalar(slhs, lhs.dtype)

        def _vcmpsel_tsst_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16":
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
                    dict_args = dict()
                    dict_args["errCode"] = "E90002"
                    dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                                  "input op_name: %s" % operation
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                return lambda_func

            lambda_func = _get_cmpsel_tsst_lambda_func(operation,
                                                       lhs, rhs, slhs, srhs)
            name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
            NAME_INDEX[0] += 1
            cmpsel_op = cmpsel_op + "|" + operation
            with tvm.tag_scope(cmpsel_op):
                tmp = tvm.compute(shape, lambda_func, name=name)

            return tmp

        return _vcmpsel_tsst_compute(cmp_op, sel_op, cmpsel_op, shape,
                                     lhs, rhs, operation, slhs, srhs)

    if input_type_str == "TENSOR_TENSOR_SCALAR":
        _vcmpsel_data_shape_check(lhs, rhs, slhs)
        _vcmpsel_data_dtype_check(lhs, rhs, slhs)
        if not in_dynamic_and_static_unify():
            lhs = auto_cast_tensor(lhs, "vsel")
            rhs = auto_cast_tensor(rhs, "vsel")
            slhs = auto_cast_tensor(slhs, "vsel")
        srhs = get_tvm_scalar(srhs, lhs.dtype)

        def _vcmpsel_ttts_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16":
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
                        tvm.select(lhs(*indice) < rhs(*indice), slhs(*indice), srhs)
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs(*indice), slhs(*indice), srhs)
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs(*indice), slhs(*indice), srhs)
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs(*indice), slhs(*indice), srhs)
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs(*indice), slhs(*indice), srhs)
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs(*indice), slhs(*indice), srhs)
                else:
                    dict_args = dict()
                    dict_args["errCode"] = "E90002"
                    dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                                  "input op_name: %s" % operation
                    raise RuntimeError(dict_args, get_error_message(dict_args))
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

    if input_type_str == "TENSOR_SCALAR_TENSOR":
        _vcmpsel_data_shape_check(lhs, rhs, srhs)
        _vcmpsel_data_dtype_check(lhs, rhs, srhs)
        if not in_dynamic_and_static_unify():
            lhs = auto_cast_tensor(lhs, "vsel")
            rhs = auto_cast_tensor(rhs, "vsel")
            srhs = auto_cast_tensor(srhs, "vsel")
        slhs = get_tvm_scalar(slhs, lhs.dtype)

        def _vcmpsel_ttst_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16":
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
                        tvm.select(lhs(*indice) < rhs(*indice), slhs, srhs(*indice))
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs(*indice), slhs, srhs(*indice))
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs(*indice), slhs, srhs(*indice))
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs(*indice), slhs, srhs(*indice))
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs(*indice), slhs, srhs(*indice))
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs(*indice), slhs, srhs(*indice))
                else:
                    dict_args = dict()
                    dict_args["errCode"] = "E90002"
                    dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                                  "input op_name: %s" % operation
                    raise RuntimeError(dict_args, get_error_message(dict_args))
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

    if input_type_str == "SCALAR_TENSOR_TENSOR":
        _vcmpsel_data_shape_check(lhs, slhs, srhs)
        _vcmpsel_data_dtype_check(lhs, slhs, srhs)
        if not in_dynamic_and_static_unify():
            lhs = auto_cast_tensor(lhs, "vsel")
            slhs = auto_cast_tensor(slhs, "vsel")
            srhs = auto_cast_tensor(srhs, "vsel")
        rhs = get_tvm_scalar(rhs, lhs.dtype)

        def _vcmpsel_tstt_compute(cmp_op, sel_op, cmpsel_op, shape,
                                  lhs, rhs, operation, slhs, srhs):
            if lhs.dtype == "float16":
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
                        tvm.select(lhs(*indice) < rhs, slhs(*indice), srhs(*indice))
                elif operation == 'gt':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) > rhs, slhs(*indice), srhs(*indice))
                elif operation == 'le':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) <= rhs, slhs(*indice), srhs(*indice))
                elif operation == 'ge':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) >= rhs, slhs(*indice), srhs(*indice))
                elif operation == 'eq':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) == rhs, slhs(*indice), srhs(*indice))
                elif operation == 'ne':
                    lambda_func = lambda *indice: \
                        tvm.select(lhs(*indice) != rhs, slhs(*indice), srhs(*indice))
                else:
                    dict_args = dict()
                    dict_args["errCode"] = "E90002"
                    dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                                  "input op_name: %s" % operation
                    raise RuntimeError(dict_args, get_error_message(dict_args))
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

    _vcmpsel_data_shape_check(lhs, rhs, slhs, srhs)
    _vcmpsel_data_dtype_check(lhs, rhs, slhs, srhs)

    if not in_dynamic_and_static_unify():
        lhs = auto_cast_tensor(lhs, "vsel")
        rhs = auto_cast_tensor(rhs, "vsel")
        slhs = auto_cast_tensor(slhs, "vsel")
        srhs = auto_cast_tensor(srhs, "vsel")

    def _vcmpsel_tttt_compute(cmp_op, sel_op, cmpsel_op, shape,
                              lhs, rhs, operation, slhs, srhs):
        if lhs.dtype == "float16":
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
                dict_args = dict()
                dict_args["errCode"] = "E90002"
                dict_args["detailed_cause"] = "vcmpsel do not support the " \
                                              "input op_name: %s" % operation
                raise RuntimeError(dict_args, get_error_message(dict_args))
            return lambda_func

        lambda_func = _get_cmpsel_tttt_lambda_func(operation,
                                                   lhs, rhs, slhs, srhs)
        name = cmpsel_op.split("_")[-2] + "_" + str(NAME_INDEX[0])
        NAME_INDEX[0] += 1
        cmpsel_op = cmpsel_op + "|" + operation
        with tvm.tag_scope(cmpsel_op):
            tmp = tvm.compute(shape, lambda_func, name=name)

        return tmp

    return _vcmpsel_tttt_compute(cmp_op, sel_op, cmpsel_op, shape,
                                 lhs, rhs, operation, slhs, srhs)
