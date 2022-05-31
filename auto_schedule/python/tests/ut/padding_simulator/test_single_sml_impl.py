# # -*- coding:utf-8 -*-
import warnings

import numpy as np
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe.dsl.base.padding import util
from tbe.dsl.base.padding.simulators.single_sml import (AbsSimulator,
                                                        ExpSimulator,
                                                        LogSimulator,
                                                        RecSimulator,
                                                        RsqrtSimulator,
                                                        SqrtSimulator)
from tbe.dsl.base.padding.value import PaddingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_padding_impl")


######## Abs UT ########
@add_cust_test_func(ut_case)
def test_abs_calc_exact(_):
    pvalue0 = util.new_pvalue_x(-100, "int32")
    pvalue = AbsSimulator._do_calc(pvalue0)
    return pvalue.value == 100


@add_cust_test_func(ut_case)
def test_abs_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = AbsSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_abs_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue = AbsSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.ANY


######## Exp UT ########
@add_cust_test_func(ut_case)
def test_exp_adjust_exact_none(_):
    pvalue0 = util.new_pvalue_x(11, "float16")
    x = ExpSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_exp_adjust_exact_0(_):
    pvalue0 = util.new_pvalue_x(11.1, "float16")
    x = ExpSimulator._do_adjust(pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_exp_adjust_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = ExpSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_exp_adjust_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = ExpSimulator._do_adjust(pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_exp_calc_exact(_):
    pvalue0 = util.new_pvalue_x(23.2, "float32")
    pvalue = ExpSimulator._do_calc(pvalue0)
    return np.equal(pvalue.value, np.exp(np.float32(23.2)))


@add_cust_test_func(ut_case)
def test_exp_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = ExpSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


######## Log UT ########
@add_cust_test_func(ut_case)
def test_log_adjust_exact_none(_):
    pvalue0 = util.new_pvalue_x(11.08, "float16")
    x = LogSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_log_adjust_exact_1(_):
    pvalue0 = util.new_pvalue_x(-11.1, "float16")
    x = LogSimulator._do_adjust(pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_log_adjust_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = LogSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_log_adjust_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = LogSimulator._do_adjust(pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_log_calc_exact(_):
    pvalue0 = util.new_pvalue_x(23.2, "float32")
    pvalue = LogSimulator._do_calc(pvalue0)
    return np.equal(pvalue.value, np.log(np.float32(23.2)))


@add_cust_test_func(ut_case)
def test_log_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = LogSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


######## Rec UT ########
@add_cust_test_func(ut_case)
def test_rec_adjust_exact_none(_):
    pvalue0 = util.new_pvalue_x(11.08, "float16")
    x = RecSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_rec_adjust_exact_1(_):
    pvalue0 = util.new_pvalue_x(0, "float16")
    x = RecSimulator._do_adjust(pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_rec_adjust_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = RecSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_rec_adjust_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = RecSimulator._do_adjust(pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_rec_calc_exact(_):
    pvalue0 = util.new_pvalue_x(23.2, "float32")
    pvalue = RecSimulator._do_calc(pvalue0)
    return np.equal(pvalue.value, np.reciprocal(np.float32(23.2)))


@add_cust_test_func(ut_case)
def test_rec_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = RecSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


######## Sqrt UT ########
@add_cust_test_func(ut_case)
def test_sqrt_adjust_exact_none(_):
    pvalue0 = util.new_pvalue_x(11.08, "float16")
    x = SqrtSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_sqrt_adjust_exact_0(_):
    pvalue0 = util.new_pvalue_x(-6.2, "float16")
    x = SqrtSimulator._do_adjust(pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_sqrt_adjust_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = SqrtSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_sqrt_adjust_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = SqrtSimulator._do_adjust(pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_sqrt_calc_exact(_):
    pvalue0 = util.new_pvalue_x(23.2, "float32")
    pvalue = SqrtSimulator._do_calc(pvalue0)
    return np.equal(pvalue.value, np.sqrt(np.float32(23.2)))


@add_cust_test_func(ut_case)
def test_sqrt_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = SqrtSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


######## Rsqrt UT ########
@add_cust_test_func(ut_case)
def test_rsqrt_adjust_exact_none(_):
    pvalue0 = util.new_pvalue_x(11.08, "float16")
    x = RsqrtSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_rsqrt_adjust_exact_1(_):
    pvalue0 = util.new_pvalue_x(-6.2, "float16")
    x = RsqrtSimulator._do_adjust(pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_rsqrt_adjust_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = RsqrtSimulator._do_adjust(pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_rsqrt_adjust_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = RsqrtSimulator._do_adjust(pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_rsqrt_calc_exact(_):
    pvalue0 = util.new_pvalue_x(23.2, "float32")
    pvalue = RsqrtSimulator._do_calc(pvalue0)
    return np.equal(pvalue.value, np.reciprocal(np.sqrt(np.float32(23.2))))


@add_cust_test_func(ut_case)
def test_rsqrt_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = RsqrtSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        ret = v.test_func(None)
        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
