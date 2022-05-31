# # -*- coding:utf-8 -*-
import warnings

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe.dsl.base.padding import util
from tbe.dsl.base.padding.simulators.binary_sml import (AddSimulator,
                                                        DivSimulator,
                                                        MaxSimulator,
                                                        MinSimulator,
                                                        MulSimulator,
                                                        SubSimulator)
from tbe.dsl.base.padding.value import PaddingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_padding_impl")


######## Add UT ########
@add_cust_test_func(ut_case)
def test_add_adjust_exact_exact(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_x(8, "int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_add_adjust_exact_tensor(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_add_adjust_exact_any(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_add_adjust_tensor_exact(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(2147483640, "int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


@add_cust_test_func(ut_case)
def test_add_adjust_tensor_any(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


@add_cust_test_func(ut_case)
def test_add_adjust_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(2147483640, "int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_add_adjust_any_tensor(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_add_adjust_any_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = AddSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_add_calc_exact_exact(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_x(7, "int32")

    pvalue = AddSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483647


@add_cust_test_func(ut_case)
def test_add_calc_exact_tensor(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = AddSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_add_calc_exact_any(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = AddSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_add_calc_tensor_exact(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    pvalue = AddSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_add_calc_tensor_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = AddSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_add_adjust_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    pvalue = AddSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_sub_adjust_exact_exact(_):
    pvalue0 = util.new_pvalue_x(-2147483640, "int32")
    pvalue1 = util.new_pvalue_x(9, "int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


######## Sub UT ########
@add_cust_test_func(ut_case)
def test_sub_adjust_exact_tensor(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_sub_adjust_exact_any(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_sub_adjust_tensor_exact(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(2147483640, "int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


@add_cust_test_func(ut_case)
def test_sub_adjust_tensor_any(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


@add_cust_test_func(ut_case)
def test_sub_adjust_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(2147483640, "int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_sub_adjust_any_tensor(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_sub_adjust_any_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = SubSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


@add_cust_test_func(ut_case)
def test_sub_calc_exact_exact(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_x(-7, "int32")

    pvalue = SubSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483647


@add_cust_test_func(ut_case)
def test_sub_calc_exact_tensor(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = SubSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_sub_calc_exact_any(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = SubSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_sub_calc_tensor_exact(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    pvalue = SubSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_sub_calc_tensor_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = SubSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_sub_calc_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    pvalue = SubSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


######## Mul UT ########
@add_cust_test_func(ut_case)
def test_mul_adjust_exact_exact(_):
    pvalue0 = util.new_pvalue_x(214748365, "int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_mul_adjust_exact_tensor(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_mul_adjust_exact_any(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_mul_adjust_tensor_exact(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


@add_cust_test_func(ut_case)
def test_mul_adjust_tensor_any(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 0)


@add_cust_test_func(ut_case)
def test_mul_adjust_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_mul_adjust_any_tensor(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_mul_adjust_any_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = MulSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_mul_calc_exact_exact(_):
    pvalue0 = util.new_pvalue_x(214748364, "int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483640


@add_cust_test_func(ut_case)
def test_mul_calc_exact_0_tensor(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_mul_calc_exact_1_tensor(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_mul_calc_exact_0_any(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_mul_calc_exact_1_any(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_mul_calc_tensor_exact_0(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_mul_calc_tensor_exact_1(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(1, "int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_mul_calc_tensor_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_mul_calc_any_exact_0(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_mul_calc_any_exact_1(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(1, "int32")

    pvalue = MulSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


######## Div UT ########
@add_cust_test_func(ut_case)
def test_div_adjust_exact_exact(_):
    pvalue0 = util.new_pvalue_x(2147483647, "int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 1)


@add_cust_test_func(ut_case)
def test_div_adjust_exact_tensor(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_div_adjust_exact_any(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_div_adjust_tensor_exact(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(2147483640, "int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, None)


@add_cust_test_func(ut_case)
def test_div_adjust_tensor_exact_0(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(0, "int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 1)


@add_cust_test_func(ut_case)
def test_div_adjust_tensor_any(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 1)


@add_cust_test_func(ut_case)
def test_div_adjust_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(2147483640, "int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 1)


@add_cust_test_func(ut_case)
def test_div_adjust_any_tensor(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (0, None)


@add_cust_test_func(ut_case)
def test_div_adjust_any_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_any("int32")

    i_x = DivSimulator._do_adjust(pvalue0, pvalue1)
    return i_x == (None, 1)


@add_cust_test_func(ut_case)
def test_div_calc_exact_exact(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 214748364


@add_cust_test_func(ut_case)
def test_div_calc_exact_0_tensor(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_div_calc_exact_1_tensor(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_div_calc_exact_0_any(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_div_calc_exact_1_any(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_div_calc_tensor_exact(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(1, "int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_div_calc_tensor_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_div_calc_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(1, "int32")

    pvalue = DivSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


######## Max UT ########
@add_cust_test_func(ut_case)
def test_max_calc_exact_exact(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_x(7, "int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483640


@add_cust_test_func(ut_case)
def test_max_calc_exact_max_tensor(_):
    pvalue0 = util.new_pvalue_x(2147483647, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483647


@add_cust_test_func(ut_case)
def test_max_calc_exact_min_tensor(_):
    pvalue0 = util.new_pvalue_x(-2147483648, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_max_calc_exact_x_tensor(_):
    pvalue0 = util.new_pvalue_x(-100, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_max_calc_exact_max_any(_):
    pvalue0 = util.new_pvalue_x(2147483647, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483647


@add_cust_test_func(ut_case)
def test_max_calc_exact_x_any(_):
    pvalue0 = util.new_pvalue_x(100, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_max_calc_tensor_exact_max(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(2147483647, "int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483647


@add_cust_test_func(ut_case)
def test_max_calc_tensor_exact_min(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(-2147483648, "int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_max_calc_tensor_exact_x(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(-100, "int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_max_calc_tensor_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_max_calc_any_exact_max(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(2147483647, "int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 2147483647


@add_cust_test_func(ut_case)
def test_max_calc_any_exact_x(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    pvalue = MaxSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


######## Min UT ########
@add_cust_test_func(ut_case)
def test_min_calc_exact_exact(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    pvalue1 = util.new_pvalue_x(7, "int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == 7


@add_cust_test_func(ut_case)
def test_min_calc_exact_min_tensor(_):
    pvalue0 = util.new_pvalue_x(-2147483648, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == -2147483648


@add_cust_test_func(ut_case)
def test_min_calc_exact_max_tensor(_):
    pvalue0 = util.new_pvalue_x(2147483647, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_min_calc_exact_x_tensor(_):
    pvalue0 = util.new_pvalue_x(-100, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_min_calc_exact_min_any(_):
    pvalue0 = util.new_pvalue_x(-2147483648, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == -2147483648


@add_cust_test_func(ut_case)
def test_min_calc_exact_x_any(_):
    pvalue0 = util.new_pvalue_x(100, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_min_calc_tensor_exact_min(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(-2147483648, "int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == -2147483648


@add_cust_test_func(ut_case)
def test_min_calc_tensor_exact_max(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(2147483647, "int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_min_calc_tensor_exact_x(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(-100, "int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_min_calc_tensor_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_min_calc_any_exact_min(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(-2147483648, "int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.value == -2147483648


@add_cust_test_func(ut_case)
def test_min_calc_any_exact_x(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    pvalue = MinSimulator._do_calc(pvalue0, pvalue1)
    return pvalue.type == PaddingValueType.ANY


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        ret = v.test_func(None)
        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
