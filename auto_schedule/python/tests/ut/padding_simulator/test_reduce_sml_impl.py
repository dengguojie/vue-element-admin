# # -*- coding:utf-8 -*-
import warnings
from sre_constants import ANY

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe.dsl.base.padding import util
from tbe.dsl.base.padding.simulators.reduce_sml import (ReduceMaxSimulator,
                                                        ReduceMinSimulator,
                                                        ReduceProdSimulator,
                                                        ReduceSumSimulator)
from tbe.dsl.base.padding.value import PaddingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_tensor_scalar_impl")


######## Reduce sum UT ########
@add_cust_test_func(ut_case)
def test_sum_adjust_exact_0_with_pad(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    x = ReduceSumSimulator._do_adjust(True, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_sum_adjust_exact_other_with_pad(_):
    pvalue0 = util.new_pvalue_x(3, "int32")
    x = ReduceSumSimulator._do_adjust(True, pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_sum_adjust_tensor_with_pad(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = ReduceSumSimulator._do_adjust(True, pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_sum_adjust_any_with_pad(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = ReduceSumSimulator._do_adjust(True, pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_sum_adjust_exact_0_without_pad(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    x = ReduceSumSimulator._do_adjust(False, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_sum_adjust_exact_other_without_pad(_):
    pvalue0 = util.new_pvalue_x(3, "int32")
    x = ReduceSumSimulator._do_adjust(False, pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_sum_adjust_tensor_without_pad(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = ReduceSumSimulator._do_adjust(False, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_sum_adjust_any_without_pad(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = ReduceSumSimulator._do_adjust(False, pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_sum_calc_exact_0(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue = ReduceSumSimulator._do_calc(pvalue0)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_sum_calc_exact_other(_):
    pvalue0 = util.new_pvalue_x(3, "int32")
    pvalue = ReduceSumSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_sum_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = ReduceSumSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_sum_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue = ReduceSumSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.ANY


######## Reduce prod UT ########
@add_cust_test_func(ut_case)
def test_prod_adjust_exact_1_with_pad(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    x = ReduceProdSimulator._do_adjust(True, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_prod_adjust_exact_other_with_pad(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    x = ReduceProdSimulator._do_adjust(True, pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_prod_adjust_tensor_with_pad(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = ReduceProdSimulator._do_adjust(True, pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_prod_adjust_any_with_pad(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = ReduceProdSimulator._do_adjust(True, pvalue0)
    return x == 1


@add_cust_test_func(ut_case)
def test_prod_adjust_exact_0_without_pad(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    x = ReduceProdSimulator._do_adjust(False, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_prod_adjust_exact_1_without_pad(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    x = ReduceProdSimulator._do_adjust(False, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_prod_adjust_exact_other_without_pad(_):
    pvalue0 = util.new_pvalue_x(3, "int32")
    x = ReduceProdSimulator._do_adjust(False, pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_prod_adjust_tensor_without_pad(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = ReduceProdSimulator._do_adjust(False, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_prod_adjust_any_without_pad(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = ReduceProdSimulator._do_adjust(False, pvalue0)
    return x == 0


@add_cust_test_func(ut_case)
def test_prod_calc_exact_0(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    pvalue = ReduceProdSimulator._do_calc(pvalue0)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_prod_calc_exact_1(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    pvalue = ReduceProdSimulator._do_calc(pvalue0)
    return pvalue.value == 1


@add_cust_test_func(ut_case)
def test_prod_calc_exact_other(_):
    pvalue0 = util.new_pvalue_x(3, "int32")
    pvalue = ReduceProdSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_prod_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = ReduceProdSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_prod_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue = ReduceProdSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.ANY


######## Reduce max UT ########
@add_cust_test_func(ut_case)
def test_max_adjust_exact_min_with_pad(_):
    pvalue0 = util.new_pvalue_min("int32")
    x = ReduceMaxSimulator._do_adjust(True, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_max_adjust_exact_other_with_pad(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    x = ReduceMaxSimulator._do_adjust(True, pvalue0)
    return x == util.get_min("int32")


@add_cust_test_func(ut_case)
def test_max_adjust_tensor_with_pad(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = ReduceMaxSimulator._do_adjust(True, pvalue0)
    return x == util.get_min("int32")


@add_cust_test_func(ut_case)
def test_max_adjust_any_with_pad(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = ReduceMaxSimulator._do_adjust(True, pvalue0)
    return x == util.get_min("int32")


@add_cust_test_func(ut_case)
def test_max_calc_exact_x(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    pvalue = ReduceMaxSimulator._do_calc(pvalue0)
    return pvalue.value == 10


@add_cust_test_func(ut_case)
def test_max_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = ReduceMaxSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_max_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue = ReduceMaxSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.ANY


######## Reduce min UT ########
@add_cust_test_func(ut_case)
def test_min_adjust_exact_max_with_pad(_):
    pvalue0 = util.new_pvalue_max("int32")
    x = ReduceMinSimulator._do_adjust(True, pvalue0)
    return x is None


@add_cust_test_func(ut_case)
def test_min_adjust_exact_other_with_pad(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    x = ReduceMinSimulator._do_adjust(True, pvalue0)
    return x == util.get_max("int32")


@add_cust_test_func(ut_case)
def test_min_adjust_tensor_with_pad(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    x = ReduceMinSimulator._do_adjust(True, pvalue0)
    return x == util.get_max("int32")


@add_cust_test_func(ut_case)
def test_min_adjust_any_with_pad(_):
    pvalue0 = util.new_pvalue_any("int32")
    x = ReduceMinSimulator._do_adjust(True, pvalue0)
    return x == util.get_max("int32")


@add_cust_test_func(ut_case)
def test_min_calc_exact_x(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    pvalue = ReduceMinSimulator._do_calc(pvalue0)
    return pvalue.value == 10


@add_cust_test_func(ut_case)
def test_min_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue = ReduceMinSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_min_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue = ReduceMinSimulator._do_calc(pvalue0)
    return pvalue.type == PaddingValueType.ANY


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_calc_padding_softmax_div":
        #     continue

        try:
            ret = v.test_func(None)
        except Exception:
            import traceback
            print(f"\033[93mException: {k}\033[0m")
            print(traceback.format_exc())
            continue

        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
