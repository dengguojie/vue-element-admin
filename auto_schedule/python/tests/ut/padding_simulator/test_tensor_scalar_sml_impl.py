# # -*- coding:utf-8 -*-
import warnings
from sre_constants import ANY

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.dsl.padding import util
from tbe.dsl.padding.simulators.tensor_scalar_sml import AddsSimulator
from tbe.dsl.padding.simulators.tensor_scalar_sml import MaxsSimulator
from tbe.dsl.padding.simulators.tensor_scalar_sml import MinsSimulator
from tbe.dsl.padding.simulators.tensor_scalar_sml import MulsSimulator
from tbe.dsl.padding.value import PaddingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_tensor_scalar_sml_impl")


# Adds
@add_cust_test_func(ut_case)
def test_adds_adjust_exact_const_none(_):
    pvalue0 = util.new_pvalue_x(-100, "int32")
    scalar1 = tvm.const(10, "int32")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_adds_adjust_exact_const_0(_):
    pvalue0 = util.new_pvalue_x(2147483640, "int32")
    scalar1 = tvm.const(10, "int32")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_adds_adjust_exact_max(_):
    pvalue0 = util.new_pvalue_x(65504.0, "float16")
    scalar1 = tvm.var("float16")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_adds_adjust_exact_min(_):
    pvalue0 = util.new_pvalue_x(-65504.0, "float16")
    scalar1 = tvm.var("float16")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_adds_adjust_exact_other(_):
    pvalue0 = util.new_pvalue_x(1, "float16")
    scalar1 = tvm.var("float16")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_adds_adjust_tensor(_):
    pvalue0 = util.new_pvalue_tensor("float16")
    scalar1 = tvm.var("float16")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_adds_adjust_any_0(_):
    pvalue0 = util.new_pvalue_any("float32")
    scalar1 = tvm.const(0, "float32")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_adds_adjust_any_other(_):
    pvalue0 = util.new_pvalue_any("float16")
    scalar1 = tvm.var("float16")
    x = AddsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_adds_calc_exact_const(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    scalar1 = tvm.const(10, "int32")
    pvalue = AddsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.value == 11


@add_cust_test_func(ut_case)
def test_adds_calc_exact_var(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    scalar1 = tvm.var("int32")
    pvalue = AddsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_adds_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    scalar1 = tvm.const(4, "int32")
    pvalue = AddsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_adds_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    scalar1 = tvm.var("int32")
    pvalue = AddsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


# Muls
@add_cust_test_func(ut_case)
def test_muls_adjust_exact_const_none(_):
    pvalue0 = util.new_pvalue_x(-100, "int32")
    scalar1 = tvm.const(10, "int32")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_muls_adjust_exact_const_0(_):
    pvalue0 = util.new_pvalue_x(214748365, "int32")
    scalar1 = tvm.const(10, "int32")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_muls_adjust_exact_max(_):
    pvalue0 = util.new_pvalue_x(65504.0, "float16")
    scalar1 = tvm.var("float16")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_muls_adjust_exact_min(_):
    pvalue0 = util.new_pvalue_x(-65504.0, "float16")
    scalar1 = tvm.var("float16")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_muls_adjust_exact_other(_):
    pvalue0 = util.new_pvalue_x(32, "float16")
    scalar1 = tvm.var("float16")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_muls_adjust_tensor(_):
    pvalue0 = util.new_pvalue_tensor("float16")
    scalar1 = tvm.var("float16")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_muls_adjust_any_0(_):
    pvalue0 = util.new_pvalue_any("float32")
    scalar1 = tvm.const(0, "float32")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x is None


@add_cust_test_func(ut_case)
def test_muls_adjust_any_other(_):
    pvalue0 = util.new_pvalue_any("int32")
    scalar1 = tvm.var("float16")
    x = MulsSimulator._do_adjust(pvalue0, scalar1)
    return x == 0


@add_cust_test_func(ut_case)
def test_muls_calc_exact_const(_):
    pvalue0 = util.new_pvalue_x(2, "int32")
    scalar1 = tvm.const(10, "int32")
    pvalue = MulsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.value == 20


@add_cust_test_func(ut_case)
def test_muls_calc_exact_var(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    scalar1 = tvm.var("int32")
    pvalue = MulsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_muls_calc_exact_0(_):
    pvalue0 = util.new_pvalue_x(0, "int32")
    scalar1 = tvm.var("int32")
    pvalue = MulsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_muls_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    scalar1 = tvm.const(4, "int32")
    pvalue = MulsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_muls_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    scalar1 = tvm.var("int32")
    pvalue = MulsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


# Maxs
@add_cust_test_func(ut_case)
def test_maxs_calc_exact_const(_):
    pvalue0 = util.new_pvalue_x(3, "int32")
    scalar1 = tvm.const(10, "int32")
    pvalue = MaxsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.value == 10


@add_cust_test_func(ut_case)
def test_maxs_calc_exact_var(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    scalar1 = tvm.var("int32")
    pvalue = MaxsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_maxs_calc_exact_max(_):
    pvalue0 = util.new_pvalue_x(3.4028234663852886e+38, "float32")
    scalar1 = tvm.var("float32")
    pvalue = MaxsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.value == 3.4028234663852886e+38


@add_cust_test_func(ut_case)
def test_maxs_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    scalar1 = tvm.const(4, "int32")
    pvalue = MaxsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_maxs_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    scalar1 = tvm.var("int32")
    pvalue = MaxsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


# Maxs
@add_cust_test_func(ut_case)
def test_mins_calc_exact_const(_):
    pvalue0 = util.new_pvalue_x(3, "int32")
    scalar1 = tvm.const(10, "int32")
    pvalue = MinsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.value == 3


@add_cust_test_func(ut_case)
def test_mins_calc_exact_var(_):
    pvalue0 = util.new_pvalue_x(1, "int32")
    scalar1 = tvm.var("int32")
    pvalue = MinsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_mins_calc_exact_min(_):
    pvalue0 = util.new_pvalue_x(-3.4028234663852886e+38, "float32")
    scalar1 = tvm.var("float32")
    pvalue = MinsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.value == -3.4028234663852886e+38


@add_cust_test_func(ut_case)
def test_mins_calc_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    scalar1 = tvm.const(4, "int32")
    pvalue = MinsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_mins_calc_any(_):
    pvalue0 = util.new_pvalue_any("int32")
    scalar1 = tvm.var("int32")
    pvalue = MinsSimulator._do_calc(pvalue0, scalar1)
    return pvalue.type == PaddingValueType.ANY


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
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
