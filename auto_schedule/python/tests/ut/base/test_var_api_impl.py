# # -*- coding:utf-8 -*-
import warnings

from tbe import tvm
from tbe.dsl.base import var_api

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("var_api", "var_api.test_var_api_impl")


@add_cust_test_func(ut_case)
def test_const(_):
    var_ = var_api.const(20)

    return isinstance(var_, tvm.expr.IntImm) and str(var_) == "20"


@add_cust_test_func(ut_case)
def test_var(_):
    var_ = var_api.var("var_0")

    return isinstance(var_, tvm.expr.Var) and str(var_) == "var_0"


@add_cust_test_func(ut_case)
def test_div(_):
    var_0 = var_api.var("var_0")
    var_ = var_api.div(var_0, 10)

    return isinstance(var_, tvm.expr.Div) and str(var_) == "(var_0/10)"


@add_cust_test_func(ut_case)
def test_indexdiv(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_ = var_api.indexdiv(var_0, var_1)

    return isinstance(var_, tvm.expr.FloorDiv) and str(var_) == "floordiv(var_0, var_1)"


@add_cust_test_func(ut_case)
def test_indexmod(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_ = var_api.indexmod(var_0, var_1)

    return isinstance(var_, tvm.expr.FloorMod) and str(var_) == "floormod(var_0, var_1)"


@add_cust_test_func(ut_case)
def test_truncdiv(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_ = var_api.truncdiv(var_0, var_1)

    return isinstance(var_, tvm.expr.Div) and str(var_) == "(var_0/var_1)"


@add_cust_test_func(ut_case)
def test_truncmod(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_ = var_api.truncmod(var_0, var_1)

    return isinstance(var_, tvm.expr.Mod) and str(var_) == "(var_0 % var_1)"


@add_cust_test_func(ut_case)
def test_floordiv(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_ = var_api.floordiv(var_0, var_1)

    return isinstance(var_, tvm.expr.FloorDiv) and str(var_) == "floordiv(var_0, var_1)"


@add_cust_test_func(ut_case)
def test_floormod(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_ = var_api.floormod(var_0, var_1)

    return isinstance(var_, tvm.expr.FloorMod) and str(var_) == "floormod(var_0, var_1)"


@add_cust_test_func(ut_case)
def test_sum(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_2 = var_api.var("var_1")
    var_ = var_api.sum(var_0, var_1, var_2)

    return isinstance(var_, tvm.expr.Add) and str(var_) == "((var_0 + var_1) + var_1)"


@add_cust_test_func(ut_case)
def test_min(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_2 = var_api.var("var_1")
    var_ = var_api.min(var_0, var_1, var_2)

    return isinstance(var_, tvm.expr.Min) and str(var_) == "min(min(var_0, var_1), var_1)"


@add_cust_test_func(ut_case)
def test_max(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_2 = var_api.var("var_1")
    var_ = var_api.max(var_0, var_1, var_2)

    return isinstance(var_, tvm.expr.Max) and str(var_) == "max(max(var_0, var_1), var_1)"


@add_cust_test_func(ut_case)
def test_prod(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_2 = var_api.var("var_1")
    var_ = var_api.prod(var_0, var_1, var_2)

    return isinstance(var_, tvm.expr.Mul) and str(var_) == "((var_0*var_1)*var_1)"


@add_cust_test_func(ut_case)
def test_bit(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_2 = var_api.var("var_1")
    var_ = var_api.bit(var_0, var_1, var_2)

    return isinstance(var_, tvm.expr.Call) and str(var_) == "bitwise_and(bitwise_and(var_0, var_1), var_1)"
