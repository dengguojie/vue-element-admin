# # -*- coding:utf-8 -*-
import warnings

import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from sch_test_frame.ut.helper import padding_helper
from tbe import tvm
from tbe.dsl.base import operation
from tbe.dsl.base import var_api
from tbe.dsl.padding import util
from tbe.dsl.padding.graph import Graph
from tbe.dsl.padding.simulators.cmpsel import scmp
from tbe.dsl.padding.simulators.cmpsel.scmp import CmpMode
from tbe.dsl.padding.value import PaddingValueType
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_scmp_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


# cmp
@add_cust_test_func(ut_case)
def test_cmp(_):
    pvalue0 = util.new_pvalue_0("float16")
    pvalue1 = util.new_pvalue_0("float16")

    rpv = scmp.cmp(pvalue0, pvalue1, "float16", scmp.CmpMode.EQ)

    return rpv.value == 1


# _get_cmp_func
@add_cust_test_func(ut_case)
def test_get_cmp_func_exact_exact(_):
    func = scmp._get_cmp_func((PaddingValueType.EXACT, PaddingValueType.EXACT))

    return func == scmp._cmp_exact_exact


@add_cust_test_func(ut_case)
def test_get_cmp_func_exact_tensor(_):
    func = scmp._get_cmp_func((PaddingValueType.EXACT, PaddingValueType.TENSOR))

    return func == scmp._cmp_exact_tensor


@add_cust_test_func(ut_case)
def test_get_cmp_func_exact_any(_):
    func = scmp._get_cmp_func((PaddingValueType.EXACT, PaddingValueType.ANY))

    return func == scmp._cmp_exact_any


@add_cust_test_func(ut_case)
def test_get_cmp_func_tensor_exact(_):
    func = scmp._get_cmp_func((PaddingValueType.TENSOR, PaddingValueType.EXACT))

    return func == scmp._cmp_tensor_exact


@add_cust_test_func(ut_case)
def test_get_cmp_func_tensor_tensor(_):
    func = scmp._get_cmp_func((PaddingValueType.TENSOR, PaddingValueType.TENSOR))

    return func == scmp._cmp_tensor_tensor


@add_cust_test_func(ut_case)
def test_get_cmp_func_tensor_any(_):
    func = scmp._get_cmp_func((PaddingValueType.TENSOR, PaddingValueType.ANY))

    return func == scmp._cmp_tensor_any


@add_cust_test_func(ut_case)
def test_get_cmp_func_any_exact(_):
    func = scmp._get_cmp_func((PaddingValueType.ANY, PaddingValueType.EXACT))

    return func == scmp._cmp_any_exact


@add_cust_test_func(ut_case)
def test_get_cmp_func_any_tensor(_):
    func = scmp._get_cmp_func((PaddingValueType.ANY, PaddingValueType.TENSOR))

    return func == scmp._cmp_any_tensor


@add_cust_test_func(ut_case)
def test_get_cmp_func_any_any(_):
    func = scmp._get_cmp_func((PaddingValueType.ANY, PaddingValueType.ANY))

    return func == scmp._cmp_any_any


# deal_selected_hs
@add_cust_test_func(ut_case)
def test_deal_selected_hs_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        brc_0 = tbe.dsl.broadcast(3, shape)
        adds_0 = tbe.dsl.vadds(brc_0, 100)

        graph = Graph([brc_0])
        node = graph.get_nodes()[0]
        node.set_pvalue(util.new_pvalue_1("float16"))

        pvalue = scmp.deal_selected_hs(node, adds_0)

        assert_target = node.get_pvalue().target == [adds_0]
        assert_pvalue = pvalue.value == 1

        return all((assert_target, assert_pvalue))


@add_cust_test_func(ut_case)
def test_deal_selected_hs_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        brc_0 = tbe.dsl.broadcast(3, shape)
        adds_0 = tbe.dsl.vadds(brc_0, 100)

        graph = Graph([brc_0])
        node = graph.get_nodes()[0]
        node.set_pvalue(util.new_pvalue_1("float16"))

        node.set_pvalue(util.new_pvalue_1("float16"))
        svalue0 = SettingValue(SettingValueType.NORMAL, "float16", value=0)
        node.add_svalue(svalue0)

        pvalue = scmp.deal_selected_hs(node, adds_0)

        assert_target = node.get_svalues()[0].target == [adds_0]
        assert_pvalue = pvalue.value == 0

        return all((assert_target, assert_pvalue))


@add_cust_test_func(ut_case)
def test_deal_selected_hs_const(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        brc_0 = tbe.dsl.broadcast(3, shape)
        adds_0 = tbe.dsl.vadds(brc_0, 100)

        pvalue = scmp.deal_selected_hs(tvm.const(100), adds_0)

        assert_pvalue = pvalue.value == 100

        return all((assert_pvalue,))


@add_cust_test_func(ut_case)
def test_deal_selected_hs_var(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        brc_0 = tbe.dsl.broadcast(3, shape)
        adds_0 = tbe.dsl.vadds(brc_0, 100)

        pvalue = scmp.deal_selected_hs(tvm.var("var0"), adds_0)

        assert_pvalue = pvalue.type == PaddingValueType.TENSOR

        return all((assert_pvalue,))


# cmp
@add_cust_test_func(ut_case)
def test_cmp_exact_exact_l(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    pvalue1 = util.new_pvalue_x(20, "int32")

    pgt = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.value == 0
    assert_ge = pge.value == 0
    assert_lt = plt.value == 1
    assert_le = ple.value == 1
    assert_eq = peq.value == 0
    assert_ne = pne.value == 1

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_exact_exact_g(_):
    pvalue0 = util.new_pvalue_x(20, "int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    pgt = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.value == 1
    assert_ge = pge.value == 1
    assert_lt = plt.value == 0
    assert_le = ple.value == 0
    assert_eq = peq.value == 0
    assert_ne = pne.value == 1

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_exact_exact_e(_):
    pvalue0 = util.new_pvalue_x(10, "int32")
    pvalue1 = util.new_pvalue_x(10, "int32")

    pgt = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_exact_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.value == 0
    assert_ge = pge.value == 1
    assert_lt = plt.value == 0
    assert_le = ple.value == 1
    assert_eq = peq.value == 1
    assert_ne = pne.value == 0

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_exact_tensor_max(_):
    pvalue0 = util.new_pvalue_max("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pgt = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.value == 1
    assert_lt = plt.value == 0
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_exact_tensor_min(_):
    pvalue0 = util.new_pvalue_min("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pgt = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.value == 0
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.value == 1
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_exact_tensor_x(_):
    pvalue0 = util.new_pvalue_x(100, "int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pgt = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_exact_tensor(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_exact_any(_):
    pvalue0 = util.new_pvalue_x(100, "int32")
    pvalue1 = util.new_pvalue_any("int32")

    pgt = scmp._cmp_exact_any(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_exact_any(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_exact_any(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_exact_any(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_exact_any(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_exact_any(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_tensor_exact_max(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_max("int32")

    pgt = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.value == 0
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.value == 1
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_tensor_exact_min(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_min("int32")

    pgt = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.value == 1
    assert_lt = plt.value == 0
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_tensor_exact_x(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_x(200, "int32")

    pgt = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_tensor_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_tensor_tensor(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pgt = scmp._cmp_tensor_tensor(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_tensor_tensor(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_tensor_tensor(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_tensor_tensor(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_tensor_tensor(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_tensor_tensor(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.TENSOR
    assert_ge = pge.type == PaddingValueType.TENSOR
    assert_lt = plt.type == PaddingValueType.TENSOR
    assert_le = ple.type == PaddingValueType.TENSOR
    assert_eq = peq.type == PaddingValueType.TENSOR
    assert_ne = pne.type == PaddingValueType.TENSOR

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_tensor_any(_):
    pvalue0 = util.new_pvalue_tensor("int32")
    pvalue1 = util.new_pvalue_any("int32")

    pgt = scmp._cmp_tensor_any(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_tensor_any(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_tensor_any(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_tensor_any(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_tensor_any(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_tensor_any(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_any_exact(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_min("int32")

    pgt = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.value == 1
    assert_lt = plt.value == 0
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_any_tensor(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_tensor("int32")

    pgt = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_any_exact(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_any_any_same_node(_):
    pvalue0 = util.new_pvalue_any("int32")

    pgt = scmp._cmp_any_any(pvalue0, pvalue0, "int32", CmpMode.GT)
    pge = scmp._cmp_any_any(pvalue0, pvalue0, "int32", CmpMode.GE)
    plt = scmp._cmp_any_any(pvalue0, pvalue0, "int32", CmpMode.LT)
    ple = scmp._cmp_any_any(pvalue0, pvalue0, "int32", CmpMode.LE)
    peq = scmp._cmp_any_any(pvalue0, pvalue0, "int32", CmpMode.EQ)
    pne = scmp._cmp_any_any(pvalue0, pvalue0, "int32", CmpMode.NE)

    assert_gt = pgt.value == 0
    assert_ge = pge.value == 1
    assert_lt = plt.value == 0
    assert_le = ple.value == 1
    assert_eq = peq.value == 1
    assert_ne = pne.value == 0

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


@add_cust_test_func(ut_case)
def test_cmp_any_any_alone_node(_):
    pvalue0 = util.new_pvalue_any("int32")
    pvalue1 = util.new_pvalue_any("int32")

    pgt = scmp._cmp_any_any(pvalue0, pvalue1, "int32", CmpMode.GT)
    pge = scmp._cmp_any_any(pvalue0, pvalue1, "int32", CmpMode.GE)
    plt = scmp._cmp_any_any(pvalue0, pvalue1, "int32", CmpMode.LT)
    ple = scmp._cmp_any_any(pvalue0, pvalue1, "int32", CmpMode.LE)
    peq = scmp._cmp_any_any(pvalue0, pvalue1, "int32", CmpMode.EQ)
    pne = scmp._cmp_any_any(pvalue0, pvalue1, "int32", CmpMode.NE)

    assert_gt = pgt.type == PaddingValueType.ANY
    assert_ge = pge.type == PaddingValueType.ANY
    assert_lt = plt.type == PaddingValueType.ANY
    assert_le = ple.type == PaddingValueType.ANY
    assert_eq = peq.type == PaddingValueType.ANY
    assert_ne = pne.type == PaddingValueType.ANY

    return all((assert_gt, assert_ge, assert_lt, assert_le, assert_eq, assert_ne))


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_cmp_any_any_alone_node":
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
