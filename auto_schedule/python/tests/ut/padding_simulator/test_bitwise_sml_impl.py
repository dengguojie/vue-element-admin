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
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_bitwise_sml_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


# bitwise and
@add_cust_test_func(ut_case)
def test_and_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        and_0 = tbe.dsl.vand(ph_0, ph_1)

        graph = Graph([and_0])
        node_ph_0, node_ph_1, node_and_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        node_and_0._simulator.adjust_calc()
        pvalue = node_and_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [and_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [and_0]
        assert_and_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_ph_1_target, assert_and_0_pvalue))


@add_cust_test_func(ut_case)
def test_and_all_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        and_0 = tbe.dsl.vand(ph_0, ph_1)

        graph = Graph([and_0])
        node_ph_0, node_ph_1, node_and_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        node_and_0._simulator.adjust_calc()
        pvalue = node_and_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [and_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [and_0]
        assert_and_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_ph_1_target, assert_and_0_pvalue))


@add_cust_test_func(ut_case)
def test_and_all_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        and_0 = tbe.dsl.vand(ph_0, ph_1)

        graph = Graph([and_0])
        node_ph_0, node_ph_1, node_and_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int32", value=1)
        node_ph_0.add_svalue(svalue0)
        svalue1 = SettingValue(SettingValueType.NORMAL, "int32", value=3)
        node_ph_1.add_svalue(svalue1)

        node_and_0._simulator.adjust_calc()
        pvalue = node_and_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [and_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [and_0]
        assert_and_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_ph_1_target, assert_and_0_pvalue))


@add_cust_test_func(ut_case)
def test_and_all_pvalue_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        and_0 = tbe.dsl.vand(ph_0, ph_1)

        graph = Graph([and_0])
        node_ph_0, node_ph_1, node_and_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_1("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        svalue1 = SettingValue(SettingValueType.NORMAL, "int32", value=3)
        node_ph_1.add_svalue(svalue1)

        node_and_0._simulator.adjust_calc()
        pvalue = node_and_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [and_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [and_0]
        assert_and_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_ph_1_target, assert_and_0_pvalue))


@add_cust_test_func(ut_case)
def test_and_all_svalue_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        and_0 = tbe.dsl.vand(ph_0, ph_1)

        graph = Graph([and_0])
        node_ph_0, node_ph_1, node_add_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_1("int32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int32", value=3)
        node_ph_0.add_svalue(svalue0)

        node_add_0._simulator.adjust_calc()
        pvalue = node_add_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [and_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [and_0]
        assert_and_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_ph_1_target, assert_and_0_pvalue))


# bitwise or
@add_cust_test_func(ut_case)
def test_or_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        or_0 = tbe.dsl.vor(ph_0, ph_1)

        graph = Graph([or_0])
        node_ph_0, node_ph_1, node_or_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        node_or_0._simulator.adjust_calc()
        pvalue = node_or_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [or_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [or_0]
        assert_or_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_ph_1_target, assert_or_0_pvalue))


@add_cust_test_func(ut_case)
def test_or_all_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        or_0 = tbe.dsl.vor(ph_0, ph_1)

        graph = Graph([or_0])
        node_ph_0, node_ph_1, node_or_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        node_or_0._simulator.adjust_calc()
        pvalue = node_or_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [or_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [or_0]
        assert_or_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_ph_1_target, assert_or_0_pvalue))


@add_cust_test_func(ut_case)
def test_or_all_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        or_0 = tbe.dsl.vor(ph_0, ph_1)

        graph = Graph([or_0])
        node_ph_0, node_ph_1, node_or_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int32", value=1)
        node_ph_0.add_svalue(svalue0)
        svalue1 = SettingValue(SettingValueType.NORMAL, "int32", value=3)
        node_ph_1.add_svalue(svalue1)

        node_or_0._simulator.adjust_calc()
        pvalue = node_or_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [or_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [or_0]
        assert_or_0_pvalue = pvalue.value == 3

        return all((assert_ph_0_target, assert_ph_1_target, assert_or_0_pvalue))


@add_cust_test_func(ut_case)
def test_or_all_pvalue_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        or_0 = tbe.dsl.vor(ph_0, ph_1)

        graph = Graph([or_0])
        node_ph_0, node_ph_1, node_or_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_1("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_0("int32"))

        svalue1 = SettingValue(SettingValueType.NORMAL, "int32", value=3)
        node_ph_1.add_svalue(svalue1)

        node_or_0._simulator.adjust_calc()
        pvalue = node_or_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [or_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [or_0]
        assert_or_0_pvalue = pvalue.value == 3

        return all((assert_ph_0_target, assert_ph_1_target, assert_or_0_pvalue))


@add_cust_test_func(ut_case)
def test_or_all_svalue_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="int32", name="ph_1")

        or_0 = tbe.dsl.vor(ph_0, ph_1)

        graph = Graph([or_0])
        node_ph_0, node_ph_1, node_or_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))
        node_ph_1.set_pvalue(util.new_pvalue_1("int32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int32", value=3)
        node_ph_0.add_svalue(svalue0)

        node_or_0._simulator.adjust_calc()
        pvalue = node_or_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [or_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [or_0]
        assert_or_0_pvalue = pvalue.value == 3

        return all((assert_ph_0_target, assert_ph_1_target, assert_or_0_pvalue))


# bitwise not
@add_cust_test_func(ut_case)
def test_not_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        not_0 = tbe.dsl.vnot(ph_0)

        graph = Graph([not_0])
        node_ph_0, node_not_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))

        node_not_0._simulator.adjust_calc()
        pvalue = node_not_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [not_0]
        assert_not_0_pvalue = pvalue.value == -1

        return all((assert_ph_0_target, assert_not_0_pvalue))


@add_cust_test_func(ut_case)
def test_not_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        not_0 = tbe.dsl.vnot(ph_0)

        graph = Graph([not_0])
        node_ph_0, node_not_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))

        node_not_0._simulator.adjust_calc()
        pvalue = node_not_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [not_0]
        assert_not_0_pvalue = pvalue.value == -1

        return all((assert_ph_0_target, assert_not_0_pvalue))


@add_cust_test_func(ut_case)
def test_not_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="int32", name="ph_0")
        not_0 = tbe.dsl.vnot(ph_0)

        graph = Graph([not_0])
        node_ph_0, node_not_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int32", value=1)
        node_ph_0.add_svalue(svalue0)

        node_not_0._simulator.adjust_calc()
        pvalue = node_not_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [not_0]
        assert_not_0_pvalue = pvalue.value == -2

        return all((assert_ph_0_target, assert_not_0_pvalue))


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_tensor_brc_adjust_calc_with_pad_and_exist_pad_brc_and_src_not_exist_pad":
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
