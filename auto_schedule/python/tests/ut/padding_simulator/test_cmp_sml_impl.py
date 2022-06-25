# # -*- coding:utf-8 -*-
import warnings

import tbe
import tbe.dsl.padding.simulators.cmpsel.scmp as scmp
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from sch_test_frame.ut.helper import padding_helper
from tbe import tvm
from tbe.dsl.base import operation
from tbe.dsl.base import var_api
from tbe.dsl.padding import util
from tbe.dsl.padding.graph import Graph
from tbe.dsl.padding.simulators.cmpsel.cmp_sml import CmpEqSimulator
from tbe.dsl.padding.simulators.cmpsel.cmp_sml import CmpGeSimulator
from tbe.dsl.padding.simulators.cmpsel.cmp_sml import CmpGtSimulator
from tbe.dsl.padding.simulators.cmpsel.cmp_sml import CmpLeSimulator
from tbe.dsl.padding.simulators.cmpsel.cmp_sml import CmpLtSimulator
from tbe.dsl.padding.simulators.cmpsel.cmp_sml import CmpNeSimulator
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_cmp_sml_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


# base
@add_cust_test_func(ut_case)
def test_eq_tensor_scalar_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        cmp_0 = tbe.dsl.vcmp(ph_0, 0, "eq", "bit")

        graph = Graph([cmp_0])
        node_ph_0, node_cmp_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float16"))

        node_cmp_0._simulator.adjust_calc()
        pvalue = node_cmp_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmp_0]
        assert_cmp_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_cmp_0_pvalue))


@add_cust_test_func(ut_case)
def test_ne_tensor_tensor_all_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        cmp_0 = tbe.dsl.vcmp(ph_0, ph_1, "ne", "bit")

        graph = Graph([cmp_0])
        node_ph_0, node_ph_1, node_cmp_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_0("float32"))
        node_ph_1.set_pvalue(util.new_pvalue_1("float32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float32", value=0)
        node_ph_0.add_svalue(svalue0)
        svalue1 = SettingValue(SettingValueType.NORMAL, "float32", value=0)
        node_ph_1.add_svalue(svalue1)

        node_cmp_0._simulator.adjust_calc()
        pvalue = node_cmp_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [cmp_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [cmp_0]
        assert_cmp_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_ph_1_target, assert_cmp_0_pvalue))


@add_cust_test_func(ut_case)
def test_lt_tensor_scalar_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        cmp_0 = tbe.dsl.vcmp(ph_0, 0, "lt", "bit")

        graph = Graph([cmp_0])
        node_ph_0, node_cmp_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float16"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float16", value=0)
        node_ph_0.add_svalue(svalue0)

        node_cmp_0._simulator.adjust_calc()
        pvalue = node_cmp_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [cmp_0]
        assert_cmp_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_cmp_0_pvalue))


# value
@add_cust_test_func(ut_case)
def test_le_tensor_scalar_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        cmp_0 = tbe.dsl.vcmp(ph_0, 0, "le", "bit")

        graph = Graph([cmp_0])
        node_ph_0, node_cmp_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float16"))

        node_cmp_0._simulator.adjust_calc()
        pvalue = node_cmp_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmp_0]
        assert_cmp_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_cmp_0_pvalue))


@add_cust_test_func(ut_case)
def test_gt_tensor_tensor_svalue_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        cmp_0 = tbe.dsl.vcmp(ph_0, ph_1, "gt", "bit")

        graph = Graph([cmp_0])
        node_ph_0, node_ph_1, node_cmp_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_0("float32"))
        node_ph_1.set_pvalue(util.new_pvalue_1("float32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float32", value=2)
        node_ph_0.add_svalue(svalue0)

        node_cmp_0._simulator.adjust_calc()
        pvalue = node_cmp_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [cmp_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [cmp_0]
        assert_cmp_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_ph_1_target, assert_cmp_0_pvalue))


@add_cust_test_func(ut_case)
def test_ge_tensor_tensor_all_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        cmp_0 = tbe.dsl.vcmp(ph_0, ph_1, "ge", "bit")

        graph = Graph([cmp_0])
        node_ph_0, node_ph_1, node_cmp_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_0("float32"))
        node_ph_1.set_pvalue(util.new_pvalue_1("float32"))

        node_cmp_0._simulator.adjust_calc()
        pvalue = node_cmp_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmp_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [cmp_0]
        assert_cmp_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_ph_1_target, assert_cmp_0_pvalue))


@add_cust_test_func(ut_case)
def test_eq_tensor_tensor_pvalue_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        cmp_0 = tbe.dsl.vcmp(ph_0, ph_1, "eq", "bit")

        graph = Graph([cmp_0])
        node_ph_0, node_ph_1, node_cmp_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_0("float32"))
        node_ph_1.set_pvalue(util.new_pvalue_1("float32"))

        svalue1 = SettingValue(SettingValueType.NORMAL, "float32", value=0)
        node_ph_1.add_svalue(svalue1)

        node_cmp_0._simulator.adjust_calc()
        pvalue = node_cmp_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmp_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [cmp_0]
        assert_cmp_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_ph_1_target, assert_cmp_0_pvalue))


# cmp mode
@add_cust_test_func(ut_case)
def test_gt_mode(_):
    return CmpGtSimulator.get_cmp_mode() == scmp.CmpMode.GT


@add_cust_test_func(ut_case)
def test_ge_mode(_):
    return CmpGeSimulator.get_cmp_mode() == scmp.CmpMode.GE


@add_cust_test_func(ut_case)
def test_lt_mode(_):
    return CmpLtSimulator.get_cmp_mode() == scmp.CmpMode.LT


@add_cust_test_func(ut_case)
def test_le_mode(_):
    return CmpLeSimulator.get_cmp_mode() == scmp.CmpMode.LE


@add_cust_test_func(ut_case)
def test_eq_mode(_):
    return CmpEqSimulator.get_cmp_mode() == scmp.CmpMode.EQ


@add_cust_test_func(ut_case)
def test_ne_mode(_):
    return CmpNeSimulator.get_cmp_mode() == scmp.CmpMode.NE


# type
@add_cust_test_func(ut_case)
def test_gt_type(_):
    return CmpGtSimulator.get_type() == "elewise_binary_vcmpv_gt"


@add_cust_test_func(ut_case)
def test_ge_type(_):
    return CmpGeSimulator.get_type() == "elewise_binary_vcmpv_ge"


@add_cust_test_func(ut_case)
def test_lt_type(_):
    return CmpLtSimulator.get_type() == "elewise_binary_vcmpv_lt"


@add_cust_test_func(ut_case)
def test_le_type(_):
    return CmpLeSimulator.get_type() == "elewise_binary_vcmpv_le"


@add_cust_test_func(ut_case)
def test_eq_type(_):
    return CmpEqSimulator.get_type() == "elewise_binary_vcmpv_eq"


@add_cust_test_func(ut_case)
def test_ne_type(_):
    return CmpNeSimulator.get_type() == "elewise_binary_vcmpv_ne"


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
