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
from tbe.dsl.padding.simulators.cmpsel.cmpsel_sml import CmpselEqSimulator
from tbe.dsl.padding.simulators.cmpsel.cmpsel_sml import CmpselGeSimulator
from tbe.dsl.padding.simulators.cmpsel.cmpsel_sml import CmpselGtSimulator
from tbe.dsl.padding.simulators.cmpsel.cmpsel_sml import CmpselLeSimulator
from tbe.dsl.padding.simulators.cmpsel.cmpsel_sml import CmpselLtSimulator
from tbe.dsl.padding.simulators.cmpsel.cmpsel_sml import CmpselNeSimulator
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_cmpsel_sml_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


# base
@add_cust_test_func(ut_case)
def test_eq_tensor_scalar_scalar_scalar_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, 0, "eq", 1, 2)

        graph = Graph([cmpsel_0])
        node_ph_0, node_cmpsel_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float32"))

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmpsel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_ne_tensor_scalar_scalar_tensor_sx_xs(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        srhs = tvm.placeholder(shape, dtype="float32", name="srhs")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, 0, "ne", 1, srhs)

        graph = Graph([cmpsel_0])
        node_ph_0, node_srhs, node_cmpsel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("float32"))
        node_srhs.set_pvalue(util.new_pvalue_0("float32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float32", value=0)
        node_ph_0.add_svalue(svalue0)

        svalue3 = SettingValue(SettingValueType.NORMAL, "float32", value=4)
        node_srhs.add_svalue(svalue3)

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [cmpsel_0]
        assert_srhs_target = node_srhs.get_svalues()[0].target == [cmpsel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 4

        return all((assert_ph_0_target, assert_srhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_lt_tensor_scalar_tensor_scalar_px_px(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        slhs = tvm.placeholder(shape, dtype="float32", name="srhs")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, 0, "lt", slhs, 1)

        graph = Graph([cmpsel_0])
        node_ph_0, node_slhs, node_cmpsel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("float32"))
        node_slhs.set_pvalue(util.new_pvalue_0("float32"))

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmpsel_0]
        assert_slhs_target = node_slhs.get_pvalue().target == []
        assert_cmpsel_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_slhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_le_tensor_scalar_tensor_tensor_px_pp(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        slhs = tvm.placeholder(shape, dtype="float32", name="srhs")
        srhs = tvm.placeholder(shape, dtype="float32", name="srhs")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, 1, "le", slhs, srhs)

        graph = Graph([cmpsel_0])
        node_ph_0, node_slhs, node_srhs, node_cmpsel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_x(1, "float32"))
        node_slhs.set_pvalue(util.new_pvalue_x(2, "float32"))
        node_srhs.set_pvalue(util.new_pvalue_x(3, "float32"))

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmpsel_0]
        assert_slhs_target = node_slhs.get_pvalue().target == [cmpsel_0]
        assert_srhs_target = node_srhs.get_pvalue().target == []
        assert_cmpsel_0_pvalue = pvalue.value == 2

        return all((assert_ph_0_target, assert_slhs_target, assert_srhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_gt_tensor_tensor_scalar_scalar_ss_xx(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, ph_1, "gt", 10, 20)

        graph = Graph([cmpsel_0])
        node_ph_0, node_ph_1, node_cmpsel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_x(1, "float32"))
        node_ph_1.set_pvalue(util.new_pvalue_x(2, "float32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float32", value=2)
        node_ph_0.add_svalue(svalue0)

        svalue1 = SettingValue(SettingValueType.NORMAL, "float32", value=1)
        node_ph_1.add_svalue(svalue1)

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [cmpsel_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [cmpsel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 10

        return all((assert_ph_0_target, assert_ph_1_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_ge_tensor_tensor_scalar_tensor_pp_xp(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        srhs = tvm.placeholder(shape, dtype="float32", name="srhs")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, ph_1, "ge", 10, srhs)

        graph = Graph([cmpsel_0])
        node_ph_0, node_ph_1, node_srhs, node_cmpsel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_x(1, "float32"))
        node_ph_1.set_pvalue(util.new_pvalue_x(2, "float32"))
        node_srhs.set_pvalue(util.new_pvalue_x(3, "float32"))

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmpsel_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [cmpsel_0]
        assert_srhs_target = node_srhs.get_pvalue().target == [cmpsel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 3

        return all((assert_ph_0_target, assert_ph_1_target, assert_srhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_eq_tensor_tensor_tensor_scalar_sp_sx(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        slhs = tvm.placeholder(shape, dtype="float32", name="slhs")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, ph_1, "eq", slhs, 10)

        graph = Graph([cmpsel_0])
        node_ph_0, node_ph_1, node_slhs, node_cmpsel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_x(1, "float32"))
        node_ph_1.set_pvalue(util.new_pvalue_x(2, "float32"))
        node_slhs.set_pvalue(util.new_pvalue_x(3, "float32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float32", value=2)
        node_ph_0.add_svalue(svalue0)

        svalue2 = SettingValue(SettingValueType.NORMAL, "float32", value=3)
        node_slhs.add_svalue(svalue2)

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [cmpsel_0]
        assert_ph_1_target = node_ph_1.get_pvalue().target == [cmpsel_0]
        assert_slhs_target = node_slhs.get_svalues()[0].target == [cmpsel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 3

        return all((assert_ph_0_target, assert_ph_1_target, assert_slhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_ne_tensor_tensor_tensor_tensor_ps_sp(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
        slhs = tvm.placeholder(shape, dtype="float32", name="slhs")
        srhs = tvm.placeholder(shape, dtype="float32", name="srhs")
        cmpsel_0 = tbe.dsl.vcmpsel(ph_0, ph_1, "ne", slhs, srhs)

        graph = Graph([cmpsel_0])
        node_ph_0, node_ph_1, node_slhs, node_srhs,node_cmpsel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_x(1, "float32"))
        node_ph_1.set_pvalue(util.new_pvalue_x(2, "float32"))
        node_slhs.set_pvalue(util.new_pvalue_x(3, "float32"))
        node_srhs.set_pvalue(util.new_pvalue_x(4, "float32"))

        svalue1 = SettingValue(SettingValueType.NORMAL, "float32", value=1)
        node_ph_1.add_svalue(svalue1)

        svalue2 = SettingValue(SettingValueType.NORMAL, "float32", value=6)
        node_slhs.add_svalue(svalue2)

        node_cmpsel_0._simulator.adjust_calc()
        pvalue = node_cmpsel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cmpsel_0]
        assert_ph_1_target = node_ph_1.get_svalues()[0].target == [cmpsel_0]
        assert_slhs_target = node_slhs.get_svalues()[0].target == []
        assert_srhs_target = node_srhs.get_pvalue().target == [cmpsel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 4

        return all((assert_ph_0_target, assert_ph_1_target, assert_slhs_target, 
                    assert_srhs_target, assert_cmpsel_0_pvalue))


# cmpsel mode
@add_cust_test_func(ut_case)
def test_gt_mode(_):
    return CmpselGtSimulator.get_cmp_mode() == scmp.CmpMode.GT


@add_cust_test_func(ut_case)
def test_ge_mode(_):
    return CmpselGeSimulator.get_cmp_mode() == scmp.CmpMode.GE


@add_cust_test_func(ut_case)
def test_lt_mode(_):
    return CmpselLtSimulator.get_cmp_mode() == scmp.CmpMode.LT


@add_cust_test_func(ut_case)
def test_le_mode(_):
    return CmpselLeSimulator.get_cmp_mode() == scmp.CmpMode.LE


@add_cust_test_func(ut_case)
def test_eq_mode(_):
    return CmpselEqSimulator.get_cmp_mode() == scmp.CmpMode.EQ


@add_cust_test_func(ut_case)
def test_ne_mode(_):
    return CmpselNeSimulator.get_cmp_mode() == scmp.CmpMode.NE


# type
@add_cust_test_func(ut_case)
def test_gt_type(_):
    return CmpselGtSimulator.get_type() == "elewise_binary_cmpsel_gt"


@add_cust_test_func(ut_case)
def test_ge_type(_):
    return CmpselGeSimulator.get_type() == "elewise_binary_cmpsel_ge"


@add_cust_test_func(ut_case)
def test_lt_type(_):
    return CmpselLtSimulator.get_type() == "elewise_binary_cmpsel_lt"


@add_cust_test_func(ut_case)
def test_le_type(_):
    return CmpselLeSimulator.get_type() == "elewise_binary_cmpsel_le"


@add_cust_test_func(ut_case)
def test_eq_type(_):
    return CmpselEqSimulator.get_type() == "elewise_binary_cmpsel_eq"


@add_cust_test_func(ut_case)
def test_ne_type(_):
    return CmpselNeSimulator.get_type() == "elewise_binary_cmpsel_ne"


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
