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
from tbe.dsl.padding.simulators.cmpsel.sel_sml import SelSimulator
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_sel_sml_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


# base
@add_cust_test_func(ut_case)
def test_tensor_scalar_scalar_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="uint1", name="ph_0")
        sel_0 = tbe.dsl.vsel(ph_0, 1, 2)

        graph = Graph([sel_0])
        node_ph_0, node_sel_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("int8"))

        node_sel_0._simulator.adjust_calc()
        pvalue = node_sel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [sel_0]
        assert_sel_0_pvalue = pvalue.value == 2

        return all((assert_ph_0_target, assert_sel_0_pvalue))


@add_cust_test_func(ut_case)
def test_tensor_scalar_tensor_p_xp(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="uint1", name="ph_0")
        srhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        sel_0 = tbe.dsl.vsel(ph_0, 1, srhs)

        graph = Graph([sel_0])
        node_ph_0, node_srhs, node_sel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("int8"))
        node_srhs.set_pvalue(util.new_pvalue_0("float16"))

        node_sel_0._simulator.adjust_calc()
        pvalue = node_sel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [sel_0]
        assert_srhs_target = node_srhs.get_pvalue().target == []
        assert_cmpsel_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_srhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_tensor_tensor_scalar_s_px(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="uint1", name="ph_0")
        slhs = tvm.placeholder(shape, dtype="float16", name="slhs")
        sel_0 = tbe.dsl.vsel(ph_0, slhs, 10)

        graph = Graph([sel_0])
        node_ph_0, node_slhs, node_sel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("int8"))
        node_slhs.set_pvalue(util.new_pvalue_0("float16"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int8", value=0)
        node_ph_0.add_svalue(svalue0)

        node_sel_0._simulator.adjust_calc()
        pvalue = node_sel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [sel_0]
        assert_slhs_target = node_slhs.get_pvalue().target == []
        assert_cmpsel_0_pvalue = pvalue.value == 10

        return all((assert_ph_0_target, assert_slhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_tensor_tensor_tensor_p_pp(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="uint1", name="ph_0")
        slhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        srhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        sel_0 = tbe.dsl.vsel(ph_0, slhs, srhs)

        graph = Graph([sel_0])
        node_ph_0, node_slhs, node_srhs, node_sel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("int8"))
        node_slhs.set_pvalue(util.new_pvalue_x(20, "float16"))
        node_srhs.set_pvalue(util.new_pvalue_x(30, "float16"))

        node_sel_0._simulator.adjust_calc()
        pvalue = node_sel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [sel_0]
        assert_slhs_target = node_slhs.get_pvalue().target == [sel_0]
        assert_srhs_target = node_srhs.get_pvalue().target == []
        assert_cmpsel_0_pvalue = pvalue.value == 20

        return all((assert_ph_0_target, assert_slhs_target, assert_srhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_tensor_tensor_tensor_p_ss(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="uint1", name="ph_0")
        slhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        srhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        sel_0 = tbe.dsl.vsel(ph_0, slhs, srhs)

        graph = Graph([sel_0])
        node_ph_0, node_slhs, node_srhs, node_sel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("int8"))
        node_slhs.set_pvalue(util.new_pvalue_x(20, "float16"))
        node_srhs.set_pvalue(util.new_pvalue_x(30, "float16"))

        svalue1 = SettingValue(SettingValueType.NORMAL, "float16", value=2)
        node_slhs.add_svalue(svalue1)

        svalue2 = SettingValue(SettingValueType.NORMAL, "float16", value=3)
        node_srhs.add_svalue(svalue2)

        node_sel_0._simulator.adjust_calc()
        pvalue = node_sel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [sel_0]
        assert_slhs_target = node_slhs.get_svalues()[0].target == [sel_0]
        assert_srhs_target = node_srhs.get_svalues()[0].target == []
        assert_cmpsel_0_pvalue = pvalue.value == 2

        return all((assert_ph_0_target, assert_slhs_target, assert_srhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_tensor_tensor_tensor_s_sp(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="uint1", name="ph_0")
        slhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        srhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        sel_0 = tbe.dsl.vsel(ph_0, slhs, srhs)

        graph = Graph([sel_0])
        node_ph_0, node_slhs, node_srhs, node_sel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("int8"))
        node_slhs.set_pvalue(util.new_pvalue_x(20, "float16"))
        node_srhs.set_pvalue(util.new_pvalue_x(30, "float16"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int8", value=0)
        node_ph_0.add_svalue(svalue0)

        svalue1 = SettingValue(SettingValueType.NORMAL, "float16", value=2)
        node_slhs.add_svalue(svalue1)

        node_sel_0._simulator.adjust_calc()
        pvalue = node_sel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [sel_0]
        assert_slhs_target = node_slhs.get_svalues()[0].target == []
        assert_srhs_target = node_srhs.get_pvalue().target == [sel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 30

        return all((assert_ph_0_target, assert_slhs_target, assert_srhs_target, assert_cmpsel_0_pvalue))


@add_cust_test_func(ut_case)
def test_tensor_tensor_tensor_s_ps(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="uint1", name="ph_0")
        slhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        srhs = tvm.placeholder(shape, dtype="float16", name="srhs")
        sel_0 = tbe.dsl.vsel(ph_0, slhs, srhs)

        graph = Graph([sel_0])
        node_ph_0, node_slhs, node_srhs, node_sel_0 = graph.get_nodes()

        node_ph_0.set_pvalue(util.new_pvalue_1("int8"))
        node_slhs.set_pvalue(util.new_pvalue_x(20, "float16"))
        node_srhs.set_pvalue(util.new_pvalue_x(30, "float16"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "int8", value=0)
        node_ph_0.add_svalue(svalue0)

        svalue2 = SettingValue(SettingValueType.NORMAL, "float16", value=2)
        node_srhs.add_svalue(svalue2)

        node_sel_0._simulator.adjust_calc()
        pvalue = node_sel_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [sel_0]
        assert_slhs_target = node_slhs.get_pvalue().target == []
        assert_srhs_target = node_srhs.get_svalues()[0].target == [sel_0]
        assert_cmpsel_0_pvalue = pvalue.value == 2

        return all((assert_ph_0_target, assert_slhs_target, assert_srhs_target, assert_cmpsel_0_pvalue))


# type
@add_cust_test_func(ut_case)
def test_get_type(_):
    return SelSimulator.get_type() == "elewise_multiple_sel"


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
