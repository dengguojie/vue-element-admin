# # -*- coding:utf-8 -*-
import warnings

import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from sch_test_frame.ut.helper import padding_helper
from tbe import tvm
from tbe.dsl.base import operation
from tbe.dsl.base import var_api
from tbe.dsl.padding import smath
from tbe.dsl.padding import util
from tbe.dsl.padding.graph import Graph
from tbe.dsl.padding.simulators.cast_sml import CastSimulator
from tbe.dsl.padding.simulators.cast_sml import CeilSimulator
from tbe.dsl.padding.simulators.cast_sml import FloorSimulator
from tbe.dsl.padding.simulators.cast_sml import RounddSimulator
from tbe.dsl.padding.simulators.cast_sml import RoundSimulator
from tbe.dsl.padding.simulators.cast_sml import TruncSimulator
from tbe.dsl.padding.value import PaddingValueType
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_cast_sml_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


# base
@add_cust_test_func(ut_case)
def test_cast_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        cast_0 = tbe.dsl.cast_to(ph_0, "int32")

        graph = Graph([cast_0])
        node_ph_0, node_cast_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float32"))

        node_cast_0._simulator.adjust_calc()
        pvalue = node_cast_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cast_0]
        assert_cast_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_cast_0_pvalue))


@add_cust_test_func(ut_case)
def test_cast_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        cast_0 = tbe.dsl.cast_to(ph_0, "int32")

        graph = Graph([cast_0])
        node_ph_0, node_cast_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_1("float32"))

        node_cast_0._simulator.adjust_calc()
        pvalue = node_cast_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cast_0]
        assert_cast_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_cast_0_pvalue))


@add_cust_test_func(ut_case)
def test_cast_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        cast_0 = tbe.dsl.cast_to(ph_0, "int32")

        graph = Graph([cast_0])
        node_ph_0, node_cast_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float32"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float32", value=1)
        node_ph_0.add_svalue(svalue0)

        node_cast_0._simulator.adjust_calc()
        pvalue = node_cast_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [cast_0]
        assert_cast_0_pvalue = pvalue.value == 1

        return all((assert_ph_0_target, assert_cast_0_pvalue))


# value
@add_cust_test_func(ut_case)
def test_cast_tensor(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        cast_0 = tbe.dsl.cast_to(ph_0, "int32")

        graph = Graph([cast_0])
        node_ph_0, node_cast_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_tensor("float32"))

        node_cast_0._simulator.adjust_calc()
        pvalue = node_cast_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cast_0]
        assert_cast_0_pvalue = pvalue.type == PaddingValueType.TENSOR

        return all((assert_ph_0_target, assert_cast_0_pvalue))


@add_cust_test_func(ut_case)
def test_cast_any(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float32", name="ph_0")
        cast_0 = tbe.dsl.cast_to(ph_0, "int32")

        graph = Graph([cast_0])
        node_ph_0, node_cast_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_any("float32"))

        node_cast_0._simulator.adjust_calc()
        pvalue = node_cast_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [cast_0]
        assert_cast_0_pvalue = pvalue.type == PaddingValueType.ANY

        return all((assert_ph_0_target, assert_cast_0_pvalue))


# func
@add_cust_test_func(ut_case)
def test_cast_func(_):
    return CastSimulator._get_cast_func() == smath.cast_


@add_cust_test_func(ut_case)
def test_ceil_func(_):
    return CeilSimulator._get_cast_func() == smath.ceil_


@add_cust_test_func(ut_case)
def test_floor_func(_):
    return FloorSimulator._get_cast_func() == smath.floor_


@add_cust_test_func(ut_case)
def test_trunc_func(_):
    return TruncSimulator._get_cast_func() == smath.trunc_


@add_cust_test_func(ut_case)
def test_round_func(_):
    return RoundSimulator._get_cast_func() == smath.round_


@add_cust_test_func(ut_case)
def test_round_d_func(_):
    return RounddSimulator._get_cast_func() == smath.round_d_


# type
@add_cust_test_func(ut_case)
def test_cast_type(_):
    return CastSimulator.get_type() == "elewise_single_cast"


@add_cust_test_func(ut_case)
def test_ceil_type(_):
    return CeilSimulator.get_type() == "elewise_single_ceil"


@add_cust_test_func(ut_case)
def test_floor_type(_):
    return FloorSimulator.get_type() == "elewise_single_floor"


@add_cust_test_func(ut_case)
def test_trunc_type(_):
    return TruncSimulator.get_type() == "elewise_single_trunc"


@add_cust_test_func(ut_case)
def test_round_type(_):
    return RoundSimulator.get_type() == "elewise_single_round"


@add_cust_test_func(ut_case)
def test_round_d_type(_):
    return RounddSimulator.get_type() == "elewise_single_round_d"


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
