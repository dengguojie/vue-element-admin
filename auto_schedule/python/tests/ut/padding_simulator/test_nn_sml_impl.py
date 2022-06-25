# # -*- coding:utf-8 -*-
import warnings

import numpy as np
import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from sch_test_frame.ut.helper import padding_helper
from tbe import tvm
from tbe.dsl.base import operation
from tbe.dsl.base import var_api
from tbe.dsl.padding import util
from tbe.dsl.padding.graph import Graph
from tbe.dsl.padding.simulators.nn_sml import LeakyReluSimulator
from tbe.dsl.padding.simulators.nn_sml import ReluSimulator
from tbe.dsl.padding.value import PaddingValueType
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_nn_sml_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


# relu
@add_cust_test_func(ut_case)
def test_relu_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        relu_0 = tbe.dsl.vrelu(ph_0)

        graph = Graph([relu_0])
        node_ph_0, node_relu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float16"))

        node_relu_0._simulator.adjust_calc()
        pvalue = node_relu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [relu_0]
        assert_relu_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_relu_0_pvalue))


@add_cust_test_func(ut_case)
def test_relu_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        relu_0 = tbe.dsl.vrelu(ph_0)

        graph = Graph([relu_0])
        node_ph_0, node_relu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_x(-3.2, "float16"))

        node_relu_0._simulator.adjust_calc()
        pvalue = node_relu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [relu_0]
        assert_relu_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_relu_0_pvalue))


@add_cust_test_func(ut_case)
def test_relu_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        relu_0 = tbe.dsl.vrelu(ph_0)

        graph = Graph([relu_0])
        node_ph_0, node_relu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float16"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float16", value=1.3)
        node_ph_0.add_svalue(svalue0)

        node_relu_0._simulator.adjust_calc()
        pvalue = node_relu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [relu_0]
        assert_relu_0_pvalue = pvalue.value == np.float16(1.3)

        return all((assert_ph_0_target, assert_relu_0_pvalue))


@add_cust_test_func(ut_case)
def test_relu_tensor(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        relu_0 = tbe.dsl.vrelu(ph_0)

        graph = Graph([relu_0])
        node_ph_0, node_relu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_tensor("float16"))

        node_relu_0._simulator.adjust_calc()
        pvalue = node_relu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [relu_0]
        assert_relu_0_pvalue = pvalue.type == PaddingValueType.TENSOR

        return all((assert_ph_0_target, assert_relu_0_pvalue))


@add_cust_test_func(ut_case)
def test_relu_any(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        relu_0 = tbe.dsl.vrelu(ph_0)

        graph = Graph([relu_0])
        node_ph_0, node_relu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_any("float16"))

        node_relu_0._simulator.adjust_calc()
        pvalue = node_relu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [relu_0]
        assert_relu_0_pvalue = pvalue.type == PaddingValueType.ANY

        return all((assert_ph_0_target, assert_relu_0_pvalue))


@add_cust_test_func(ut_case)
def test_relu_type(_):
    return ReluSimulator.get_type() == "elewise_single_relu"


# leaky relu
@add_cust_test_func(ut_case)
def test_lrelu_no_padding(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend310P3"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        lrelu_0 = tbe.dsl.vlrelu(ph_0, 0.5)

        graph = Graph([lrelu_0])
        node_ph_0, node_lrelu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float16"))

        node_lrelu_0._simulator.adjust_calc()
        pvalue = node_lrelu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [lrelu_0]
        assert_lrelu_0_pvalue = pvalue.value == 0

        return all((assert_ph_0_target, assert_lrelu_0_pvalue))


@add_cust_test_func(ut_case)
def test_lrelu_pvalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend310P3"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        lrelu_0 = tbe.dsl.vlrelu(ph_0, 0.5)

        graph = Graph([lrelu_0])
        node_ph_0, node_lrelu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_x(-3.2, "float16"))

        node_lrelu_0._simulator.adjust_calc()
        pvalue = node_lrelu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [lrelu_0]
        assert_lrelu_0_pvalue = pvalue.value == np.float16(-3.2) * 0.5

        return all((assert_ph_0_target, assert_lrelu_0_pvalue))


@add_cust_test_func(ut_case)
def test_lrelu_svalue(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend310P3"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        lrelu_0 = tbe.dsl.vlrelu(ph_0, 0.5)

        graph = Graph([lrelu_0])
        node_ph_0, node_lrelu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_0("float16"))

        svalue0 = SettingValue(SettingValueType.NORMAL, "float16", value=1.3)
        node_ph_0.add_svalue(svalue0)

        node_lrelu_0._simulator.adjust_calc()
        pvalue = node_lrelu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_svalues()[0].target == [lrelu_0]
        assert_lrelu_0_pvalue = pvalue.value == np.float16(1.3)

        return all((assert_ph_0_target, assert_lrelu_0_pvalue))


@add_cust_test_func(ut_case)
def test_lrelu_tensor(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend310P3"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        lrelu_0 = tbe.dsl.vlrelu(ph_0, 0.5)

        graph = Graph([lrelu_0])
        node_ph_0, node_lrelu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_tensor("float16"))

        node_lrelu_0._simulator.adjust_calc()
        pvalue = node_lrelu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [lrelu_0]
        assert_lrelu_0_pvalue = pvalue.type == PaddingValueType.TENSOR

        return all((assert_ph_0_target, assert_lrelu_0_pvalue))


@add_cust_test_func(ut_case)
def test_lrelu_any(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend310P3"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)

        ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
        lrelu_0 = tbe.dsl.vlrelu(ph_0, 0.5)

        graph = Graph([lrelu_0])
        node_ph_0, node_lrelu_0 = graph.get_nodes()
        node_ph_0.set_pvalue(util.new_pvalue_any("float16"))

        node_lrelu_0._simulator.adjust_calc()
        pvalue = node_lrelu_0.get_pvalue()

        assert_ph_0_target = node_ph_0.get_pvalue().target == [lrelu_0]
        assert_lrelu_0_pvalue = pvalue.type == PaddingValueType.ANY

        return all((assert_ph_0_target, assert_lrelu_0_pvalue))


@add_cust_test_func(ut_case)
def test_lrelu_type(_):
    return LeakyReluSimulator.get_type() == "elewise_single_lrelu"


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_lrelu_pvalue":
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
