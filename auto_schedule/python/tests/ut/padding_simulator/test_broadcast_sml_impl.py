# # -*- coding:utf-8 -*-
import warnings

import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from sch_test_frame.ut.helper import padding_helper
from tbe import tvm
from tbe.dsl.base import operation, var_api
from tbe.dsl.base.padding import util
from tbe.dsl.base.padding.graph import Graph
from tbe.dsl.base.padding.value import PaddingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_padding_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


@add_cust_test_func(ut_case)
def test_scalar_brc_adjust_calc_with_pad(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        brc_0 = tbe.dsl.broadcast(3, shape)

        graph = Graph([brc_0])
        node = graph.get_nodes()[0]
        node._simulator.adjust_calc()
        pvalue = node.get_pvalue()
        return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_scalar_brc_adjust_calc_without_pad(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        shape = (4, 3, 6, 8, 16)
        brc_0 = tbe.dsl.broadcast(3, shape)

        graph = Graph([brc_0])
        node = graph.get_nodes()[0]
        node._simulator.adjust_calc()
        pvalue = node.get_pvalue()
        return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_tensor_brc_adjust_calc_without_pad(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        shape0 = (4, 3, 1, 1, 16)
        shape = (4, 3, 6, 8, 16)
        ph_0 = tvm.placeholder(shape0, "float16", name="ph_0")
        brc_0 = tbe.dsl.broadcast(ph_0, shape)

        graph = Graph([brc_0])
        graph.get_nodes()[0].set_pvalue(util.new_pvalue_tensor("float16"))
        node = graph.get_nodes()[1]
        node._simulator.adjust_calc()
        pvalue = node.get_pvalue()
        return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_tensor_brc_adjust_calc_with_pad_and_exist_pad_brc_and_src_exist_pad(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x1_c = padding_helper.const_c(42)
        var_api.set_attr(x1_c1, "original", x1_c)
        var_api.set_attr(x1_c0, "original", x1_c)
        shape1 = (x1_n, x1_c1, x1_h, x1_w, x1_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        brc_0 = tbe.dsl.broadcast(ph_0, shape1)

        graph = Graph([brc_0])
        graph.get_nodes()[0].set_pvalue(util.new_pvalue_0("float16"))
        node = graph.get_nodes()[1]
        node0 = graph.get_nodes()[0]
        node._simulator.adjust_calc()
        pvalue = node.get_pvalue()
        svalues = node0.get_svalues()

        return pvalue.type == PaddingValueType.TENSOR and len(svalues) == 1


@add_cust_test_func(ut_case)
def test_tensor_brc_adjust_calc_with_pad_and_exist_pad_brc_and_src_not_exist_pad(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 1), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x1_c = padding_helper.const_c(42)
        var_api.set_attr(x1_c1, "original", x1_c)
        var_api.set_attr(x1_c0, "original", x1_c)
        shape1 = (x1_n, x1_c1, x1_h, x1_w, x1_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        brc_0 = tbe.dsl.broadcast(ph_0, shape1)

        graph = Graph([brc_0])
        graph.get_nodes()[0].set_pvalue(util.new_pvalue_tensor("float16"))
        node = graph.get_nodes()[1]
        node0 = graph.get_nodes()[0]
        node._simulator.adjust_calc()
        pvalue = node.get_pvalue()
        svalues = node0.get_svalues()

        return pvalue.type == PaddingValueType.TENSOR and len(svalues) == 0


@add_cust_test_func(ut_case)
def test_tensor_brc_adjust_calc_with_pad_and_not_exist_pad_brc(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 3, 1, 1, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(42)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x1_c = padding_helper.const_c(42)
        var_api.set_attr(x1_c1, "original", x1_c)
        var_api.set_attr(x1_c0, "original", x1_c)
        shape1 = (x1_n, x1_c1, x1_h, x1_w, x1_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        brc_0 = tbe.dsl.broadcast(ph_0, shape1)

        graph = Graph([brc_0])
        node0 = graph.get_nodes()[0]
        node0.set_pvalue(util.new_pvalue_tensor(fp16))
        node = graph.get_nodes()[1]
        node._simulator.adjust_calc()
        pvalue = node.get_pvalue()
        pvalue0 = node0.get_pvalue()
        svalues = node0.get_svalues()

        return pvalue.type == PaddingValueType.TENSOR and len(svalues) == 0


@add_cust_test_func(ut_case)
def test_tensor_brc_adjust(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x1_c = padding_helper.const_c(42)
        var_api.set_attr(x1_c1, "original", x1_c)
        var_api.set_attr(x1_c0, "original", x1_c)
        shape1 = (x1_n, x1_c1, x1_h, x1_w, x1_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        brc_0 = tbe.dsl.broadcast(ph_0, shape1)

        graph = Graph([brc_0])
        node0 = graph.get_nodes()[0]
        node0.set_pvalue(util.new_pvalue_0(fp16))
        node = graph.get_nodes()[1]
        node._simulator.adjust_calc()
        svalues = node0.get_svalues()

        return len(svalues) == 1


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
