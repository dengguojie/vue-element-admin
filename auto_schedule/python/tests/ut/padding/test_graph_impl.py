# # -*- coding:utf-8 -*-
import warnings

import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from sch_test_frame.ut.helper import padding_helper
from tbe import tvm
from tbe.dsl.base import operation
from tbe.dsl.base.padding.graph import Graph

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_graph_impl")


@add_cust_test_func(ut_case)
def test_graph_with_elewise(_):
    with operation.dynamic(), padding_helper.soc_910():
        shape1 = (3, 1, 16)
        fp16 = "float16"

        ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_1")
        const_num_neg_one = tvm.const(-1, dtype=fp16)
        const_num_one = tvm.const(1, dtype=fp16)
        tmp_negative = tbe.dsl.vmuls(ph_1, const_num_neg_one)
        tmp_exp = tbe.dsl.vexp(tmp_negative)
        tmp_sum = tbe.dsl.vadds(tmp_exp, const_num_one)
        tmp_rec = tbe.dsl.vrec(tmp_sum)

    graph = Graph([tmp_rec])
    nodes = graph.get_nodes()
    tensors = [n.get_tensor() for n in nodes]
    expected_tensors = [ph_1, tmp_negative, tmp_exp, tmp_sum, tmp_rec]
    assert_nodes = tensors == expected_tensors

    node_exp = graph.get_node(tmp_exp)
    t_exp = node_exp.get_tensor()
    assert_node = t_exp == tmp_exp

    assert_out_rec = graph.is_out(graph.get_node(tmp_rec)) is True
    assert_out_sum = graph.is_out(graph.get_node(tmp_sum)) is False

    return assert_nodes and assert_node and assert_out_rec and assert_out_sum


@add_cust_test_func(ut_case)
def test_graph_with_broadcast(_):
    with operation.dynamic(), padding_helper.soc_910():
        shape0 = (3, 1, 16)
        shape1 = (3, 8, 16)
        shape = (3, 8, 16)
        fp16 = "float16"

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_0")
        ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_1")

        fp32_0 = tbe.dsl.cast_to(ph_0, "float32")
        fp32_1 = tbe.dsl.cast_to(ph_1, "float32")

        brc_0 = tbe.dsl.broadcast(fp32_0, shape)
        brc_1 = tbe.dsl.broadcast(fp32_1, shape)

        div_01 = tbe.dsl.vdiv(brc_0, brc_1)
        floor_01 = tbe.dsl.floor(div_01)
        cast_01 = tbe.dsl.cast_to(floor_01, "float32")
        mul_01 = tbe.dsl.vmul(cast_01, brc_1)
        sub_01 = tbe.dsl.vsub(brc_0, mul_01)

    graph = Graph([div_01, sub_01])
    nodes = graph.get_nodes()
    tensors = [n.get_tensor() for n in nodes]
    expected_tensors = [ph_0, fp32_0, brc_0, ph_1, fp32_1, div_01, floor_01, cast_01, mul_01, sub_01]
    assert_nodes = tensors == expected_tensors

    node_exp = graph.get_node(div_01)
    t_exp = node_exp.get_tensor()
    assert_node = t_exp == div_01

    assert_out_div = graph.is_out(graph.get_node(div_01)) is True
    assert_out_sub = graph.is_out(graph.get_node(sub_01)) is True

    return assert_nodes and assert_node and assert_out_div and assert_out_sub


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        ret = v.test_func(None)
        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
