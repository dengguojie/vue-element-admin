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
from tbe.dsl.padding.graph import Node
from tbe.dsl.padding.value import PaddingValueType
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType
from tbe.tvm.expr import ConstExpr

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_util_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


@add_cust_test_func(ut_case)
def test_is_placeholder_true(_):
    ph_1 = tvm.placeholder((2, 16), dtype="float32", name="ph_1")
    return util.is_placeholder(ph_1) is True


@add_cust_test_func(ut_case)
def test_is_placeholder_false(_):
    shape = (2, 16)
    ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
    adds_1 = tvm.compute(shape, lambda *i: ph_1[i] + 10, name="adds_1")
    return util.is_placeholder(adds_1) is False


@add_cust_test_func(ut_case)
def test_get_insn_non_vl(_):
    shape = (2, 16)
    ph_1 = tvm.placeholder(shape, dtype="float32", name="ph_1")
    abs_1 = tbe.dsl.vabs(ph_1)
    return "elewise_single_abs" == util.get_insn(Node(Graph([abs_1]), abs_1))


@add_cust_test_func(ut_case)
def test_get_insn_with_vl(_):
    shape = (3, 16)
    ph_0 = tvm.placeholder(shape, dtype="float16", name="ph_0")
    ph_1 = tvm.placeholder(shape, dtype="float16", name="ph_1")
    ph_2 = tvm.placeholder(shape, dtype="float16", name="ph_2")
    madd_0 = tbe.dsl.vmadd(ph_0, ph_1, ph_2)
    return "elewise_multiple_madd" == util.get_insn(Node(Graph([madd_0]), madd_0))


@add_cust_test_func(ut_case)
def test_np_num_to_tvm(_):
    np_128 = np.int32(128)
    tvm_128 = util.np_num_to_tvm(np_128)

    assert_type = isinstance(tvm_128, tvm.expr.IntImm)
    assert_dtype = tvm_128.dtype == "int32"
    assert_value = tvm_128.value == 128

    return assert_type and assert_dtype and assert_value


@add_cust_test_func(ut_case)
def test_tvm_const_to_np(_):
    tvm_128 = tvm.const(128, "int32")
    np_128 = util.tvm_const_to_np(tvm_128)

    assert_type = isinstance(np_128, np.int32)
    assert_dtype = str(np_128.dtype) == "int32"
    assert_value = np_128.item() == 128

    return assert_type and assert_dtype and assert_value


@add_cust_test_func(ut_case)
def test_get_normal_condition(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((3, 3, 2, 3, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        fp16 = "float16"

        ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
        abs_1 = tbe.dsl.vabs(ph_1)

        node = Node(Graph([abs_1]), abs_1)

    condition_func = util.get_normal_condition(node)
    expected_func = lambda *i: i[1] == 42//16 and i[4] >= 42%16

    return padding_helper.cmp_condition(shape, condition_func, expected_func)


@add_cust_test_func(ut_case)
def test_get_brc_condition_value(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((3, 1, 2, 3, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((3, 3, 2, 3, 16), _5HD_FORMAT)
        x1_c = padding_helper.const_c(42)
        var_api.set_attr(x1_c1, "original", x1_c)
        var_api.set_attr(x1_c0, "original", x1_c)
        shape1 = (x1_n, x1_c1, x1_h, x1_w, x1_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_2")

        brc_0 = tbe.dsl.broadcast(ph_0, shape1)

        add_0 = tbe.dsl.vadd(brc_0, ph_1)

        node = Node(Graph([add_0]), ph_0)

    cond_func, value_func = util.get_brc_condition_value(node)
    expected_cond_func = lambda *i: i[1] == 0 and i[4] > 0
    expected_value_func = lambda *i: (i[0], i[1], i[2], i[3], 0)

    assert_cond_func = padding_helper.cmp_condition(shape0, cond_func, expected_cond_func)
    assert_value_func = padding_helper.cmp_value(shape0, value_func, expected_value_func)

    return assert_cond_func and assert_value_func


@add_cust_test_func(ut_case)
def test_is_d_format(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (tvm.const(1), n, c1, h, w, c0)
        fp16 = "float16"

        ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
        abs_1 = tbe.dsl.vabs(ph_1)

        node = Node(Graph([abs_1]), abs_1)

    return util.is_d_format(node) is True


@add_cust_test_func(ut_case)
def test_exist_pad(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        fp16 = "float16"

        ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
        abs_1 = tbe.dsl.vabs(ph_1)

        node = Node(Graph([abs_1]), abs_1)

    return util.exist_pad(node) is True


@add_cust_test_func(ut_case)
def test_exist_pad_align(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(48)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        fp16 = "float16"

        ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
        abs_1 = tbe.dsl.vabs(ph_1)

        node = Node(Graph([abs_1]), abs_1)

    return util.exist_pad(node) is False


@add_cust_test_func(ut_case)
def test_check_valid_true(_):
    a = np.int32(3)
    b = np.int32(1)
    ret = util.check_valid(lambda: a/b)
    return ret is True


@add_cust_test_func(ut_case)
def test_check_valid_false(_):
    a = np.int32(3)
    b = np.int32(0)
    ret = util.check_valid(lambda: a/b)
    return ret is False


@add_cust_test_func(ut_case)
def test_check_valid_add_false(_):
    a = np.int32(2147483640)
    b = np.int32(8)
    ret = util.check_valid(lambda: a + b)
    return ret is False


@add_cust_test_func(ut_case)
def test_raise_error(_):
    try:
        util.raise_error("test message", "E90002")
    except RuntimeError as e:
        return "test message" in e.args[1] and e.args[0]["errCode"] == "E90002"
    return False


@add_cust_test_func(ut_case)
def test_get_hs_b(_):
    ph_1 = tvm.placeholder((3, 16), dtype="float16", name="ph_1")
    adds_1 = tbe.dsl.vadds(ph_1, 100)
    node = Node(Graph([adds_1]), adds_1)
    b = util.get_hs_b(node)
    return isinstance(b, ConstExpr) and b.value == 100

@add_cust_test_func(ut_case)
def test_get_target_dtype(_):
    ph_1 = tvm.placeholder((3, 16), dtype="float16", name="ph_1")
    cast_1 = tbe.dsl.cast_to(ph_1, "int32")
    node = Node(Graph([cast_1]), cast_1)
    b = util.get_target_dtype(node)
    return b == "int32"


@add_cust_test_func(ut_case)
def test_equal_0_with_(_):
    a = np.float32(0)
    return util.equal_0(a) is True


@add_cust_test_func(ut_case)
def test_equal_1(_):
    a = tvm.const(1, "int32")
    return util.equal_1(a) is True


@add_cust_test_func(ut_case)
def test_equal_max(_):
    a = np.int32(np.iinfo("int32").max)
    return util.equal_max(a)


@add_cust_test_func(ut_case)
def test_equal_min(_):
    a = np.float32(np.finfo("float32").min)
    return util.equal_min(a)


@add_cust_test_func(ut_case)
def test_is_max_pvalue(_):
    a = util.new_pvalue_max("int32")
    return util.is_max_pvalue(a) is True


@add_cust_test_func(ut_case)
def test_is_min_pvalue(_):
    a = util.new_pvalue_min("float16")
    return util.is_min_pvalue(a) is True


@add_cust_test_func(ut_case)
def test_is_0_pvalue(_):
    a = util.new_pvalue_0("float16")
    return util.is_0_pvalue(a) is True


@add_cust_test_func(ut_case)
def test_is_1_pvalue(_):
    a = util.new_pvalue_1("float16")
    return util.is_1_pvalue(a) is True


@add_cust_test_func(ut_case)
def test_is_tensor_pvalue(_):
    pvalue = util.new_pvalue_tensor("float16")
    return util.is_tensor_pvalue(pvalue)


@add_cust_test_func(ut_case)
def test_new_np_num_max(_):
    return util.new_np_num_max("int32") == 2147483647


@add_cust_test_func(ut_case)
def test_new_np_num_min(_):
    return util.new_np_num_min("float32") == -3.4028234663852886e+38


@add_cust_test_func(ut_case)
def test_new_np_num_0(_):
    return util.new_np_num_0("float32") == 0


@add_cust_test_func(ut_case)
def test_new_np_num_1(_):
    return util.new_np_num_1("float32") == 1


@add_cust_test_func(ut_case)
def test_new_np_num_x(_):
    return util.new_np_num_x(100, "int32") == 100


@add_cust_test_func(ut_case)
def test_new_pvalue_max(_):
    a = util.new_pvalue_max("float16")
    return a.value == 6.55040e+04


@add_cust_test_func(ut_case)
def test_new_pvalue_min(_):
    a = util.new_pvalue_min("float32")
    return a.value == -3.4028234663852886e+38


@add_cust_test_func(ut_case)
def test_new_pvalue_0(_):
    a = util.new_pvalue_0("int32")
    return a.value == 0


@add_cust_test_func(ut_case)
def test_new_pvalue_1(_):
    a = util.new_pvalue_1("float32")
    return a.value == 1


@add_cust_test_func(ut_case)
def test_new_pvalue_x(_):
    a = util.new_pvalue_x(3, "int32")
    return a.value == 3


@add_cust_test_func(ut_case)
def test_new_pvalue_tensor(_):
    a = util.new_pvalue_tensor("int32")
    return a.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_new_pvalue_any(_):
    a = util.new_pvalue_any("int32")
    return a.type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_new_pvalue_input_none(_):
    return util.new_pvalue(None) is None


@add_cust_test_func(ut_case)
def test_new_pvalue_input_tensor(_):
    pvalue = util.new_pvalue_tensor("float16")
    return util.new_pvalue(pvalue).type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_new_pvalue_input_any(_):
    pvalue = util.new_pvalue_any("float16")
    return util.new_pvalue(pvalue).type == PaddingValueType.ANY


@add_cust_test_func(ut_case)
def test_new_pvalue_input_exact(_):
    pvalue = util.new_pvalue_x(np.int32(100), "int32")
    return util.new_pvalue(pvalue).value == np.int32(100)


@add_cust_test_func(ut_case)
def test_get_min(_):
    return util.get_min("int32") == -2147483648


@add_cust_test_func(ut_case)
def test_get_max(_):
    return util.get_max("float32") == 3.4028234663852886e+38


@add_cust_test_func(ut_case)
def test_is_brc_node(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((3, 1, 2, 3, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((3, 3, 2, 3, 16), _5HD_FORMAT)
        x1_c = padding_helper.const_c(42)
        var_api.set_attr(x1_c1, "original", x1_c)
        var_api.set_attr(x1_c0, "original", x1_c)
        shape1 = (x1_n, x1_c1, x1_h, x1_w, x1_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_2")

        brc_0 = tbe.dsl.broadcast(ph_0, shape1)

        add_0 = tbe.dsl.vadd(brc_0, ph_1)

        node = Node(Graph([add_0]), brc_0)

    return util.is_brc_node(node) is True


@add_cust_test_func(ut_case)
def test_eq_expr_all_none(_):
    return util.eq_expr((None, None), (None, None)) is True


@add_cust_test_func(ut_case)
def test_eq_expr_partial_none(_):
    return util.eq_expr((None, None), (None, 1)) is False


@add_cust_test_func(ut_case)
def test_eq_expr_partial_var(_):
    x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((3, 1, 2, 3, 16), _5HD_FORMAT)
    x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((3, 3, 2, 3, 16), _5HD_FORMAT)

    return util.eq_expr((x0_n, x0_c0), (x1_n, 16)) is True


@add_cust_test_func(ut_case)
def test_eq_expr_all_var(_):
    x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((3, 1, 2, 3, 16), _5HD_FORMAT)
    x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((3, 3, 2, 3, 16), _5HD_FORMAT)

    return util.eq_expr((x0_n, var_api.max(x0_c1, x1_c1), x0_c0), [x1_n, var_api.max(x0_c1, x1_c1), x1_c0]) is True


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_calc_padding_softmax_div":
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
