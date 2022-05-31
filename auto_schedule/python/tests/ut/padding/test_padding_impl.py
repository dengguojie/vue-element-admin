# # -*- coding:utf-8 -*-
import warnings
from typing import Callable

import tbe
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from sch_test_frame.ut.helper import padding_helper
from tbe import tvm
from tbe.dsl.base import operation, var_api
from tbe.dsl.base.padding import padding

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_padding_impl")

_5HD_FORMAT = ("N", "C1", "H", "W", "C0")


def _simplify(v):
    if isinstance(v, tvm.expr.ConstExpr):
        return v.value
    if isinstance(v, Callable):
        return Callable
    return None


@add_cust_test_func(ut_case)
def test_calc_padding_in_abs(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        fp16 = "float16"

        ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
        abs_1 = tbe.dsl.vabs(ph_1)

        actions = padding.calc_padding(abs_1)

        return len(actions) == 0


@add_cust_test_func(ut_case)
def test_calc_padding_in_div(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        n, c1, h, w, c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        c = padding_helper.const_c(42)
        var_api.set_attr(c1, "original", c)
        var_api.set_attr(c0, "original", c)
        shape = (n, c1, h, w, c0)
        fp16 = "float16"

        ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
        ph_2 = tvm.placeholder(shape, dtype=fp16, name="ph_2")
        div_1 = tbe.dsl.vdiv(ph_1, ph_2)

        actions = padding.calc_padding(div_1)

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [ph_2]
        expect_values = [1]
        expect_targets = [[]]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_broadcast(_):
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

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_0")
        ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_1")
        brc_0 = tbe.dsl.broadcast(ph_0, shape1)
        add_0 = tbe.dsl.vadd(brc_0, ph_1)

        actions = padding.calc_padding(add_0)

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [ph_0, add_0]
        expect_values = [Callable, 0]
        expect_targets = [[], []]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_sum_without_pad_axis(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        adds_0 = tbe.dsl.vadds(cast_0, 1)
        reduce_sum_0 = tbe.dsl.reduce_sum(adds_0, (0, 1, 2, 3, 4))

        actions = padding.calc_padding([reduce_sum_0])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [adds_0]
        expect_values = [0]
        expect_targets = [[]]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_sum_with_pad_axis(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        reduce_sum_0 = tbe.dsl.reduce_sum(cast_0, (0, 2, 3))
        cast_1 = tbe.dsl.cast_to(reduce_sum_0, "float16")

        actions = padding.calc_padding([cast_1])

        return len(actions) == 0


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_sum_with_pad_axis_and_pad_0(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        reduce_sum_0 = tbe.dsl.reduce_sum(cast_0, (0, 2, 3))
        cast_1 = tbe.dsl.cast_to(reduce_sum_0, "float16")
        adds_1 = tbe.dsl.vadds(cast_1, 1)

        actions = padding.calc_padding([adds_1])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [adds_1]
        expect_values = [0]
        expect_targets = [[]]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_max_without_pad_axis(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        reduce_sum_0 = tbe.dsl.reduce_max(cast_0, (0, 1, 2, 3, 4))
        cast_1 = tbe.dsl.cast_to(reduce_sum_0, "float16")

        actions = padding.calc_padding([cast_1])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [cast_0]
        expect_values = [-3.4028234663852886e+38]
        expect_targets = [[]]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_max_with_pad_axis(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        reduce_max_0 = tbe.dsl.reduce_max(cast_0, (2, 3))
        cast_1 = tbe.dsl.cast_to(reduce_max_0, "float16")

        actions = padding.calc_padding([cast_1])

        return len(actions) == 0


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_min_with_pad_axis_fuse_hw(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        _5HD_FORMAT = ("N", "C1", ["H", "W"], "C0")
        x0_n, x0_c1, x0_hw, x0_c0 = padding_helper.const_x((4, 1, 6*8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(10)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_hw, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        reduce_min_0 = tbe.dsl.reduce_min(cast_0, (1, 3))
        cast_1 = tbe.dsl.cast_to(reduce_min_0, "float16")

        actions = padding.calc_padding([cast_1])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [cast_0]
        expect_values = [3.4028234663852886e+38]
        expect_targets = [[]]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_max_with_same_inputs_in_binary(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(1)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")

        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        reduce_max_0 = tbe.dsl.reduce_min(cast_0, (2, 3))
        cast_1 = tbe.dsl.cast_to(reduce_max_0, "float16")
        div_1 = tbe.dsl.vdiv(cast_1, cast_1)
        div_2 = tbe.dsl.vdiv(cast_1, cast_1)
        mul_1 = tbe.dsl.vmul(div_1, div_2)

        actions = padding.calc_padding([mul_1])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [cast_1, mul_1]
        expect_values = [1, 0]
        expect_targets = [[], []]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_min_with_pad_axis_keepdims(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"
        _5HD_FORMAT = ("N", ["C1"], "H", "W", ["C0"])
        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 1, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(10)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")

        cast_0 = tbe.dsl.cast_to(ph_0, "float32")
        reduce_min_0 = tbe.dsl.reduce_min(cast_0, (1, 4), keepdims=True)
        cast_1 = tbe.dsl.cast_to(reduce_min_0, "float16")

        actions = padding.calc_padding([cast_1])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [cast_0]
        expect_values = [3.4028234663852886e+38]
        expect_targets = [[]]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_softmax(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"
        _5HD_FORMAT = ("N", "C1", "H", "W", "C0")
        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(38)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_0")
        reduce_max_0 = tbe.dsl.reduce_max(ph_0, (1, 4), keepdims=True)
        brc_0 = tbe.dsl.broadcast(reduce_max_0, shape0)
        sub_0 = tbe.dsl.vsub(ph_0, brc_0)
        exp_0 = tbe.dsl.vexp(sub_0)
        reduce_sum_0 = tbe.dsl.reduce_sum(exp_0, (1, 4), keepdims=True)
        log_0 = tbe.dsl.vlog(reduce_sum_0)
        brc_1 = tbe.dsl.broadcast(log_0, shape0)
        sub_1 = tbe.dsl.vsub(sub_0, brc_1)

        actions = padding.calc_padding([sub_1])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [ph_0, exp_0, sub_1]
        expect_values = [-65504, 0, 0]
        expect_targets = [[reduce_max_0], [], []]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_softmax_align(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"
        _5HD_FORMAT = ("N", "C1", "H", "W", "C0")
        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(48)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        reduce_max_0 = tbe.dsl.reduce_max(ph_0, (1, 4), keepdims=True)
        brc_0 = tbe.dsl.broadcast(reduce_max_0, shape0)
        sub_0 = tbe.dsl.vsub(ph_0, brc_0)
        exp_0 = tbe.dsl.vexp(sub_0)
        reduce_sum_0 = tbe.dsl.reduce_sum(exp_0, (1, 4), keepdims=True)
        log_0 = tbe.dsl.vlog(reduce_sum_0)
        brc_1 = tbe.dsl.broadcast(log_0, shape0)
        sub_1 = tbe.dsl.vsub(sub_0, brc_1)

        actions = padding.calc_padding([sub_1])
        return len(actions) == 0


@add_cust_test_func(ut_case)
def test_calc_padding_softmax_with_multi_step_brc(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"
        _5HD_FORMAT = ("N", "C1", "H", "W", "C0")
        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(38)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)
        x0_c1_1 = var_api.const(1, annotation={"axis_type": "C1", "original": x0_c})
        shape_brc_0 = (x0_n, x0_c1_1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        reduce_max_0 = tbe.dsl.reduce_max(ph_0, (1, 4), keepdims=True)
        brc_0 = tbe.dsl.broadcast(reduce_max_0, shape_brc_0)
        brc_1 = tbe.dsl.broadcast(brc_0, shape0)
        sub_0 = tbe.dsl.vsub(ph_0, brc_1)
        exp_0 = tbe.dsl.vexp(sub_0)
        reduce_sum_0 = tbe.dsl.reduce_sum(exp_0, (1, 4), keepdims=True)
        log_0 = tbe.dsl.vlog(reduce_sum_0)
        brc_2 = tbe.dsl.broadcast(log_0, shape_brc_0)
        brc_3 = tbe.dsl.broadcast(brc_2, shape0)
        sub_1 = tbe.dsl.vsub(sub_0, brc_3)

        actions = padding.calc_padding([sub_1])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [ph_0, exp_0, sub_1]
        expect_values = [-65504, 0, 0]
        expect_targets = [[reduce_max_0], [], []]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_softmax_div(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"
        _5HD_FORMAT = ("N", "C1", "H", "W", "C0")
        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(38)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_0")
        reduce_max_0 = tbe.dsl.reduce_max(ph_0, (1, 4), keepdims=True)
        brc_0 = tbe.dsl.broadcast(reduce_max_0, shape0)
        sub_0 = tbe.dsl.vsub(ph_0, brc_0)
        exp_0 = tbe.dsl.vexp(sub_0)
        reduce_sum_0 = tbe.dsl.reduce_sum(exp_0, (1, 4), keepdims=True)
        brc_1 = tbe.dsl.broadcast(reduce_sum_0, shape0)
        div_0 = tbe.dsl.vdiv(exp_0, brc_1)

        actions = padding.calc_padding([div_0])

        tensors = [x.get_tensor() for x in actions]
        values = [_simplify(x.get_value()) for x in actions]
        targets = [x.get_target_tensors() for x in actions]
        expect_tensors = [ph_0, exp_0]
        expect_values = [-65504, 0]
        expect_targets = [[reduce_max_0], []]
        assert_tensor = tensors == expect_tensors
        assert_value = values == expect_values
        assert_target = targets == expect_targets
        return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_reduce_prod_without_pad_axis(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"
        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((4, 3, 6, 8, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(40)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
        reduce_prod_0 = tbe.dsl.reduce_prod(ph_0, (2, 3))

        actions = padding.calc_padding([reduce_prod_0])

        return len(actions) == 0


# @add_cust_test_func(ut_case)
# def test_calc_padding_div(_):
#     with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
#         fp16 = "float16"

#         x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((2, 1, 128, 16, 16), _5HD_FORMAT)
#         x0_c = padding_helper.const_c(1)
#         var_api.set_attr(x0_c1, "original", x0_c)
#         var_api.set_attr(x0_c0, "original", x0_c)
#         shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

#         x1_n, x1_c1, x1_h, x1_w, x1_c0 = padding_helper.const_x((2, 1, 128, 1, 16), _5HD_FORMAT)
#         x1_c = padding_helper.const_c(16)
#         var_api.set_attr(x1_c1, "original", x1_c)
#         var_api.set_attr(x1_c0, "original", x1_c)
#         shape1 = (x1_n, x1_c1, x1_h, x1_w, x1_c0)

#         max_ = var_api.max
#         shape_max = (max_(x0_n, x1_n), max_(x0_c1, x1_c1), max_(x0_h, x1_h), 
#                      max_(x0_w, x1_w), max_(x0_c0, x1_c0))

#         ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_1")
#         ph_1 = tvm.placeholder(shape1, dtype=fp16, name="ph_2")
#         brc_0 = tbe.dsl.broadcast(ph_0, shape_max)
#         brc_1 = tbe.dsl.broadcast(ph_1, shape_max)
#         res = tbe.dsl.vdiv(brc_0, brc_1)

#         actions = padding.calc_padding(res)

#         tensors = [x.get_tensor() for x in actions]
#         values = [_simplify(x.get_value()) for x in actions]
#         targets = [x.get_target_tensors() for x in actions]
#         expect_tensors = [ph_0]
#         expect_values = [Callable]
#         expect_targets = [[]]
#         assert_tensor = tensors == expect_tensors
#         assert_value = values == expect_values
#         assert_target = targets == expect_targets
#         return all((assert_tensor, assert_value, assert_target))


@add_cust_test_func(ut_case)
def test_calc_padding_relu(_):
    with operation.dynamic(), padding_helper.soc_context("Ascend910A"):
        fp16 = "float16"

        x0_n, x0_c1, x0_h, x0_w, x0_c0 = padding_helper.const_x((2, 3, 128, 16, 16), _5HD_FORMAT)
        x0_c = padding_helper.const_c(42)
        var_api.set_attr(x0_c1, "original", x0_c)
        var_api.set_attr(x0_c0, "original", x0_c)
        shape0 = (x0_n, x0_c1, x0_h, x0_w, x0_c0)

        ph_0 = tvm.placeholder(shape0, dtype=fp16, name="ph_0")
        relu_0 = tbe.dsl.vrelu(ph_0)
        lrelu_0 = tbe.dsl.vlrelu(relu_0)

        actions = padding.calc_padding(lrelu_0)

        return len(actions) == 0


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_calc_padding_softmax":
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
