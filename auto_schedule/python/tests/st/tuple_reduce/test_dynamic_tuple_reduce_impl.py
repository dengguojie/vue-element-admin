# # -*- coding:utf-8 -*-
import os
from sch_test_frame.ut import OpUT
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.dsl.compute.reduce import tuple_sum


def tuple_reduce_compute(ph_dx, ph_normalized_x, reduce_axis):
    res_grad_gamma = tbe.dsl.vmul(ph_dx, ph_normalized_x)
    res_grad_gamma, res_grad_beta = tuple_sum([res_grad_gamma, ph_normalized_x], reduce_axis, keepdims=True)
    return res_grad_gamma, res_grad_beta


@register_operator("tuple_reduce")
def dsl_dynamic_tuple_reduce(input_normalized_x, input_dx, shape_gamma, kernel_name="dsl_dynamic_tuple_reduce"):
    reduce_axis = [1, 2] if input_dx.get("format") == "FRACTAL_NZ" else list(
        range(len(input_normalized_x.get("shape")) - len(shape_gamma)))
    tensors, schedules = [], []

    ins = tbe.dsl.classify([input_dx, input_normalized_x, reduce_axis], "tuple_reduce")
    for dx, normalized_x, reduce_axis in ins:
        with tbe.dsl.compute():
            var_dx, var_normalized_x = shape_util.variable_shape([dx, normalized_x], op_mode="tuple_reduce")
            ph_dx = tvm.placeholder(shape=var_dx, name="dx", dtype=dx.get("dtype"))
            ph_normalized_x = tvm.placeholder(shape=var_normalized_x, name="normalized_x",
                                              dtype=normalized_x.get("dtype"))
            outs = tuple_reduce_compute(ph_dx, ph_normalized_x, reduce_axis)
            tensors.append([ph_dx, ph_normalized_x] + list(outs))
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(outs)
        schedules.append(sch)
        config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)

ut_case = OpUT("tuple_reduce", "tuple_reduce.test_dynamic_tuple_reduce_impl", "dsl_dynamic_tuple_reduce")

case1 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, [1024]],
    "case_name":
        "test_dynamic_tuple_reduce_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend710"], case1)

