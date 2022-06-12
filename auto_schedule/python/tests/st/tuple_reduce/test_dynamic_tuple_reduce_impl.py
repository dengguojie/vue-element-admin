# # -*- coding:utf-8 -*-
import os
from sch_test_frame.ut import OpUT
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.dsl.compute.reduce import tuple_sum


def tuple_reduce_compute(x, reduce_axis):
    x = tbe.dsl.cast_to(x, "float32")
    square_x = tbe.dsl.vmul(x, x)
    x_sum, square_x_sum = tbe.dsl.tuple_sum([x, square_x], reduce_axis, True)
    return [x_sum, square_x_sum]


@register_operator("tuple_reduce")
def dsl_dynamic_tuple_reduce(x, reduce_axis, kernel_name="dsl_dynamic_tuple_reduce"):
    tensors, schedules = [], []

    ins = tbe.dsl.classify([x, reduce_axis], "tuple_reduce")
    for _x, _reduce_axis in ins:
        with tbe.dsl.compute():
            shape_var = shape_util.variable_shape([_x], op_mode="tuple_reduce")[0]
            data_input = tvm.placeholder(shape_var, name="data_input", dtype=_x.get("dtype"))
            outs = tuple_reduce_compute(data_input, _reduce_axis)
            tensor_list = [data_input] + list(outs)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(outs)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)

ut_case = OpUT("tuple_reduce", "tuple_reduce.test_dynamic_tuple_reduce_impl", "dsl_dynamic_tuple_reduce")

case1 = {
    "params": [{
        "shape": (-1, -1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]
    }, [0, 2, 3]],
    "case_name":
        "test_dynamic_tuple_reduce_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (32, 4, 112, 112, 16),
        "dtype": "float16",
        "range": [(32, 32), (4, 4), (112, 112), (112, 112), (16, 16)]
    }, [0, 2, 3]],
    "case_name":
        "test_dynamic_tuple_reduce_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend310P3"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend310P3"], case2)
