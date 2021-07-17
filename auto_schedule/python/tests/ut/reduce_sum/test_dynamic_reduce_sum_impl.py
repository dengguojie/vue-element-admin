# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("reduce_sum")
def dsl_dync_reduce_sum(x, y, axis, keepdims, kernel_name="dsl_dync_reduce_sum"):
    input_dtype = x.get("dtype")
    x["rel_pos_to_reduce"] = 'before'
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})

    schedules, tensors = [], []

    for (x, axis) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.reduce_sum(data1, axis.get("value"), keepdims)
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("reduce_sum", "reduce_sum.test_dynamic_reduce_sum_impl", "dsl_dync_reduce_sum")

case_float32_1 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, [1, 3], False],
    "case_name":
        "test_dync_reduce_sum_float32_1",
    "expect":
        "success",
    "support_expect":
        True
}

case_float32_2 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, [0, 2], False],
    "case_name":
        "test_dync_reduce_sum_float32_2",
    "expect":
        "success",
    "support_expect":
        True
}

case_int32_1 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, [0, 2], False],
    "case_name":
        "test_dync_reduce_sum_int32_1",
    "expect":
        "success",
    "support_expect":
        True
}


def calc_expect_func(x, y, axis, keepdims):
    x_value = x.get("value")
    res = np.prod(x_value, axis=axis, keepdims=keepdims)
    return (res,)


compile_case_list = [
    case_float32_1,
    case_float32_2,
    case_int32_1
]

for item in compile_case_list:
    ut_case.add_case(["Ascend920A", "Ascend310", "Ascend910A"], case=item)

