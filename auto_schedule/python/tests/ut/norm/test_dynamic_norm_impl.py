# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


def norm_compute(input_x, axis):
    shape = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype
    axis = list(axis)
    last_dim = len(input_x.shape) - 1
    vcmax_flag = False

    for i in axis:
        if (i == -1) or (i == last_dim):
            vcmax_flag = True
    has_improve_precision = False
    if dtype == "float16":
        has_improve_precision = True
        input_x = tbe.dsl.cast_to(input_x, "float32")
        data_max = tbe.dsl.reduce_max(input_x, axis=axis, keepdims=True)
    else:
        data_max = tbe.dsl.reduce_max(input_x, axis=axis, keepdims=True)

    data_max = tbe.dsl.broadcast(data_max, shape)
    data_subtrac = tbe.dsl.vsub(input_x, data_max)
    data_exp = tbe.dsl.vexp(data_subtrac)
    data_expsum = tbe.dsl.reduce_sum(data_exp, axis, keepdims=True)
    data_expsum = tbe.dsl.broadcast(data_expsum, shape)
    output = tbe.dsl.vdiv(data_exp, data_expsum)
    if has_improve_precision and dtype == "float16":
        output = tbe.dsl.cast_to(output, "float16")

    return output


@register_operator("norm")
def dsl_dync_norm(x, y, axis, kernel_name="dsl_dync_norm"):
    input_dtype = x.get("dtype")
    x["rel_pos_to_reduce"] = 'before'
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    ins = tbe.dsl.classify([x, input_axis], "norm")
    schedules, tensors = [], []

    for (x, axis) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = norm_compute(data1, axis.get("value"))
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("norm", "norm.test_dynamic_norm_impl", "dsl_dync_norm")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, [1]],
    "case_name":
        "test_dync_norm_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [1]],
    "case_name":
        "test_dync_norm_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, [0]],
    "case_name":
        "test_dync_norm_3",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [0]],
    "case_name":
        "test_dync_norm_4",
    "expect":
        "success",
    "support_expect":
        True
}

case5 = {
    "params": [{
        "shape": (-2, ),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [1]],
    "case_name":
        "test_dync_norm_5",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
