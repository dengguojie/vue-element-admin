# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


def norm_fake_node_compute(input_features, input_labels, input_axis):
    shape_features = shape_util.shape_to_list(input_features.shape)
    shape_labels = shape_util.shape_to_list(input_labels.shape)
    dtype = input_features.dtype.lower()

    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            shape_util.broadcast_shapes(shape_features, shape_labels, param_name_input1="input_features",
                                        param_name_input2="input_labels")
        input_features = tbe.dsl.broadcast(input_features, shape_broadcast,
                                           dtype)
        input_labels = tbe.dsl.broadcast(input_labels, shape_broadcast,
                                         dtype)
    else:
        shape_broadcast = shape_features

    has_improve_precision = False
    if dtype == "float16":
        input_features = tbe.dsl.cast_to(input_features, "float32")
        input_labels = tbe.dsl.cast_to(input_labels, "float32")
        has_improve_precision = True

    data_max = tbe.dsl.reduce_max(input_features, axis=input_axis, keepdims=True)

    data_max_broadcast = tbe.dsl.broadcast(data_max, shape_broadcast)
    data_sub = tbe.dsl.vsub(input_features, data_max_broadcast)
    data_exp = tbe.dsl.vexp(data_sub)
    data_sum = tbe.dsl.reduce_sum(data_exp, axis=input_axis, keepdims=True)
    data_sum_broadcast = tbe.dsl.broadcast(data_sum, shape_broadcast)
    data_div = tbe.dsl.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.dsl.vlog(data_sum_broadcast)
    data_log = tbe.dsl.vsub(data_sub, data_log_tmp)
    data_mul = tbe.dsl.vmul(input_labels, data_log)
    data_muls = tbe.dsl.vmuls(data_mul, -1)
    loss = tbe.dsl.reduce_sum(data_muls, axis=input_axis, keepdims=True)
    backprop = tbe.dsl.vsub(data_div, input_labels)

    if has_improve_precision:
        loss = tbe.dsl.cast_to(loss, "float16")
        backprop = tbe.dsl.cast_to(backprop, "float16")

    res = [loss, backprop]

    return res



@register_operator("norm_fake_node")
def dsl_dync_norm_fake_node(x, y, out1, out2, axis, kernel_name="dsl_dync_norm_fake_node"):
    input_dtype = x.get("dtype")
    ins = tbe.dsl.classify([x, y, axis], "norm", {"input_shape_type": [1, 1], "same_input_shape_group":[[]]})
    schedules, tensors = [], []

    for (x1, x2, reduce_axis) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x1, x2], op_mode="norm")
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            res = norm_fake_node_compute(data1, data2, reduce_axis)
            tensor_list = [data1, data2] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("norm_fake_node", "norm.test_dynamic_norm_fake_node_impl", "dsl_dync_norm_fake_node")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, [1]],
    "case_name":
        "test_dync_norm_fake_node_1",
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
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [1]],
    "case_name":
        "test_dync_norm_fake_node_2",
    "expect":
        "success",
    "support_expect":
        True
}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
