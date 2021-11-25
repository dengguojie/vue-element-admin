# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


def norm_mid_output_compute(input_x, input_gamma, input_beta, input_axis):
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    input_x1 = input_x
    cast_dtype = dtype
    cast_dtype_precision = dtype
    is_cast = False
    if dtype == "float16":
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        input_x = tbe.dsl.cast_to(input_x, "float32")
        input_x1 = tbe.dsl.cast_to(input_x1, "float32")
        input_gamma = tbe.dsl.cast_to(input_gamma, "float32")
        input_beta = tbe.dsl.cast_to(input_beta, "float32")
        is_cast = True

    reduce_elts = 1.0
    for i in input_axis:
        reduce_elts *= shape_x[i]
    if isinstance(reduce_elts, float):
        mean_cofs = reduce_elts ** (-1)
        mean_cof = tvm.const(mean_cofs, dtype=cast_dtype)
    else:
        mean_cof = tbe.dsl.var("mean_cof", dtype=cast_dtype)

    # DSL description of the mean calculation process
    mean_muls = tbe.dsl.vmuls(input_x, mean_cof)
    mean = tbe.dsl.reduce_sum(mean_muls, axis=input_axis, keepdims=True)
    # workspace special case
    if is_cast:
        mean_16 = tbe.dsl.cast_to(mean, "float16")
        mean = tbe.dsl.cast_to(mean_16, "float32")

    # DSL description of the variance calculation process
    mean_variance_broadcast = tbe.dsl.broadcast(mean, shape_x)
    variance_sub = tbe.dsl.vsub(input_x1, mean_variance_broadcast)
    variance_mul = tbe.dsl.vmul(variance_sub, variance_sub)
    variance_muls = tbe.dsl.vmuls(variance_mul, mean_cof)
    variance = tbe.dsl.reduce_sum(variance_muls, axis=input_axis, keepdims=True)
    if is_cast:
        variance_16 = tbe.dsl.cast_to(variance, "float16")
        variance = tbe.dsl.cast_to(variance_16, "float32")
    normalize_sub = variance_sub

    # DSL description of the normalize calculation process
    epsilon = tvm.const(1, dtype=cast_dtype)
    variance_normalize_broadcast = tbe.dsl.broadcast(variance, shape_x)
    normalize_add = tbe.dsl.vadds(variance_normalize_broadcast, epsilon)
    normalize_log = tbe.dsl.vlog(normalize_add)
    normalize_log_mul = tbe.dsl.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
    normalize_exp = tbe.dsl.vexp(normalize_log_mul)
    normalize_mul = tbe.dsl.vmul(normalize_sub, normalize_exp)

    gamma_broadcast = tbe.dsl.broadcast(input_gamma, shape_x)
    beta_broadcast = tbe.dsl.broadcast(input_beta, shape_x)
    scale_mul = tbe.dsl.vmul(normalize_mul, gamma_broadcast)
    res = tbe.dsl.vadd(scale_mul, beta_broadcast)

    if is_cast:
        res = tbe.dsl.cast_to(res, "float16")
        return mean_16, variance_16, res

    return mean, variance, res



@register_operator("norm_mid_output")
def dsl_dync_norm_mid_output(x, y, z, out1, out2, out3, reduce_axis, broadcast_axis, kernel_name="dsl_dync_norm_mid_output"):
    input_dtype = x.get("dtype")
    ins = tbe.dsl.classify([x, y, z, reduce_axis], "norm", {"input_shape_type": [0, 1, 1], "same_input_shape_group":[[1, 2]],
                                                            "compile_broadcast_axes": {1: broadcast_axis, 2: broadcast_axis}})
    schedules, tensors = [], []

    for (x1, x2, x3, reduce_axis) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y, shape_z = shape_util.variable_shape([x1, x2, x3], op_mode="norm")
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            data3 = tvm.placeholder(shape_z, name='data3', dtype=input_dtype)
            y1, y2, y3 = norm_mid_output_compute(data1, data2, data3, reduce_axis)
            tensors.append([data1, data2, data3, y1, y2, y3])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule([y1, y2, y3])
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("norm_mid_output", "norm.test_dynamic_norm_mid_output_impl", "dsl_dync_norm_mid_output")

case1 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1,),
        "dtype": "float32",
        "range": [(1, None)]
    }, {
        "shape": (-1,),
        "dtype": "float32",
        "range": [(1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, [2], [0, 1]],
    "case_name":
        "test_dync_norm_mid_output_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None)]
    }, [2], [0, 1]],
    "case_name":
        "test_dync_norm_mid_output_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
