# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator

def norm_5hd_compute(input_x, axis):
    shape = input_x.shape
    dtype = input_x.dtype
    axis = list(axis)
    has_improve_precision = False

    data_max = tbe.dsl.reduce_max(input_x, axis=axis, keepdims=True)
    data_max = tbe.dsl.broadcast(data_max, shape)
    data_subtrac = tbe.dsl.vsub(input_x, data_max)
    if dtype == "float16":
        has_improve_precision = True
        data_subtrac = tbe.dsl.cast_to(data_subtrac, "float32")
    data_exp = tbe.dsl.vexp(data_subtrac)

    data_expsum = tbe.dsl.reduce_sum(data_exp, axis, keepdims=True)
    data_expsum = tbe.dsl.broadcast(data_expsum, shape)
    output = tbe.dsl.vdiv(data_exp, data_expsum)
    if has_improve_precision and dtype == "float16":
        output = tbe.dsl.cast_to(output, "float16")

    return output


@register_operator("norm_5hd")
def dsl_dync_norm_5hd(x, y, axis, kernel_name="dsl_dync_norm_5hd"):
    input_dtype = x.get("dtype")
    extra_params = {"ignore_fractal_format": False}
    ins = tbe.dsl.classify([x, axis], "norm", extra_params)
    schedules, tensors = [], []

    for _, (x, reduce_axis) in enumerate(ins):
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x], op_mode="norm")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = norm_5hd_compute(data1, reduce_axis)
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("norm_5hd", "norm.test_dynamic_norm_5hd_impl", "dsl_dync_norm_5hd")

case1 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, [1, 4]],
    "case_name":
        "test_dync_norm_5hd_1",
    "expect":
        "success",
    "support_expect":
        True
}


ut_case.add_case(["Ascend910A"], case1)
