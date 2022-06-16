# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator

def elewise_5hd_compute(input_x, input_y, output, kernel_name):
    dtype_x = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    res = tbe.dsl.vdiv(input_x, input_y)

    if dtype_x in int_list:
        res = tbe.dsl.floor(res)

    res = tbe.dsl.cast_to(res, dtype_x)

    return res


@register_operator("elewise_5hd")
def dsl_dync_elewise_5hd(x1, x2, output, kernel_name="dsl_dync_elewise_5hd"):
    input_dtype = x1.get("dtype")
    extra_params = {"ignore_fractal_format": False}
    ins = tbe.dsl.classify([x1, x2], "elewise", extra_params)
    schedules, tensors = [], []

    for (input_x, input_y) in ins:
        with tbe.dsl.compute():
            x_shape, y_shape = shape_util.variable_shape([input_x, input_y])
            tensor_x = tvm.placeholder(x_shape, input_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, input_dtype, "tensor_y")
            res = elewise_5hd_compute(tensor_x, tensor_y, output, kernel_name)
            tensors.append((tensor_x, tensor_y, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("elewise", "elewise.test_dynamic_elewise_5hd_impl", "dsl_dync_elewise_5hd")

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
    },  {
        "dtype": "float32",
        "ori_shape": (-1, -1, -1, -1),
        "ori_format": "NHWC",
        "shape": (-1, -1, -1, -1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }],
    "case_name":
        "test_dynamic_elewise_5hd_impl_case1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 1),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 1),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    },  {
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 1),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }],
    "case_name":
        "test_dynamic_elewise_5hd_impl_case2",
    "case_type":
    "static",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
