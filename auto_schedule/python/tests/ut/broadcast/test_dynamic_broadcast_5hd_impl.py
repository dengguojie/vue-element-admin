# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator

def broadcast_5hd_compute(input_x, input_y, output, kernel_name):
    x_shape, y_shape, z_shape = shape_util.broadcast_shapes(input_x.shape, input_y.shape,
                                                            param_name_input1="input_x",
                                                            param_name_input2="input_y")
    dtype_x = input_x.dtype
    int_list = ("int8", "uint8", "int32")
    input_x = tbe.dsl.broadcast(input_x, z_shape)
    input_y = tbe.dsl.broadcast(input_y, z_shape)
    res = tbe.dsl.vdiv(input_x, input_y)

    if dtype_x in int_list:
        res = tbe.dsl.floor(res)

    res = tbe.dsl.cast_to(res, dtype_x)

    return res


@register_operator("broadcast_5hd")
def dsl_dync_broadcast_5hd(x1, x2, output, kernel_name="dsl_dync_broadcast_5hd"):
    input_dtype = x1.get("dtype")
    extra_params = {"ignore_fractal_format": False}
    ins = tbe.dsl.classify([x1, x2], "broadcast", extra_params)
    schedules, tensors = [], []

    for (input_x, input_y) in ins:
        with tbe.dsl.compute():
            x_shape, y_shape = shape_util.variable_shape([input_x, input_y])
            tensor_x = tvm.placeholder(x_shape, input_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, input_dtype, "tensor_y")
            res = broadcast_5hd_compute(tensor_x, tensor_y, output, kernel_name)
            tensors.append((tensor_x, tensor_y, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("broadcast", "broadcast.test_dynamic_broadcast_5hd_impl", "dsl_dync_broadcast_5hd")

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
        "test_dsl_dync_broadcast_5hd_1",
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
        "ori_shape": (2, 128, 1, 16),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    },  {
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 16),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }],
    "case_name":
        "test_dsl_const_broadcast_5hd_1",
    "case_type":
    "static",
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
        "ori_shape": (2, 128, 1, 16),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    },  {
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 16),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }],
    "case_name":
        "test_dsl_const_broadcast_5hd_1",
    "case_type":
    "static",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 1),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (2, 128, 1, 16),
        "ori_format": "NCHW",
        "shape": (2, 1, 128, 1, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    },  {
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 16),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }],
    "case_name":
        "test_dsl_const_broadcast_5hd_invalid_format",
    "case_type":
    "static",
    "expect":RuntimeError,
    "support_expect":False
}

case4 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 1),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 14),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (2, 128, 1, 16),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 1, 14),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    },  {
        "dtype": "float32",
        "ori_shape": (2, 128, 16, 16),
        "ori_format": "NHWC",
        "shape": (2, 1, 128, 16, 16),
        "format": "NC1HWC0",
        "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
    }],
    "case_name":
        "test_dsl_const_broadcast_5hd_invalid_C0",
    "case_type":"static",
    "expect": RuntimeError,
    "support_expect":False
}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
