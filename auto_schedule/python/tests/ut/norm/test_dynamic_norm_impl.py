# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


FP16_MAX = tvm.const(6.5e04, dtype="float16")
FP32_MAX = tvm.const(3.4e38, dtype="float32")

def norm_compute(input_x, pad, axis):
    shape = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype
    axis = list(axis)
    last_dim = len(input_x.shape) - 1
    vcmax_flag = False

    for i in axis:
        if (i == -1) or (i == last_dim):
            vcmax_flag = True
    has_improve_precision = False

    MAX = FP32_MAX if dtype == "float32" else FP16_MAX
    if len(pad) == 2:
        pad_m, pad_n = pad
        input_x = tbe.dsl.set_value(input_x, lambda *i: tvm.any(tvm.all(i[-4] > shape[-4] - 2, i[-1] > pad_n - 1),
                                                                tvm.all(i[-3] > shape[-3] - 2, i[-2] > pad_m - 1)), -MAX)
    elif len(pad) == 1:
        pad_c = pad[0]
        input_x = tbe.dsl.set_value(input_x, lambda *i: tvm.all(i[1] > shape[1] - 2, i[-1] > pad_c - 1), -MAX)

    if dtype == "float16":
        has_improve_precision = True
        input_x = tbe.dsl.cast_to(input_x, "float32")
        data_max = tbe.dsl.reduce_max(input_x, axis=axis, keepdims=True)
    else:
        data_max = tbe.dsl.reduce_max(input_x, axis=axis, keepdims=True)

    data_max = tbe.dsl.broadcast(data_max, shape)
    data_subtrac = tbe.dsl.vsub(input_x, data_max)
    data_exp = tbe.dsl.vexp(data_subtrac)

    if len(pad) == 2:
        pad_m, pad_n = pad
        data_exp = tbe.dsl.set_value(data_exp, lambda *i: tvm.any(tvm.all(i[-4] > shape[-4] - 2, i[-1] > pad_n - 1),
                                                                  tvm.all(i[-3] > shape[-3] - 2, i[-2] > pad_m - 1)), 0)
    elif len(pad) == 1:
        pad_c = pad[0]
        data_exp = tbe.dsl.set_value(data_exp, lambda *i: tvm.all(i[1] > shape[1] - 2, i[-1] > pad_c - 1), 0)

    data_expsum = tbe.dsl.reduce_sum(data_exp, axis, keepdims=True)
    data_expsum = tbe.dsl.broadcast(data_expsum, shape)
    output = tbe.dsl.vdiv(data_exp, data_expsum)
    if has_improve_precision and dtype == "float16":
        output = tbe.dsl.cast_to(output, "float16")

    return output


@register_operator("norm")
def dsl_dync_norm(x, y, axis, kernel_name="dsl_dync_norm"):
    input_dtype = x.get("dtype")
    extra_params = {"disable_optimization": False}
    format = x.get("format")
    ori_format = x.get("ori_format")
    if "ori_shape" in x:
        ori_shape = x.get("ori_shape")
    if format in ["NC1HWC0", "FRACTAL_NZ"]:
        extra_params["disable_optimization"] = True
    x["rel_pos_to_reduce"] = 'before'
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    ins = tbe.dsl.classify([x, input_axis], "norm", extra_params)
    schedules, tensors = [], []

    for (x, axis) in ins:
        with tbe.dsl.compute():
            if format == "NC1HWC0":
                ori_format = ori_format.upper()
                c = ori_shape[ori_format.find('C')]
                c = tbe.dsl.var('c') if c == -1 else c
                pad = [tvm.floormod(c-1, x.get("shape")[-1]) + 1]
            elif format == "FRACTAL_NZ":
                m, n = ori_shape[-2:]
                m = tbe.dsl.var('m') if m == -1 else m
                n = tbe.dsl.var('n') if n == -1 else n
                pad = [tvm.floormod(m-1, x.get("shape")[-2]) + 1, tvm.floormod(n-1, x.get("shape")[-1]) + 1]
            else:
                pad = []
            shape_x = shape_util.variable_shape([x, axis], op_mode="norm")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = norm_compute(data1, pad, axis.get("value"))
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

case6 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None)]
    }, [0, 2]],
    "case_name":
        "test_dync_norm_6",
    "expect":
        "success",
    "support_expect":
        True
}

case7 = {
    "params": [{
        "shape": (10, 10),
        "dtype": "float32",
        "range": [(10, 10), (10, 10)]
    }, {
        "shape": (10, 10),
        "dtype": "float32",
        "range": [(10, 10), (10, 10)]
    }, [1]],
    "case_name":
        "test_dync_norm_7",
    "expect":
        "success",
    "support_expect":
        True
}

case8 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (1, 1, 1, 1),
        "ori_format": "NCHW",
        "shape": (1, 1, 1, 1, 16),
        "format": "NC1HWC0",
        "range": [(1, 1), (1, 1), (1, 1), (1, 1), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (1, 1, 1, 1),
        "ori_format": "NCHW",
        "shape": (1, 1, 1, 1, 16),
        "format": "NC1HWC0",
        "range": [(1, 1), (1, 1), (1, 1), (1, 1), (16, 16)]
    }, [0]],
    "case_name":
        "test_dync_norm_9",
    "expect":
        "success",
    "support_expect":
        True
}

case9 = {
    "params": [{
        "dtype": "float32",
        "ori_shape": (1, 1, 1000, 1000),
        "ori_format": "NCHW",
        "shape": (1, 1, 1000, 1000, 16),
        "format": "NC1HWC0",
        "range": [(1, 1), (1, 1), (1000, 1000), (1000, 1000), (16, 16)]
    }, {
        "dtype": "float32",
        "ori_shape": (1, 1, 1000, 1000),
        "ori_format": "NCHW",
        "shape": (1, 1, 1000, 1000, 16),
        "format": "NC1HWC0",
        "range": [(1, 1), (1, 1), (1000, 1000), (1000, 1000), (16, 16)]
    }, [2, 3]],
    "case_name":
        "test_dync_norm_9",
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
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
