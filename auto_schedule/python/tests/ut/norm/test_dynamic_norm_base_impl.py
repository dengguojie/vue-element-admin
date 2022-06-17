# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


FP16_MAX = tvm.const(6.5e04, dtype="float16")
FP32_MAX = tvm.const(3.4e38, dtype="float32")

def norm_base_compute(input_x, axis):
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
    attributes = input_x.op.attrs
    disable_fuse_axes = attributes["disable_fuse_axes"]
    ori_shape = shape_util.shape_to_list(attributes["ori_shape"])
    ori_format = attributes["ori_format"].value
    format = attributes["format"].value

    if format == "NC1HWC0":
        idx_c1, idx_c0 = shape_util.shape_to_list(disable_fuse_axes)
        ori_format = ori_format.upper()
        c = ori_shape[ori_format.find('C')]
        c = tbe.dsl.var('c') if c == -1 else c
        pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1

    if format == "NC1HWC0":
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

    if format == "NC1HWC0":
        data_exp = tbe.dsl.set_value(data_exp, lambda *i: tvm.all(i[1] > shape[1] - 2, i[-1] > pad_c - 1), 0)

    data_expsum = tbe.dsl.reduce_sum(data_exp, axis, keepdims=True)
    data_expsum = tbe.dsl.broadcast(data_expsum, shape)
    output = tbe.dsl.vdiv(data_exp, data_expsum)
    if has_improve_precision and dtype == "float16":
        output = tbe.dsl.cast_to(output, "float16")

    return output


@register_operator("norm_base")
def dsl_dync_norm_base(x, y, axis, kernel_name="dsl_dync_norm_base"):
    input_dtype = x.get("dtype")
    extra_params = {}
    format = "ND"
    if "format" in x:
        format = x.get("format")
    ori_format = "ND"
    if "ori_format" in x:
        ori_format = x.get("ori_format")
    ori_shape = x.get("shape")
    if "ori_shape" in x:
        ori_shape = x.get("ori_shape")
    if format == "NC1HWC0":
        extra_params.update({"disable_fuse_axes": [1, 4]})
    ins = tbe.dsl.classify([x, axis], "norm", extra_params)
    schedules, tensors = [], []

    for idx, (x, reduce_axis) in enumerate(ins):
        with tbe.dsl.compute():
            disable_fuse_axes = []
            if "disable_fuse_axes" in extra_params:
                disable_fuse_axes = extra_params.get("disable_fuse_axes")[idx]
            shape_x = shape_util.variable_shape([x], op_mode="norm")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype,
                                    attrs={"ori_shape": ori_shape, "ori_format": ori_format, "format": format,
                                           "disable_fuse_axes": disable_fuse_axes})
            res = norm_base_compute(data1, reduce_axis)
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("norm_base", "norm.test_dynamic_norm_base_impl", "dsl_dync_norm_base")

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
        "test_dync_norm_base_1",
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
        "test_dync_norm_base_2",
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
        "test_dync_norm_base_3",
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
        "test_dync_norm_base_4",
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
        "test_dync_norm_base_5",
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
        "test_dync_norm_base_6",
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
        "test_dync_norm_base_7",
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
        "test_dync_norm_base_8",
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
        "test_dync_norm_base_9",
    "expect":
        "success",
    "support_expect":
        True
}

case10 = {
    "params": [{
        "shape": (6, 12, 24, 24, 16, 16),
        "dtype": "float16",
        "range": [(6, 6), (12, 12), (24, 24), (24, 24), (16, 16), (16, 16)]
    }, {
        "shape": (6, 12, 24, 24, 16, 16),
        "dtype": "float16",
        "range": [(6, 6), (12, 12), (24, 24), (24, 24), (16, 16), (16, 16)]
    }, [2, 5]],
    "case_name":
        "test_dync_norm_base_10",
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
ut_case.add_case(["Ascend910A"], case10)
