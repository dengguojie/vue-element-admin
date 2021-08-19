# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("trunc")
def dsl_dynamic_trunc(x, y, kernel_name="dsl_dynamic_trunc"):
    input_dtype = x.get("dtype")
    output_dtype = y.get("dtype")

    ins = tbe.dsl.classify([x], "elewise")
    schedules, tensors = [], []

    for (x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x])[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.trunc(data1, output_dtype)

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("trunc", "trunc.test_dynamic_trunc_dst_dtype_impl", "dsl_dynamic_trunc")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "bfloat16",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_trunc_dst_f12_bf16",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (2, -1),
        "dtype": "int64",
        "range": [(2, 2), (1, None)]
    }, {
        "shape": (2, -1),
        "dtype": "float32",
        "range": [(2, 2), (1, None)]
    }],
    "case_name":
        "test_dync_trunc_dst_s64_f32",
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
        "dtype": "int64",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_trunc_dst_f32_s64",
    "expect":
        "success",
    "support_expect":
        True
}


case4 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "bfloat16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_trunc_dst_bf16_s32",
    "expect":
        "success",
    "support_expect":
        True
}


compile_case_list = [
    case1,
    case2,
    case3,
    case4

]
for item in compile_case_list:
    ut_case.add_case(case=item, support_soc="Ascend920A")



