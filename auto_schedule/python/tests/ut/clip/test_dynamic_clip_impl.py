# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("clip")
def dsl_dynamic_clip(x, y, max_value, min_value, kernel_name="dsl_dynamic_clip"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x], "elewise")
    schedules, tensors = [], []

    for (x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x])[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.clip(data1, max_value, min_value)

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("clip", "clip.test_dynamic_clip_impl", "dsl_dynamic_clip")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, 1.0, 0],
    "case_name":
        "test_dync_clip_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (2, -1),
        "dtype": "float16",
        "range": [(2, 2), (1, None)]
    }, {
        "shape": (2, -1),
        "dtype": "float16",
        "range": [(2, 2), (1, None)]
    }, 2.0, 0.5],
    "case_name":
        "test_dync_clip_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (-1, 32, 199),
        "dtype": "int32",
        "range": [(1, None), (32, 32), (199, 199)]
    }, {
        "shape": (-1, 32, 199),
        "dtype": "int32",
        "range": [(1, None), (32, 32), (199, 199)]
    }, 2, 0],
    "case_name":
        "test_dync_clip_3",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend310"], case3)


def calc_expect_func(x, y, max_value, min_value):
    x_value = x.get("value")
    res = np.clip(x_value, min_value, max_value)
    return (res, )


ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        3.0, 0.5],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_clip_prec_01"
    })

ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        2.0, -2.0],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_clip_prec_02"
    })
