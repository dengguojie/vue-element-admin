# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("vmins")
def dsl_dync_vmins(x, y, value, kernel_name="dsl_dync_vmins"):
    input_dtype = x.get("dtype")

    ins = classify([x], "elewise")
    schedules, tensors = [], []

    for (x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x])[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.vmins(data1, tvm.const(value,dtype=input_dtype))

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vmins", "vmins.test_dynamic_vmins_impl", "dsl_dync_vmins")

case1 = {
    "params": [{
        "shape": (5, -1, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16)]
    }, {
        "shape": (5, -1, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16)]
    }, 2.0],
    "case_name":
        "test_dync_vmins_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-1, 55),
        "dtype": "float32",
        "range": [(1, 100), (55, 55)]
    }, {
        "shape": (-1, 55),
        "dtype": "float32",
        "range": [(1, 100), (55, 55)]
    }, 2.0],
    "case_name":
        "test_dync_vmins_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (12, -1, 13),
        "dtype": "int32",
        "range": [(12, 12), (1, 100), (13, 13)]
    }, {
        "shape": (12, -1, 13),
        "dtype": "int32",
        "range": [(12, 12), (1, 100), (13, 13)]
    }, 3],
    "case_name":
        "test_dync_vmins_3",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend310"], case3)


def calc_expect_func(x, y, value):
    x_value = x.get("value")
    res = np.minimum(x_value, value)
    return (res, )


ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            }, 2.0],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001),
        "case_name":
            "test_dync_vmins_prec_01"
    })


ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            }, 3.0],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name":
            "test_dync_vmins_prec_02"
    })

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (2, -1),
                "dtype": "int32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "int32",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            }, 3],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001),
        "case_name":
            "test_dync_vmins_prec_03"
    })
