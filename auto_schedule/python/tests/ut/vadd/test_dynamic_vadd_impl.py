# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("vadd")
def dsl_dync_vadd(x, y, z, kernel_name="dsl_dync_vadd"):
    input_dtype = x.get("dtype")

    ins = classify([x, y], "elewise")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            res = tbe.dsl.vadd(data1, data2)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vadd", "vadd.test_dynamic_vadd_impl", "dsl_dync_vadd")

case1 = {
    "params": [{
        "shape": (5, -1, 16, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16), (16, 16)]
    }, {
        "shape": (5, -1, 16, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16), (16, 16)]
    }, {
        "shape": (5, -1, 16, 16),
        "dtype": "float16",
        "range": [(5, 5), (1, 10), (16, 16), (16, 16)]
    }],
    "case_name":
        "test_dync_vadd_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (30000, -1),
        "dtype": "float32",
        "range": [(30000, 30000), [1, 100]]
    }, {
        "shape": (30000, -1),
        "dtype": "float32",
        "range": [(30000, 30000), [1, 100]]
    }, {
        "shape": (30000, -1),
        "dtype": "float32",
        "range": [(30000, 30000), [1, 100]]
    }],
    "case_name":
        "test_dync_vadd_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (30000,),
        "dtype": "float32",
        "range": [(30000, 30000),]
    }, {
        "shape": (30000,),
        "dtype": "float32",
        "range": [(30000, 30000),]
    }, {
        "shape": (30000,),
        "dtype": "float32",
        "range": [(30000, 30000),]
    }],
    "case_name":
        "test_dync_vadd_3",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "shape": (-1,),
        "dtype": "float16",
        "range": [(0, 0),]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "range": [(0, 0),]
    }, {
        "shape": (-1,),
        "dtype": "float16",
        "range": [(0, 0),]
    }],
    "case_name":
        "test_dync_vadd_4",
    "expect":
        "success",
    "support_expect":
        True
}

case5 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    }, {
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    }, {
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    }],
    "case_name":
        "test_dync_vadd_5",
    "expect":
        "success",
    "support_expect":
        True
}

case6 = {
    "params": [{
        "shape": (-2, 2),
        "dtype": "float16",
        "range": [(1, None), (2, 2),]
    }, {
        "shape": (-2, 2),
        "dtype": "float16",
        "range": [(1, None), (2, 2),]
    }, {
        "shape": (-2, 2),
        "dtype": "float16",
        "range": [(1, None), (2, 2),]
    }],
    "case_name":
        "test_dync_vadd_6",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

ut_case.add_case(["all",], case1)
ut_case.add_case(["all",], case2)
ut_case.add_case(["all",], case3)
ut_case.add_case(["all",], case4)
ut_case.add_case(["all",], case5)
ut_case.add_case(["all",], case6)

def calc_expect_func(x, y, z):
    x_value = x.get("value")
    y_value = y.get("value")
    res = np.add(x_value, y_value)
    return res,


ut_case.add_precision_case(
    "all", {
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
                "param_type": "input"
            },
            {
                "shape": (2, -1),
                "dtype": "float16",
                "range": [(2, 2), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001),
        "case_name":
            "test_dync_vadd_prec_01"
    })
