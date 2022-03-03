# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vor")
def dsl_dynamic_vor(x, y, z, kernel_name="dsl_dynamic_vor"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "elewise")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            res = tbe.dsl.vor(data1, data2)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vor", "vor.test_dynamic_vor_impl", "dsl_dynamic_vor")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "uint16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint16",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_vor_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int16",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_vor_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_vor_int32",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "uint32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint32",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_vor_uint32",
    "expect":
        "success",
    "support_expect":
        True
}

case5 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "int32", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "int32", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "int32", "format": "ND"}
                    ],
         "case_name": "test_vor_int32",
         "expect": "success",
         "support_expect": True
         }

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)


def calc_expect_func(x, y, z):
    x_value = x.get("value")
    y_value = y.get("value")
    res = np.bitwise_or(x_value, y_value)
    return (res, )


ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "int16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "int16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "int16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vor_prec_01"
    })

ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "uint16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "uint16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "uint16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vor_prec_02"
    })


ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "int32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "int32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "int32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vor_prec_int32"
    })

ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "uint32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "uint32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "uint32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vor_prec_uint32"
    })