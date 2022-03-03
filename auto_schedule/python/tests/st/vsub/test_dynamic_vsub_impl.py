# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

from tbe import tvm
import tbe
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vsub")
def dsl_dynamic_vsub(x, y, z, kernel_name="dsl_dynamic_vsub"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "broadcast")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)

            shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
            input1 = tbe.dsl.broadcast(data1, shape_max)
            input2 = tbe.dsl.broadcast(data2, shape_max)
            res = tbe.dsl.vsub(input1, input2)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vsub", "vsub.test_dynamic_vsub_impl", "dsl_dynamic_vsub")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
    "test_dync_vsub_1",
    "expect":
    "success",
    "support_expect":
    True
}

ut_case.add_case(["Ascend910", "Ascend710"], case1)


def calc_expect_func(x, y, z):
    x_value = x.get("value")
    y_value = y.get("value")
    res = np.subtract(x_value, y_value)
    return (res, )


ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 10),
                "param_type": "output"
            },
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vsub_prec_01"
    })
