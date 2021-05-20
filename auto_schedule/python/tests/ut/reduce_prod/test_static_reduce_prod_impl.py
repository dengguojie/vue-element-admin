# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import tbe
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.common.context import op_info

warnings.filterwarnings("ignore")

@register_operator("dsl_static_reduce_prod")
def dsl_static_reduce_prod(x, y, axis, keepdims, kernel_name="dsl_static_reduce_prod"):
    with tbe.common.context.op_context.OpContext("static") as f:
        opInfo = op_info.OpInfo("reduce_prod1", "reduce_prod")
        f.add_op_info(opInfo)

        input_dtype = x.get("dtype")
        x["rel_pos_to_reduce"] = 'before'
        input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
        ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})
        schedules, tensors = [], []

        for (x, axis) in ins:
            with tbe.dsl.compute():
                shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
                data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
                res = tbe.dsl.reduce_prod(data1, axis.get("value"), keepdims)
                tensors.append([data1, res])

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(schedules, config)



ut_case = OpUT("reduce_prod", "reduce_prod.test_static_reduce_prod_impl", "dsl_static_reduce_prod")


def test_axis_in_none(_):
    try:
        input1 = tvm.placeholder((1, 128), name="input1", dtype="float16")
        tbe.dsl.reduce_prod(input1, [None])
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_axis_in_none
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)


case1 = {
    "params": [{"shape": (5, 1024), "dtype": "float16", "format": "ND"},
               {"shape": (5, ), "dtype": "float16", "format": "ND"},
               [-1],
               False
               ],
    "case_name": "test_reduce_prod_int32_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [{"shape": (32, 32, 9919910), "dtype": "int32", "format": "ND"},
               {"shape": (32, 9919910), "dtype": "int32", "format": "ND"},
               [1],
               False
               ],
    "case_name": "test_reduce_prod_int32_2",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1,
    case2,
]
for item in compile_case_list:
    ut_case.add_case(case=item)


def calc_expect_func(x, _, axis, keep_dims):
    x_value = x.get("value")
    res = np.prod(x_value, axis=axis, keepdims=keep_dims)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (7, 5, 13, 8, 7), "dtype": "int32", "param_type": "input"},
                   {"shape": (7, 5, 13, 8), "dtype": "int32", "param_type": "output"},
                   (4, ),
                   False
                   ],
        "case_name": "test_reduce_prod_precision_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (16, 128), "dtype": "int32", "param_type": "input"},
                   {"shape": (16,), "dtype": "int32", "param_type": "output"},
                   (1, ),
                   False
                   ],
        "case_name": "test_reduce_prod_precision_int32",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
