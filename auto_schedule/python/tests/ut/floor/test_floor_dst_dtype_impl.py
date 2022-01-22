# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import tbe

from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.common.context.op_info import OpInfo

warnings.filterwarnings("ignore")


@register_operator("dsl_floor")
def dsl_floor(x, y, _, kernel_name='dsl_floor'):
    with tbe.common.context.op_context.OpContext("static") as f:
        f.add_op_info(OpInfo("round", "round"))
        input_dtype = x.get("dtype")
        output_dtype = y.get("dtype")
        ins = tbe.dsl.classify([x], "elewise")
        schedules, tensors = [], []

        for (x,) in ins:
            with tbe.dsl.compute():
                shape_x = shape_util.variable_shape([x])[0]
                data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
                res = tbe.dsl.floor(data1, dtype=output_dtype)

                tensors.append((data1, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(schedules, config)


ut_case = OpUT("floor", "floor.test_floor_dst_dtype_impl", "dsl_floor")


def test_shape_value_less_then_zero(_):
    try:
        input1 = tvm.placeholder((0,), name="input1", dtype="float32")
        tbe.dsl.floor(input1, dtype="int64")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_shape_value_less_then_zero,
]
for item in test_func_list:
    ut_case.add_cust_test_func(support_soc=["Ascend920A"], test_func=item)

case1 = {"params": [{"shape": (2, 282, 282, 128), "dtype": "float32", "format": "ND"},
                    {"shape": (2, 282, 282, 128), "dtype": "bfloat16", "format": "ND"},
                    "bfloat16"
                    ],
         "case_name": "test_floor_f32_bf16",
         "expect": "success",
         "support_expect": True
         }


case2 = {"params": [{"shape": (2, 282, 282, 128), "dtype": "int64", "format": "ND"},
                    {"shape": (2, 282, 282, 128), "dtype": "float32", "format": "ND"},
                    "float32"
                    ],
         "case_name": "test_floor_s64_f32",
         "expect": "success",
         "support_expect": True
         }

case3 = {"params": [{"shape": (2, 282, 282, 128), "dtype": "float32", "format": "ND"},
                    {"shape": (2, 282, 282, 128), "dtype": "int64", "format": "ND"},
                    "int64"
                    ],
         "case_name": "test_floor_f32_s64",
         "expect": "success",
         "support_expect": True
         }


case4 = {"params": [{"shape": (17, 11, 19, 23, 7, 3), "dtype": "bfloat16", "format": "ND"},
                    {"shape": (17, 11, 19, 23, 7, 3), "dtype": "int32", "format": "ND"},
                    "int32"
                    ],
         "case_name": "test_floor_bf16_int32",
         "expect": "success",
         "support_expect": True
         }


compile_case_list = [
    #case1,
    case2,
    case3,
    #case4

]
for item in compile_case_list:
    ut_case.add_case(case=item, support_soc="Ascend920A")

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend920A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
