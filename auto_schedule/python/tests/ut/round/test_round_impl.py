# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_round(x, _, kernel_name='dsl_round'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.round(data1)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "save_temp_cce_file": True
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("round", "round.test_round_impl", "dsl_round")


def test_shape_value_less_then_zero(_):
    try:
        input1 = tvm.placeholder((0,), name="input1", dtype="float16")
        tbe.round(input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_shape_value_less_then_zero,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {"params": [{"shape": (2, 282, 282, 128), "dtype": "float16", "format": "ND"},
                    {"shape": (2, 282, 282, 128), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_round_f16_int32",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (17, 11, 19, 23, 7, 3), "dtype": "float32", "format": "ND"},
                    {"shape": (17, 11, 19, 23, 7, 3), "dtype": "float32", "format": "ND"}
                    ],
         "case_name": "test_round_f32_int32",
         "expect": "success",
         "support_expect": True
         }


compile_case_list = [
    case1,
    case2,
]
for item in compile_case_list:
    ut_case.add_case(case=item)


def calc_expect_func(x, _):
    x_value = x.get("value")
    output = np.round(x_value).astype("int32")
    return output


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (2, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (2, ), "dtype": "int32", "param_type": "output"},
                   ],
        "case_name": "test_round_precision_yolo3",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, ), "dtype": "int32", "param_type": "output"},
                   ],
        "case_name": "test_round_precision_GNMT",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1000, 1), "dtype": "float16", "param_type": "input"},
                   {"shape": (1000, 1), "dtype": "int32", "param_type": "output"},
                   ],
        "case_name": "test_round_precision_fpn_faster_rcnn_383",
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
