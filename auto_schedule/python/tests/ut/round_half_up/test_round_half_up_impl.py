# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from te.platform.cce_conf import te_set_version

warnings.filterwarnings("ignore")


def dsl_round_half_up(x, _, kernel_name='dsl_round_half_up'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    res = tbe.round_half_up(data1)

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


ut_case = OpUT("round_half_up", "round_half_up.test_round_half_up_impl", "dsl_round_half_up")


def test_shape_value_less_then_zero(_):
    try:
        input1 = tvm.placeholder((0,), name="input1", dtype="float16")
        tbe.round_half_up(input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_not_support_intrinsic_vconv_2s32a(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.round_half_up(input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True

test_func = {
    "1": [test_shape_value_less_then_zero, "Ascend910A"],
    "2": [test_not_support_intrinsic_vconv_2s32a, "Ascend310"]
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {"params": [{"shape": (2, 282, 282, 128), "dtype": "float16", "format": "ND"},
                    {"shape": (2, 282, 282, 128), "dtype": "float16", "format": "ND"}
                    ],
         "case_name": "test_round_half_up_1",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (17, 11, 19, 23, 7, 3), "dtype": "float32", "format": "ND"},
                    {"shape": (17, 11, 19, 23, 7, 3), "dtype": "float32", "format": "ND"}
                    ],
         "case_name": "test_round_half_up_2",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, "Ascend910A"],
    "2": [case2, "Ascend910A"]
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, _):
    x_value = x.get("value")
    output = np.where(np.subtract(x_value, np.trunc(x_value)) >= 0.5,
                      np.ceil(x_value), np.floor(x_value)).astype(np.int32)
    return output


ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (2, ), "dtype": "float16", "param_type": "input", "value_range": [0.5, 1]},
                   {"shape": (2, ), "dtype": "int32", "param_type": "output"},
                   ],
        "case_name": "test_round_half_up_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (1, ), "dtype": "float16", "param_type": "input", "value_range": [0.5, 1]},
                   {"shape": (1, ), "dtype": "int32", "param_type": "output"},
                   ],
        "case_name": "test_round_half_up_precision_2",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (1000, 1), "dtype": "float16", "param_type": "input", "value_range": [0.5, 1]},
                   {"shape": (1000, 1), "dtype": "int32", "param_type": "output"},
                   ],
        "case_name": "test_round_half_up_precision_3",
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
