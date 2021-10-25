# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vmuls(x, _, scalar, is_tvm_const, kernel_name='dsl_vmuls'):
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
    if is_tvm_const:
        data2 = tvm.const(scalar, dtype=input_dtype)
        res = tbe.vmuls(data1, data2)
    else:
        res = tbe.vmuls(data1, scalar)

    tensor_list = [data1, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vmuls", "vmuls.test_vmuls_impl", "dsl_vmuls")


def test_scalar_is_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input2", dtype="float16")
        tbe.vmuls(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_scalar_is_tensor, None]
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    2.0,
                    False
                    ],
         "case_name": "test_vmuls_scalar_is_float",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND"},
                    {"shape": (30000, 1), "dtype": "float32", "format": "ND"},
                    1,
                    False
                    ],
         "case_name": "test_vmuls_scalar_is_int",
         "expect": "success",
         "support_expect": True
         }

case3 = {"params": [{"shape": (128,), "dtype": "float32", "format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND"},
                    1,
                    True
                    ],
         "case_name": "test_vmuls_scalar_is_const",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, None],
    "2": [case2, None],
    "3": [case3, None],
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, _, scalar, __):
    x_value = x.get("value")
    output = np.multiply(x_value, scalar)
    return output


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 4), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4), "dtype": "float16", "param_type": "output"},
                   2.0,
                   False
                   ],
        "case_name": "test_vmuls_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 4, 256), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4, 256), "dtype": "float16", "param_type": "output"},
                   1,
                   False
                   ],
        "case_name": "test_vmuls_precision_2",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 4, 4, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (4, 4, 4, 32), "dtype": "float16", "param_type": "output"},
                   -1,
                   False
                   ],
        "case_name": "test_vmuls_precision_3",
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
