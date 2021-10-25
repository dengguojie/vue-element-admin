# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vaxpy(x, y, _, scalar, is_tvm_const, kernel_name='dsl_vaxpy'):
    input1_shape = x.get("shape")
    input1_dtype = x.get("dtype")
    input2_shape = y.get("shape")
    input2_dtype = y.get("dtype")
    data1 = tvm.placeholder(input1_shape, name='data1', dtype=input1_dtype)
    data2 = tvm.placeholder(input2_shape, name='data2', dtype=input2_dtype)
    if is_tvm_const:
        data3 = tvm.const(scalar, dtype=input1_dtype)
        res = tbe.vaxpy(data1, data2, data3)
    else:
        res = tbe.vaxpy(data1, data2, scalar)

    tensor_list = [data1, data2, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": True,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vaxpy", "vaxpy.test_vaxpy_impl", "dsl_vaxpy")


def test_scalar_is_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input2", dtype="float16")
        input3 = tvm.placeholder((128,), name="input3", dtype="float16")
        tbe.vaxpy(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_rhs_is_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.const(128, dtype="float16")
        input3 = tvm.placeholder((128,), name="input3", dtype="float16")
        tbe.vaxpy(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_dim_is_not_same(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((128, 64), name="input2", dtype="float16")
        input3 = 4
        tbe.vaxpy(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_shape_is_not_same(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((24, ), name="input2", dtype="float16")
        input3 = 4
        tbe.vaxpy(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_shape_value_less_than_zero(_):
    try:
        input1 = tvm.placeholder((-1, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((-1, ), name="input2", dtype="float16")
        input3 = 4
        tbe.vaxpy(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_scalar_is_tensor, None],
    "2": [test_rhs_is_not_tensor, None],
    "3": [test_dim_is_not_same, None],
    "4": [test_shape_is_not_same, None],
    "5": [test_shape_value_less_than_zero, None]
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    2.0,
                    False
                    ],
         "case_name": "test_vaxpy_scalar_is_float",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (30000, 1), "dtype": "float16", "format": "ND"},
                    {"shape": (30000, 1), "dtype": "float16", "format": "ND"},
                    {"shape": (30000, 1), "dtype": "float16", "format": "ND"},
                    1,
                    False
                    ],
         "case_name": "test_vaxpy_scalar_is_int",
         "expect": "success",
         "support_expect": True
         }

case3 = {"params": [{"shape": (64, ), "dtype": "float16", "format": "ND"},
                    {"shape": (64, ), "dtype": "float16", "format": "ND"},
                    {"shape": (64, ), "dtype": "float16", "format": "ND"},
                    1,
                    True
                    ],
         "case_name": "test_vaxpy_scalar_is_const",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, None],
    "2": [case2, None],
    "3": [case3, "Ascend910A"],
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, y, _, scalar, __):
    x_value = x.get("value")
    y_value = y.get("value")
    output = x_value * scalar + y_value
    return output


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 4), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4), "dtype": "float16", "param_type": "output"},
                   2.0,
                   False
                   ],
        "case_name": "test_vaxpy_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 4, 4, 256), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4, 256), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 4, 4, 256), "dtype": "float16", "param_type": "output"},
                   1,
                   False
                   ],
        "case_name": "test_vaxpy_precision_2",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (4, 4, 4, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (4, 4, 4, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (4, 4, 4, 32), "dtype": "float16", "param_type": "output"},
                   -1,
                   False
                   ],
        "case_name": "test_vaxpy_precision_3",
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
