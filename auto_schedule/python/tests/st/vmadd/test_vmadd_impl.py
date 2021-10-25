# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe


warnings.filterwarnings("ignore")


def dsl_vmadd(x, y, z, _, kernel_name='dsl_vmadd'):
    input_shape1 = x.get("shape")
    input_dtype1 = x.get("dtype")
    input_shape2 = y.get("shape")
    input_dtype2 = y.get("dtype")
    input_shape3 = z.get("shape")
    input_dtype3 = z.get("dtype")
    data1 = tvm.placeholder(input_shape1, name='data1', dtype=input_dtype1)
    data2 = tvm.placeholder(input_shape2, name='data2', dtype=input_dtype2)
    data3 = tvm.placeholder(input_shape3, name='data3', dtype=input_dtype3)
    res = tbe.vmadd(data1, data2, data3)

    tensor_list = [data1, data2, data3, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vmadd", "vmadd.test_vmadd_impl", "dsl_vmadd")


def test_tensor1_in_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.const(3, dtype="float16")
        input3 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vmadd(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensor2_in_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input1", dtype="float16")
        input3 = tvm.const(3, dtype="float16")
        tbe.vmadd(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensor0_shape_len_not_same_tensor1_shape_len(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((128, 128), name="input1", dtype="float16")
        input3 = tvm.placeholder((128, ), name="input1", dtype="float16")
        tbe.vmadd(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensor0_shape_len_not_same_tensor2_shape_len(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input3 = tvm.placeholder((128, 128), name="input1", dtype="float16")
        tbe.vmadd(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensor_shape_not_same_other_tensor(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((16, ), name="input1", dtype="float16")
        input3 = tvm.placeholder((32, ), name="input1", dtype="float16")
        tbe.vmadd(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_tensor0_shape_value_less_than_zero(_):
    try:
        input1 = tvm.placeholder((-1, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((-1, ), name="input1", dtype="float16")
        input3 = tvm.placeholder((-1, ), name="input1", dtype="float16")
        tbe.vmadd(input1, input2, input3)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_tensor1_in_not_tensor,
    test_tensor2_in_not_tensor,
    test_tensor0_shape_len_not_same_tensor1_shape_len,
    test_tensor0_shape_len_not_same_tensor2_shape_len,
    test_tensor_shape_not_same_other_tensor,
    test_tensor0_shape_value_less_than_zero,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"}
               ],
    "case_name": "test_vmadd_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (30000, 1), "dtype": "float32", "format": "ND"},
               {"shape": (30000, 1), "dtype": "float32", "format": "ND"},
               {"shape": (30000, 1), "dtype": "float32", "format": "ND"},
               {"shape": (30000, 1), "dtype": "float32", "format": "ND"}
               ],
    "case_name": "test_vmadd_2",
    "expect": "success",
    "support_expect": True
}

compile_case = [
    case1,
    case2
]
for item in compile_case:
    ut_case.add_case(case=item)


def calc_expect_func(x, y, z, _):
    x_value = x.get("value")
    y_value = y.get("value")
    z_value = z.get("value")
    res = np.add(np.multiply(x_value, z_value),  y_value)
    return (res, )


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 16), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vmadd_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (1, 1, 1, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 1, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 1, 32), "dtype": "float16", "param_type": "input"},
                   {"shape": (1, 1, 1, 32), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vmadd_precision_2",
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
