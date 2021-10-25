# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings
import operator

from te import tvm
import te.lang.cce as tbe


warnings.filterwarnings("ignore")


def dsl_vcmp(x, y, _, operation_type, mode, is_const, kernel_name='dsl_vcmp'):
    input_shape1 = x.get("shape")
    input_dtype1 = x.get("dtype")
    data1 = tvm.placeholder(input_shape1, name='data1', dtype=input_dtype1)
    if is_const:
        # data2 = tvm.const(y, dtype=input_dtype1)
        res = tbe.vcmp(data1, y, operation_type, mode)
        tensor_list = [data1, res]
    else:
        input_shape2 = y.get("shape")
        input_dtype2 = y.get("dtype")
        data2 = tvm.placeholder(input_shape2, name='data2', dtype=input_dtype2)
        res = tbe.vcmp(data1, data2, operation_type, mode)
        tensor_list = [data1, data2, res]

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "save_temp_cce_file": True
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vcmp", "vcmp.test_vcmp_impl", "dsl_vcmp")


def test_lhs_in_not_tensor(_):
    try:
        input1 = tvm.const(3, dtype="float16")
        input2 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vcmp(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_invalid_operation_type(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input2", dtype="float16")
        tbe.vcmp(input1, input2, "ee")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_invalid_mode(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((128, ), name="input2", dtype="float16")
        tbe.vcmp(input1, input2, mode="uint8")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_lhs_invalid_shape_value(_):
    try:
        input1 = tvm.placeholder((128, 9), name="input1", dtype="float16")
        input2 = tvm.placeholder((128, 9), name="input2", dtype="float16")
        tbe.vcmp(input1, input2, mode="bit")
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_dtype_not_same(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((128, ), name="input2", dtype="float32")
        tbe.vcmp(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func_list = [
    test_lhs_in_not_tensor,
    test_invalid_operation_type,
    test_invalid_mode,
    test_dtype_not_same,
    test_lhs_invalid_shape_value
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

case1 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               "eq",
               "bool",
               False
               ],
    "case_name": "test_vcmp_bool_tensor_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               "gt",
               "bool",
               False
               ],
    "case_name": "test_vcmp_bool_tensor_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float32", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float32", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float32", "format": "ND"},
               "lt",
               "bool",
               False
               ],
    "case_name": "test_vcmp_bool_tensor_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               "le",
               "bool",
               False
               ],
    "case_name": "test_vcmp_bool_tensor_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               "ge",
               "bool",
               False
               ],
    "case_name": "test_vcmp_bool_tensor_5",
    "expect": "success",
    "support_expect": True
}

case6 = {
    "params": [{"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
               "ne",
               "bool",
               False
               ],
    "case_name": "test_vcmp_bool_tensor_6",
    "expect": "success",
    "support_expect": True
}

case7 = {
    "params": [{"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               3,
               {"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               "eq",
               "bool",
               True
               ],
    "case_name": "test_vcmp_bool_const_7",
    "expect": "success",
    "support_expect": True
}

case8 = {
    "params": [{"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               3,
               {"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               "ne",
               "bool",
               True
               ],
    "case_name": "test_vcmp_bool_const_8",
    "expect": "success",
    "support_expect": True
}

case9 = {
    "params": [{"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               3,
               {"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               "lt",
               "bool",
               True
               ],
    "case_name": "test_vcmp_bool_const_9",
    "expect": "success",
    "support_expect": True
}

case10 = {
    "params": [{"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               3,
               {"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               "gt",
               "bool",
               True
               ],
    "case_name": "test_vcmp_bool_const_10",
    "expect": "success",
    "support_expect": True
}

case11 = {
    "params": [{"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               3,
               {"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               "le",
               "bool",
               True
               ],
    "case_name": "test_vcmp_bool_const_11",
    "expect": "success",
    "support_expect": True
}

case12 = {
    "params": [{"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               3,
               {"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               "ge",
               "bool",
               True
               ],
    "case_name": "test_vcmp_bool_const_12",
    "expect": "success",
    "support_expect": True
}

case13 = {
    "params": [{"shape": (3000, 8), "dtype": "float16", "format": "ND"},
               {"shape": (3000, 8), "dtype": "float16", "format": "ND"},
               {"shape": (3000, 1), "dtype": "float16", "format": "ND"},
               "ge",
               "bit",
               False
               ],
    "case_name": "test_vcmp_bit_tensor_13",
    "expect": "success",
    "support_expect": True
}

compile_case = [
    case1,
    case2,
    case3,
    case4,
    case5,
    case6,
    case7,
    case8,
    case9,
    case10,
    case11,
    case12,
    case13,
]
for item in compile_case:
    ut_case.add_case(case=item)



if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
