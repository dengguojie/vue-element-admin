# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def dsl_vsel(condition, x, y, _, kernel_name='dsl_vsel'):
    condition_shape = condition.get("shape")
    condition_dtype = condition.get("dtype")
    condition_input = tvm.placeholder(condition_shape, name='condition', dtype=condition_dtype)
    tensor_list = [condition_input]

    if not isinstance(x, dict):
        input_value = x
        data1 = tvm.const(input_value, dtype="float16")
    else:
        input_dtype = x.get("dtype")
        input_shape = x.get("shape")
        data1 = tvm.placeholder(input_shape, name='data1', dtype=input_dtype)
        tensor_list.append(data1)

    if not isinstance(y, dict):
        input2_value = y
        data2 = tvm.const(input2_value, dtype="float16")
    else:
        input2_shape = y.get("shape")
        input2_dtype = y.get("dtype")
        data2 = tvm.placeholder(input2_shape, name='data2', dtype=input2_dtype)
        tensor_list.append(data2)

    res = tbe.vsel(condition_input, data1, data2)
    tensor_list.append(res)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "save_temp_cce_file": True
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vsel", "vsel.test_vsel_impl", "dsl_vsel")


def test_condition_is_not_tensor(_):
    try:
        condition_input = tvm.const(2, dtype="bool")
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input2", dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_condition_is_not_valid(_):
    """
    condition dtype in [bool, uint8]
    """
    try:
        condition_input = tvm.placeholder((128,), name="condition", dtype="float16")
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input2", dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_lhs_is_tvm_const(_):
    try:
        condition_input = tvm.placeholder((128,), name="condition", dtype="bool")
        input1 = tvm.const(128, dtype="float32")
        input2 = tvm.const(128, dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_rhs_is_tvm_const(_):
    try:
        condition_input = tvm.placeholder((128,), name="condition", dtype="bool")
        input1 = tvm.const(128, dtype="float16")
        input2 = tvm.const(128, dtype="float32")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_input_is_tvm_const_value_is_same(_):
    try:
        condition_input = tvm.placeholder((128,), name="condition", dtype="bool")
        input1 = tvm.const(128, dtype="float16")
        input2 = tvm.const(128, dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_condition_rhs_dim_is_not_same(_):
    try:
        condition_input = tvm.placeholder((128,), name="condition", dtype="bool")
        input1 = tvm.const(128, dtype="float16")
        input2 = tvm.placeholder((128, 1), name="input2", dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_condition_lhs_dim_is_not_same(_):
    try:
        condition_input = tvm.placeholder((128,), name="condition", dtype="bool")
        input1 = tvm.placeholder((128, 1), name="input1", dtype="float16")
        input2 = tvm.const(128, dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_condition_rhs_shape_is_not_same(_):
    try:
        condition_input = tvm.placeholder((128, 2), name="condition", dtype="bool")
        input1 = tvm.const(128, dtype="float16")
        input2 = tvm.placeholder((128, 1), name="input2", dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_condition_lhs_shape_is_not_same(_):
    try:
        condition_input = tvm.placeholder((128, 3), name="condition", dtype="bool")
        input1 = tvm.placeholder((128, 1), name="input1", dtype="float16")
        input2 = tvm.const(128, dtype="float16")
        tbe.vsel(condition_input, input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


test_func = {
    "1": [test_condition_is_not_tensor, None],
    "2": [test_condition_is_not_valid, None],
    "3": [test_lhs_is_tvm_const, None],
    "4": [test_rhs_is_tvm_const, None],
    "5": [test_input_is_tvm_const_value_is_same, None],
    "6": [test_condition_lhs_dim_is_not_same, None],
    "7": [test_condition_rhs_dim_is_not_same, None],
    "8": [test_condition_rhs_shape_is_not_same, None],
    "9": [test_condition_lhs_shape_is_not_same, None],
}
for _, item in test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])


case1 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "bool", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_bool_tensor_tensor",
         "expect": "success",
         "support_expect": True
         }

case2 = {"params": [{"shape": (5, 8, 16, 16), "dtype": "bool", "format": "ND"},
                    5,
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (5, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_bool_scalar_tensor",
         "expect": "success",
         "support_expect": True
         }

case3 = {"params": [{"shape": (30000, 1), "dtype": "bool", "format": "ND"},
                    {"shape": (30000, 1), "dtype": "float16", "format": "ND"},
                    5,
                    {"shape": (30000, 1), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_bool_tensor_scalar",
         "expect": "success",
         "support_expect": True
         }

case4 = {"params": [{"shape": (128,), "dtype": "bool", "format": "ND"},
                    5,
                    6,
                    {"shape": (128,), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_bool_scalar_scalar",
         "expect": "success",
         "support_expect": True
         }

case5 = {"params": [{"shape": (64, 8, 16, 2), "dtype": "uint8", "format": "ND"},
                    {"shape": (64, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (64, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (64, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_uint8_tensor_tensor",
         "expect": "success",
         "support_expect": True
         }

case6 = {"params": [{"shape": (8, 8, 16, 2), "dtype": "uint8", "format": "ND"},
                    5,
                    {"shape": (8, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    {"shape": (8, 8, 16, 16), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_uint8_scalar_tensor",
         "expect": "success",
         "support_expect": True
         }

case7 = {"params": [{"shape": (8, 1), "dtype": "uint8", "format": "ND"},
                    {"shape": (8, 8), "dtype": "float16", "format": "ND"},
                    5,
                    {"shape": (8, 8), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_uint8_tensor_scalar",
         "expect": "success",
         "support_expect": True
         }

case8 = {"params": [{"shape": (128 // 8, ), "dtype": "uint8", "format": "ND"},
                    5,
                    6,
                    {"shape": (128, ), "dtype": "float16", "format": "ND"},
                    ],
         "case_name": "test_vsel_uint8_scalar_scalar",
         "expect": "success",
         "support_expect": True
         }

compile_case = {
    "1": [case1, None],
    "2": [case2, None],
    "3": [case3, None],
    "4": [case4, None],
    "5": [case5, None],
    "6": [case6, None],
    "7": [case7, None],
    "8": [case8, None],
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(condition, x, y, _, ):
    condition_dtype = condition.get("dtype")
    condition_value = condition.get("value")
    benchmark_data = []
    if isinstance(x, dict):
        x_value = x.get("value")
    else:
        x_value = np.full(condition.get("shape"), x)
    if isinstance(y, dict):
        y_value = y.get("value")
    else:
        y_value = np.full(condition.get("shape"), y)

    if condition_dtype == "bool":
        for i, cond in enumerate(condition_value):
            if cond:
                benchmark_data.append(x_value[i])
            else:
                benchmark_data.append(y_value[i])
    if condition_dtype == "uint8":
        for i, cond in enumerate(condition_value):
            repeat_num = 0
            while repeat_num < 8:
                if cond:
                    benchmark_data.append(x_value[i * 8 + repeat_num])
                else:
                    benchmark_data.append(y_value[i * 8 + repeat_num])
    res = np.array(benchmark_data)
    return res


ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (32, ), "dtype": "bool", "param_type": "input"},
                   {"shape": (32, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (32, ), "dtype": "float16", "param_type": "input"},
                   {"shape": (32, ), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vsel_precision_bool_tensor_tensor",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "all", {
        "params": [{"shape": (32, ), "dtype": "bool", "param_type": "input"},
                   5,
                   6,
                   {"shape": (32, ), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vsel_precision_bool_scalar_scalar",
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
