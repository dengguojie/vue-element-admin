# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe
from te.platform.cce_conf import te_set_version


warnings.filterwarnings("ignore")


def dsl_vmod(x, y, _, kernel_name='dsl_vmod'):
    input_shape1 = x.get("shape")
    input_dtype1 = x.get("dtype")
    input_shape2 = y.get("shape")
    input_dtype2 = y.get("dtype")
    data1 = tvm.placeholder(input_shape1, name='data1', dtype=input_dtype1)
    data2 = tvm.placeholder(input_shape2, name='data2', dtype=input_dtype2)
    res = tbe.vmod(data1, data2)

    tensor_list = [data1, data2, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("vmod", "vmod.test_vmod_impl", "dsl_vmod")


def test_lhs_in_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vmod(5, input1)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_rhs_in_not_tensor(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vmod(input1, 5)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_dtype_in_not_same(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float16")
        input2 = tvm.placeholder((128,), name="input1", dtype="float32")
        tbe.vmod(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_not_support_vdiv_and_not_support_vconv_f322s32f(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float32")
        input2 = tvm.placeholder((128,), name="input1", dtype="float16")
        tbe.vmod(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_support_vdiv_and_not_support_vconv_f322s32f(soc):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="float32")
        input2 = tvm.placeholder((128,), name="input1", dtype="float32")
        te_set_version("Hi3796CV300ES")
        tbe.vmod(input1, input2)
    except RuntimeError as e:
        te_set_version(soc)
        print(e.args[0].get("detailed_cause"))
    return True


def test_support_vdiv_and_support_vconv_f322s32f(_):
    try:
        input1 = tvm.placeholder((128,), name="input1", dtype="int16")
        input2 = tvm.placeholder((128,), name="input1", dtype="int16")
        tbe.vmod(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_dim_is_not_same(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((128, 128), name="input2", dtype="float16")
        tbe.vmax(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_shape_is_not_same(_):
    try:
        input1 = tvm.placeholder((128, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((24, ), name="input2", dtype="float16")
        tbe.vmax(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


def test_shape_value_less_than_zero(_):
    try:
        input1 = tvm.placeholder((-1, ), name="input1", dtype="float16")
        input2 = tvm.placeholder((-1, ), name="input2", dtype="float16")
        tbe.vmax(input1, input2)
    except RuntimeError as e:
        print(e.args[0].get("detailed_cause"))
    return True


cust_test_func = {
    "1": [test_lhs_in_not_tensor, None],
    "2": [test_rhs_in_not_tensor, None],
    "3": [test_dtype_in_not_same, "Ascend910A"],
    "4": [test_not_support_vdiv_and_not_support_vconv_f322s32f, "Ascend310"],
    "5": [test_support_vdiv_and_support_vconv_f322s32f, "Ascend910A"],
    "6": [test_support_vdiv_and_not_support_vconv_f322s32f, None],
    "7": [test_dim_is_not_same, None],
    "8": [test_shape_is_not_same, None],
    "9": [test_shape_value_less_than_zero, None],
}
for _, item in cust_test_func.items():
    ut_case.add_cust_test_func(test_func=item[0], support_soc=item[1])

case1 = {
    "params": [{"shape": (7, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (7, 8, 16, 16), "dtype": "float16", "format": "ND"},
               {"shape": (7, 8, 16, 16), "dtype": "float16", "format": "ND"}
               ],
    "case_name": "test_vmod_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{"shape": (10000, 1), "dtype": "float32", "format": "ND"},
               {"shape": (10000, 1), "dtype": "float32", "format": "ND"},
               {"shape": (10000, 1), "dtype": "float32", "format": "ND"}],
    "case_name": "test_vmod_2",
    "expect": "success",
    "support_expect": True
}

compile_case = {
    "1": [case1, None],
    "2": [case2, "Ascend910A"],
}
for _, item in compile_case.items():
    ut_case.add_case(case=item[0], support_soc=item[1])


def calc_expect_func(x, y, _):
    x_value = x.get("value")
    y_value = y.get("value")
    res = np.mod(x_value, y_value)
    return (res, )


ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (3, 39, 80), "dtype": "float16", "param_type": "input"},
                   {"shape": (3, 39, 80), "dtype": "float16", "param_type": "input"},
                   {"shape": (3, 39, 80), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vmod_precision_1",
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{"shape": (2, 80), "dtype": "float16", "param_type": "input"},
                   {"shape": (2, 80), "dtype": "float16", "param_type": "input"},
                   {"shape": (2, 80), "dtype": "float16", "param_type": "output"},
                   ],
        "case_name": "test_vmod_precision_2",
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
