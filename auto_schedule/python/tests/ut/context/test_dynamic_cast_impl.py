# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe import tvm
from tbe.dsl.compute import cast
from tbe.dsl.compute.cast import _para_check_of_cast
from tbe.common.platform import intrinsic_check_support

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_cast_impl", "dsl_cast")


def test_cast_para_check_of_cast_dynamic(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        lhs_shape = (16, 64, 1, 8)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
        cast.ceil(lhs)
        return True
    return False


def test_cast_para_check_of_cast_tensor(_):
    lhs_shape = (16, 64, 1, 8)
    try:
        cast.ceil(lhs_shape)
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_cast_ceil(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast.ceil(lhs)
    return True


def test_cast_floor(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast.floor(lhs)
    return True


def test_cast_round(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast.round(lhs)
    return True


def test_cast_trunc(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast.trunc(lhs)
    return True


def test_cast_trunc_cast_type_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
        cast.trunc(lhs)
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90002":
            return True
    return False


def test_cast_round_half_up(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast.round_half_up(lhs)
    return True


def test_cast_round_half_up_cast_type_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
        cast.round_half_up(lhs)
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90002":
            return True
    return False


def test_cast_op_for_davinci_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
        cast._cast_op(lhs, "float16", "elewise_single_cast_x")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90003":
            return True
    return False


def test_cast_op_for_davinci_not_auto_cast(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
    cast._cast_op(lhs, "float16", "elewise_single_cast", False)
    return True


def test_cast_op_for_protogenes(_):
    opTypes = ["elewise_single_round", "elewise_single_ceil", "elewise_single_floor", "elewise_single_trunc"]
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
    for opType in opTypes:
        cast._cast_op(lhs, "float16", opType)
    return True


def test_cast_op_for_protogenes_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
        cast._cast_op(lhs, "float16", "elewise_single_cast_x")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90003":
            return True
    return False


def test_cast_cast_for_same_dtype(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    tensor = cast._cast(lhs, "float16")
    if tensor == lhs:
        return True


def test_cast_cast_normal(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast._cast(lhs, "int32")
    return True


def test_cast_cast_type_float16(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
    cast._cast(lhs, "float32")
    return True


def test_cast_cast_type_float32(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int32")
    cast._cast(lhs, "int8")
    return True


def test_cast_cast_type_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
        cast._cast(lhs, "int32")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90002":
            return True
    return False


def test_cast_to_not_tensor_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        cast.cast_to(lhs_shape, "int32")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_cast_to_normal(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="int8")
    cast.cast_to(lhs, "int32")
    return True


def test_cast_to_f322s32z(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    cast.cast_to(lhs, "int32")
    return True


def test_cast_to_f162s32z(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast.cast_to(lhs, "int32")
    return True


def test_cast_to_not_f1628IntegerFlag(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    cast.cast_to(lhs, "int8")
    return True


def test_cast_to_same_dtype(_):
    lhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    tensor = cast.cast_to(lhs, "float16")
    if tensor == lhs:
        return True
    return False


def test_cast_para_check_input_not_tensor_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        cast.ceil(lhs_shape, "int32")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_cast_to_round_exception(_):
    try:
        lhs_shape = (16, 64, 1, 8)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
        cast.cast_to_round(lhs_shape, "int8")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


test_func_list_Ascend910 = [
    test_cast_para_check_of_cast_dynamic,
    test_cast_para_check_of_cast_tensor,
    test_cast_ceil,
    test_cast_floor,
    test_cast_round,
    test_cast_trunc,
    test_cast_trunc_cast_type_exception,
    test_cast_round_half_up,
    test_cast_round_half_up_cast_type_exception,
    test_cast_op_for_davinci_exception,
    test_cast_op_for_davinci_not_auto_cast,
    test_cast_op_for_protogenes,
    test_cast_op_for_protogenes_exception,
    test_cast_cast_for_same_dtype,
    test_cast_cast_normal,
    test_cast_cast_type_float16,
    test_cast_cast_type_float32,
    test_cast_to_not_tensor_exception,
    test_cast_to_normal,
    test_cast_para_check_input_not_tensor_exception,
    test_cast_to_round_exception
]

test_func_list_Ascend310 = [
    test_cast_cast_type_exception,
    test_cast_to_f322s32z,
    test_cast_to_f162s32z,
    test_cast_to_not_f1628IntegerFlag,
    test_cast_to_same_dtype,
    test_cast_para_check_input_not_tensor_exception,
    test_cast_to_round_exception
    ]
for item in test_func_list_Ascend910:
    ut_case.add_cust_test_func(["Ascend910A"], test_func=item)

for item in test_func_list_Ascend310:
    ut_case.add_cust_test_func(["Ascend310"], test_func=item)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A", "Ascend310A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
