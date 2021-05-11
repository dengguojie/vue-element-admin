# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe import tvm
from tbe.dsl.base import var
from tbe.dsl.compute import math
from tbe.dsl.base import operation as operation_context

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_op_context_impl", "dsl_vsub")


def test_var_addition(_):
    var1 = var.Var("name", "bound", "float32", var.Category.NORMAL, "addition")
    addition = var1.get_addition()
    if addition == "addition":
        return True
    return False


def test_single_elewise_op_name_check(_):
    shape_src = (16, 64, 1, 8)
    input_tensor = tvm.placeholder(shape_src, name="input_tensor", dtype="float16")
    try:
        math.__single_elewise_op(input_tensor, "float16", "broadcast_single_abc")
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90003":
            return True
    return False


def test_vlogic_operation_param(_):
    lhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="bool")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="bool")
    operation = "logic_abc"
    try:
        math.vlogic(lhs, rhs, operation)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90002":
            return True
    return False


def test_vlogic_lhs_tensor(_):
    lhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)

    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
    operation = "logic_abc"

    try:
        math.vlogic(lhs_shape, rhs, operation)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vlogic_rhs_tensor(_):
    lhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="bool")
    operation = "logic_and"
    try:
        math.vlogic(lhs, rhs_shape, operation)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_binary_elewise_op_dtype(_):
    lhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float32")
    op_name = "broadcast_binary_scalar_axpy"
    try:
        math.__binary_elewise_op(lhs, rhs, op_name)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vmla_tensor_1(_):
    lhs_shape = (16, 64, 1, 8)
    mhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float32")
    try:
        math.vmla(lhs, mhs_shape, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vmadd_tensor_1(_):
    lhs_shape = (16, 64, 1, 8)
    mhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float32")
    try:
        math.vmla(lhs, mhs_shape, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsel_bit_shape_check_shape(_):
    lhs_shape = (16, 64, 1, 8)
    mhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float32")
    try:
        math.vmla(lhs, mhs_shape, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_multiple_elewise_op_support_dtype(_):
    lhs_shape = (16, 64, 1, 8)
    mhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    mhs = tvm.placeholder(mhs_shape, name="mhs", dtype="float32")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float32")
    op_name = "elewis_multi_vmlaa"
    try:
        math.__multiple_elewise_op(lhs, mhs, rhs, op_name)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90002":
            return True
    return False


def test_multiple_elewise_op_support_dtype2(_):
    lhs_shape = (16, 64, 1, 8)
    mhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    mhs = tvm.placeholder(mhs_shape, name="mhs", dtype="float32")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float32")
    op_name = "elewis_multi_vadd"
    try:
        math.__multiple_elewise_op(lhs, mhs, rhs, op_name)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90002":
            return True
    return False


def test_multiple_elewise_op_support_name(_):
    lhs_shape = (16, 64, 1, 8)
    mhs_shape = (16, 64, 1, 8)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    mhs = tvm.placeholder(mhs_shape, name="mhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
    op_name = "elewis_multi_adds"
    try:
        math.__multiple_elewise_op(lhs, mhs, rhs, op_name)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90003":
            return True
    return False


def test_vsel_bit_shape_check_len(_):
    lhs_shape = (16, 64, 1)
    rhs_shape = (16, 64, 1, 8)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
    try:
        math._vsel_bit_shape_check(lhs, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsel_bit_shape_check_condition_lastdim(_):
    lhs_shape = (16, 32, 1, 16)
    rhs_shape = (16, 32, 1, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
    try:
        math._vsel_bit_shape_check(lhs, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsel_bit_shape_check_condition_not_lastdim(_):
    lhs_shape = (16, 64, 1, 16)
    rhs_shape = (16, 32, 1, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
    try:
        math._vsel_bit_shape_check(lhs, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsel_bit_shape_check_input_shape_value_positive(_):
    lhs_shape = (16, 32, -1, 2)
    rhs_shape = (16, 32, -1, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
    rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
    try:
        math._vsel_bit_shape_check(lhs, rhs)
    except RuntimeError as e:
        errorCode = e.args[0].get("errCode")
        if errorCode == "E90001":
            return True
    return False


def test_vsel_dynamic_check_shape(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        condition = (16, 31)
        lhs_shape = (16, 32, -1, 16)
        rhs_shape = (16, 32, -1, 16)
        condition = tvm.placeholder(condition, name="condition", dtype="bool")
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
        rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
        try:
            math.vsel(condition, lhs, rhs)
        except RuntimeError as e:
            errorCode = e.args[0].get("errCode")
            if errorCode == "E90001":
                return True
    return False


def test_vsel_dynamic_check_condition_dtype(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        condition = (16, 32, -1, 16)
        lhs_shape = (16, 32, -1, 16)
        rhs_shape = (16, 32, -1, 16)
        condition = tvm.placeholder(condition, name="condition", dtype="float16")
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
        rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")
        try:
            math.vsel(condition, lhs, rhs)
        except RuntimeError as e:
            errorCode = e.args[0].get("errCode")
            if errorCode == "E90003":
                return True
    return False


def test_vcmpsel_data_shape_check_input_value(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        lhs_shape = (0, 32, -32, 16)
        rhs_shape = (0, 32, -32, 16)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float16")
        rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")

        try:
            math._vcmpsel_data_shape_check(lhs, rhs)
        except RuntimeError as e:
            errorCode = e.args[0].get("errCode")
            if errorCode == "E90003":
                return True
    return False


def test__vcmpsel_data_dtype_check(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        lhs_shape = (16, 32, 32, 16)
        rhs_shape = (16, 32, 32, 16)
        lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
        rhs = tvm.placeholder(rhs_shape, name="rhs", dtype="float16")

        try:
            math._vcmpsel_data_dtype_check(lhs, rhs)
        except RuntimeError as e:
            errorCode = e.args[0].get("errCode")
            if errorCode == "E90001":
                return True
    return False


def get_vcmpsel_tsss_result_by_op(op):
    lhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    return math.vcmpsel(lhs, 2, "lt", 565, 333)


def test_vcmpsel_dsl_tsss_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    for op in operation:
        math.vcmpsel(lhs, 2, op, 23, 12)
    return True


def test_vcmpsel_dsl_ttss_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    rhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    rhs = tvm.placeholder(rhs_shape, name="lhs", dtype="float32")
    for op in operation:
        math.vcmpsel(lhs, rhs, op, 23, 12)
    return True


def test_vcmpsel_dsl_tsts_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    slhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    slhs = tvm.placeholder(slhs_shape, name="lhs", dtype="float32")

    for op in operation:
        math.vcmpsel(lhs, 23, op, slhs, 12)
    return True


def test_vcmpsel_dsl_tsst_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    srhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    srhs = tvm.placeholder(srhs_shape, name="lhs", dtype="float32")

    for op in operation:
        math.vcmpsel(lhs, 23, op, 12, srhs)
    return True


def test_vcmpsel_dsl_ttts_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    rhs_shape = (1, 32, 32, 16)
    slhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    rhs = tvm.placeholder(rhs_shape, name="lhs", dtype="float32")
    slhs = tvm.placeholder(slhs_shape, name="lhs", dtype="float32")

    for op in operation:
        math.vcmpsel(lhs, rhs, op, slhs, 23)
    return True


def test_vcmpsel_dsl_ttst_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    rhs_shape = (1, 32, 32, 16)
    srhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    rhs = tvm.placeholder(rhs_shape, name="lhs", dtype="float32")
    srhs = tvm.placeholder(srhs_shape, name="lhs", dtype="float32")

    for op in operation:
        math.vcmpsel(lhs, rhs, op, 12, srhs)
    return True


def test_vcmpsel_dsl_tstt_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    slhs_shape = (1, 32, 32, 16)
    srhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    slhs = tvm.placeholder(slhs_shape, name="lhs", dtype="float32")
    srhs = tvm.placeholder(srhs_shape, name="lhs", dtype="float32")

    for op in operation:
        math.vcmpsel(lhs, 12, op, slhs, srhs)
    return True


def test_vcmpsel_dsl_tttt_op(_):
    operation = ['eq', 'ne', 'lt', 'gt', 'ge', 'le']
    lhs_shape = (1, 32, 32, 16)
    rhs_shape = (1, 32, 32, 16)
    slhs_shape = (1, 32, 32, 16)
    srhs_shape = (1, 32, 32, 16)
    lhs = tvm.placeholder(lhs_shape, name="lhs", dtype="float32")
    rhs = tvm.placeholder(rhs_shape, name="lhs", dtype="float32")
    slhs = tvm.placeholder(slhs_shape, name="lhs", dtype="float32")
    srhs = tvm.placeholder(srhs_shape, name="lhs", dtype="float32")

    for op in operation:
        math.vcmpsel(lhs, rhs, op, slhs, srhs)
    return True


test_func_list = [
    test_var_addition,
    test_single_elewise_op_name_check,
    test_vlogic_operation_param,
    test_vlogic_lhs_tensor,
    test_vlogic_rhs_tensor,
    test_binary_elewise_op_dtype,
    test_vmla_tensor_1,
    test_vmadd_tensor_1,
    test_vsel_bit_shape_check_shape,
    test_multiple_elewise_op_support_dtype,
    test_multiple_elewise_op_support_name,
    test_vsel_bit_shape_check_len,
    test_vsel_bit_shape_check_condition_lastdim,
    test_vsel_bit_shape_check_condition_not_lastdim,
    test_vsel_bit_shape_check_input_shape_value_positive,
    # test_vsel_dynamic_check_shape,
    test_vsel_dynamic_check_condition_dtype,
    # test_vcmpsel_data_shape_check_input_value,
    test__vcmpsel_data_dtype_check,
    test_vcmpsel_dsl_tsss_op,
    test_vcmpsel_dsl_ttss_op,
    test_vcmpsel_dsl_tsts_op,
    test_vcmpsel_dsl_tsst_op,
    test_vcmpsel_dsl_ttts_op,
    test_vcmpsel_dsl_ttst_op,
    test_vcmpsel_dsl_tstt_op,
    test_vcmpsel_dsl_tttt_op
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
