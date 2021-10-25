# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe import tvm
from tbe.dsl.compute import util

from tbe.dsl.compute.util import dtype_check_decorator

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_util_impl", "dsl_vsub")


def test_dsl_support_dtype_dsl_name_not_str(_):
    dtypes = util.dsl_support_dtype(1)
    if len(dtypes) == 0:
        return True
    return False


def test_dsl_support_dtype_all_support_dtype_none(_):
    dtypes = util.dsl_support_dtype("1")
    if len(dtypes) == 0:
        return True
    return False


@dtype_check_decorator
def broadcast(num):
    return num


@dtype_check_decorator
def concat(listType):
    return listType


def test_dtype_check_decorator_broadcast_int32(_):
    broadcast(4)
    return True


def test_dtype_check_decorator_concat_int32(_):
    try:
        concat("123")
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False


def test_str_to_tuple(_):
    result_array = util.str_to_tuple("1,2,3")
    if "1" in result_array and "2" in result_array and "3" in result_array:
        return True
    return False


def test_str_to_tuple_Blank(_):
    result_array = util.str_to_tuple(None)
    if len(result_array) == 0:
        return True
    return False


def test_dsl_get_cast_type_dst_type_exception(_):
    try:
        util.get_cast_type("int8", "int88")
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_judge_var_num_type_exception(_):
    try:
        util.judge_var("abc")
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_int_ceil_div_numb_zero_exception(_):
    try:
        util.int_ceil_div(1, 0)
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_int_ceil_div(_):
    result = util.int_ceil_div(5, 4)
    if result == 2:
        return True
    return False


def test_align_x2_zero_exception(_):
    try:
        util.align(1, 0)
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_align(_):
    result = util.align(5, 4)
    if result == 8:
        return True
    return False


def test_get_and_res(_):
    if not util.get_and_res(True, False):
        return True
    return False


def test_get_or_res(_):
    if util.get_or_res(True, False):
        return True
    return False


def test_refine_axis_negative_axis(_):
    res = util.refine_axis(-1, (1, 2, 3, 4))
    if 3 in res:
        return True
    return False


def test_refine_axis_exceed_max_shape_length(_):
    try:
        util.refine_axis(10, (1, 2, 3, 4))
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_check_exception(_):
    try:
        util._check(False, "testException")
    except RuntimeError as e:
        detailed_cause = e.args[0].get("detailed_cause")
        if detailed_cause == "testException":
            return True
    return False


def test_get_tvm_scalar_type_tvm_const(_):
    scalar = tvm.const(1, dtype="int")
    res = util.get_tvm_scalar(scalar, "float16")
    if res.dtype == "float16":
        return True
    return False


def test_check_input_tensor_shape_negative_exception(_):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            util.check_input_tensor_shape((-4, 1, 2, 3))
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_axis_value_type_check_value_exceed_shape_len(_):
    try:
        util._axis_value_type_check(4, 5)
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_reduce_axis_check(_):
    res = util.reduce_axis_check(6, [5])
    if 5 in res:
        return True
    return False


def test_util_astype_exception(_):
    try:
        util.util_astype("dada", "float16")
    except RuntimeError as e:
        errCode = e.args[0].get("errCode")
        if errCode == "E90001":
            return True
    return False


def test_util_astype_tvm_var(_):
    res = util.util_astype(tvm.var("name", "float16"), "float32")
    if res.dtype == "float32":
        return True
    return False


def test_get_priority_flag_value(_):
    if util._get_priority_flag_value(1) == 1:
        return True
    return False


def test_get_priority_flag_value2(_):
    if util._get_priority_flag_value(tvm.const(1, "int8")) == 1:
        return True
    return False


def test_dtype_check_decorator_concat_not_tensor_exception(_):
    try:
        concat(["123", "456"])
    except RuntimeError as e:
        if e.args[0].get("errCode") == "E90001":
            return True
    return False



test_func_list = [
    test_dsl_support_dtype_dsl_name_not_str,
    test_dsl_support_dtype_all_support_dtype_none,
    test_dtype_check_decorator_broadcast_int32,
    test_dtype_check_decorator_concat_int32,
    test_str_to_tuple,
    test_str_to_tuple_Blank,
    test_dsl_get_cast_type_dst_type_exception,
    test_judge_var_num_type_exception,
    test_int_ceil_div_numb_zero_exception,
    test_int_ceil_div,
    test_align_x2_zero_exception,
    test_align,
    test_get_and_res,
    test_get_or_res,
    test_refine_axis_negative_axis,
    test_refine_axis_exceed_max_shape_length,
    test_check_exception,
    test_get_tvm_scalar_type_tvm_const,
    test_check_input_tensor_shape_negative_exception,
    test_axis_value_type_check_value_exceed_shape_len,
    test_reduce_axis_check,
    test_util_astype_exception,
    test_util_astype_tvm_var,
    test_get_priority_flag_value,
    test_get_priority_flag_value2,
    test_dtype_check_decorator_concat_not_tensor_exception,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
