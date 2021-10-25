from sch_test_frame.ut import OpUT
from tbe.common.utils import errormgr

ut_case = OpUT("errormgr", "errormgr.test_dynamic_errormgr_impl")


def test_raise_err_input_value_invalid(_):
    try:
        errormgr.raise_err_input_value_invalid("op_name", "param_name", 0, 100)
    except RuntimeError as e:
        # E80000
        return e.args[0].get("errCode") == "E80000"
    return False

def test_raise_err_miss_mandatory_parameter(_):
    try:
        errormgr.raise_err_miss_mandatory_parameter("op_name", "param_name")
    except RuntimeError as e:
        # E80001
        return e.args[0].get("errCode") == "E80001"
    return False

def test_raise_err_input_param_not_in_range(_):
    try:
        errormgr.raise_err_input_param_not_in_range("op_name", "param_name", 0, 100, 1000)
    except RuntimeError as e:
        # E80001
        return e.args[0].get("errCode") == "E80001"
    return False

def test_raise_err_input_dtype_not_supported(_):
    try:
        errormgr.raise_err_input_dtype_not_supported("op_name", "param_name", ["float",], "int")
    except RuntimeError as e:
        # E80008
        return e.args[0].get("errCode") == "E80008"
    return False

def test_raise_err_check_params_rules(_):
    try:
        errormgr.raise_err_check_params_rules("op_name", "rule_desc", "param_name", "param_value")
    except RuntimeError as e:
        # E80009
        return e.args[0].get("errCode") == "E80009"
    return False

def test_raise_err_input_format_invalid(_):
    try:
        errormgr.raise_err_input_format_invalid("op_name", "param_name", ["ND","ND1"], "NHWC")
    except RuntimeError as e:
        # conflict name
        return True
    return False

def test_raise_err_inputs_shape_not_equal(_):
    try:
        errormgr.raise_err_inputs_shape_not_equal("op_name", "param_name1",
                                                  "param_name2", "param1_shape", "param2_shape", "expect_shape")
    except RuntimeError as e:
        # E80015
        return e.args[0].get("errCode") == "E80017"
    return False

def test_raise_err_inputs_dtype_not_equal(_):
    try:
        errormgr.raise_err_inputs_dtype_not_equal("op_name", "param_name1",
                                                  "param_name2", "param1_dtype", "param2_dtype")
    except RuntimeError as e:
        # E80018
        return e.args[0].get("errCode") == "E80018"
    return False

def test_raise_err_input_shape_invalid(_):
    try:
        errormgr.raise_err_input_shape_invalid("op_name", "param_name", "error_detail")
    except RuntimeError as e:
        # E80028
        return e.args[0].get("errCode") == "E80028"
    return False

def test_raise_err_two_input_shape_invalid(_):
    try:
        errormgr.raise_err_two_input_shape_invalid("op_name", "param_name1", "param_name2", "error_detail")
    except RuntimeError as e:
        # E80029
        return e.args[0].get("errCode") == "E80029"
    return False

def test_raise_err_two_input_dtype_invalid(_):
    try:
        errormgr.raise_err_two_input_dtype_invalid("op_name", "param_name1", "param_name2", "error_detail")
    except RuntimeError as e:
        # E80030
        return e.args[0].get("errCode") == "E80030"
    return False

def test_raise_err_two_input_format_invalid(_):
    try:
        errormgr.raise_err_two_input_format_invalid("op_name", "param_name1", "param_name2", "error_detail")
    except RuntimeError as e:
        # E80031
        return e.args[0].get("errCode") == "E80031"
    return False

def test_raise_err_specific_reson(_):
    try:
        errormgr.raise_err_specific_reson("op_name", "reason")
    except RuntimeError as e:
        # E61001
        return e.args[0].get("errCode") == "E61001"
    return False

def test_raise_err_pad_mode_invalid(_):
    try:
        errormgr.raise_err_pad_mode_invalid("op_name", "expected_pad_mode", "actual_pad_mode")
    except RuntimeError as e:
        # E60021
        return e.args[0].get("errCode") == "E60021"
    return False

def test_raise_err_input_param_range_invalid(_):
    try:
        errormgr.raise_err_input_param_range_invalid("op_name", "param_name", "min_value", "max_value", "real_value")
    except RuntimeError as e:
        # E80012
        return e.args[0].get("errCode") == "E80012"
    return False

def test_raise_err_dtype_invalid(_):
    try:
        errormgr.raise_err_dtype_invalid("op_name", "param_name", "expected_list", "dtype")
    except RuntimeError as e:
        # E60005
        return e.args[0].get("errCode") == "E60005"
    return False

def test_get_error_message(_):
    args_dict = {"errCode": "E61001", "op_name": "op_name", "reason": "reason"}
    res = errormgr.get_error_message(args_dict)
    return res == "In op [op_name], [reason]"

def test_get_error_message_error(_):
    args_dict = {"errCode": "Exxxxx", "op_name": "op_name", "reason": "reason"}
    res = errormgr.get_error_message(args_dict)
    return res == "errCode = Exxxxx has not been defined"

def test_raise_runtime_error(_):
    try:
        args_dict = {"errCode": "E61001", "op_name": "op_name", "reason": "reason"}
        errormgr.raise_runtime_error(args_dict)
    except RuntimeError as e:
        # E60005
        return e.args[0].get("errCode") == "E61001"
    return False

case_list = [
    test_raise_err_input_value_invalid,
    test_raise_err_miss_mandatory_parameter,
    test_raise_err_input_param_not_in_range,
    test_raise_err_input_dtype_not_supported,
    test_raise_err_check_params_rules,
    test_raise_err_input_format_invalid,
    test_raise_err_specific_reson,
    test_raise_err_inputs_shape_not_equal,
    test_raise_err_inputs_dtype_not_equal,
    test_raise_err_input_shape_invalid,
    test_raise_err_two_input_shape_invalid,
    test_raise_err_two_input_dtype_invalid,
    test_raise_err_two_input_format_invalid,
    test_raise_err_pad_mode_invalid,
    test_raise_err_input_param_range_invalid,
    test_raise_err_dtype_invalid,
    test_get_error_message,
    test_get_error_message_error,
    test_raise_runtime_error,
]

for item in case_list:
    ut_case.add_cust_test_func(test_func=item)
