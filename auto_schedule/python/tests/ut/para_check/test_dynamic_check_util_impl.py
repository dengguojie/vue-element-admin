import tbe
from sch_test_frame.ut import OpUT
from tbe.common.utils import para_check


ut_case = OpUT("check_util", "para_check.test_dynamic_check_util_impl")

def test_check_dtype_None(_):
    try:
        para_check.check_dtype(None, ("float16", "float32"))
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80007"
    return False
    
def test_check_dtype_not_str(_):
    try:
        para_check.check_dtype(1, ("float16", "float32"))
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80003"
    return False

def test_check_dtype_not_contains(_):
    try:
        para_check.check_dtype("int8", ("float16", "float32"))
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80008"
    return False

def test_check_format_None(_):
    try:
        para_check.check_dtype(None)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80007"
    return False

def test_check_format_XXXX(_):
    try:
        para_check.check_dtype("XXXX")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80008"
    return False

def test_check_elewise_shape_range_static(_):
    try:
        para_check.check_elewise_shape_range([])
    except RuntimeError as e:
        return False
    return True

def test_check_elewise_shape_range_not_dict(_):
    try:
        input_list = [
            [],
            {"shape":[7, -1], "dtype":"float16"}
        ]
        with tbe.common.context.op_context.OpContext("dynamic"):
            para_check.check_elewise_shape_range(input_list)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80003"
    return False

def test_check_elewise_shape_range_dict_key(_):
    try:
        input_list = [
            {"shape":[7, -1], },
            {"shape":[7, -1], "dtype":"float16"}
        ]
        with tbe.common.context.op_context.OpContext("dynamic"):
            para_check.check_elewise_shape_range(input_list)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80004"
    return False

def test_check_elewise_shape_range_range_len(_):
    try:
        input_list = [
            {"shape":[7, -1], "range": [(7, 7), (1, 16, 1)]},
            {"shape":[7, -1], "range": [(7, 7), (1, 16)]}
        ]
        with tbe.common.context.op_context.OpContext("dynamic"):
            para_check.check_elewise_shape_range(input_list)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80023"
    return False

def test_check_elewise_shape_range_range_intersetctions(_):
    try:
        input_list = [
            {"shape":[7, -1], "range": [(7, 7), (1, 16)]},
            {"shape":[7, -1], "range": [(7, 7), (17, 100)]}
        ]
        with tbe.common.context.op_context.OpContext("dynamic"):
            para_check.check_elewise_shape_range(input_list)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80024"
    return False

def test_check_elewise_shape_range_range_intersetctions_broadcast(_):
    try:
        input_list = [
            {"shape":[7, -1], "range": [(7, 7), (2, 16)]},
            {"shape":[7, -1], "range": [(7, 7), (17, 100)]}
        ]
        with tbe.common.context.op_context.OpContext("dynamic"):
            para_check.check_elewise_shape_range(input_list, support_broadcast=True)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80024"
    return False

@para_check.check_input_type(dict, dict, dict, str)
def test_inner_function_len(a, b, c, k="kernel_name"):
    return

def test_check_input_type(_):
    try:
        test_inner_function_len("a", "a", "a")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60038"
    return False

def test_check_input_kw_type(_):
    try:
        test_inner_function_len({"a":1}, {"a":1}, {"a":1}, k=7)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60038"
    return False

def test_check_input_inside_pass(_):
    try:
        para_check._check_input_type_dict({"shape":[7, -1], "dtype":"float16"}, ["shape", "dtype"], "input0")
    except RuntimeError as e:
        return False
    return True

def test_check_input_inside_dtype_type(_):
    try:
        para_check._check_input_type_dict({"shape":[7, -1], "dtype":16}, ["shape", "dtype"], "input0")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60037"
    return False

def test_check_input_inside(_):
    try:
        para_check._check_input_type_dict({"shape":[7, -1], "dtype":"float16"}, ["shape", "value"], "input0")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60038"
    return False

def test_check_dtype_rule_None(_):
    try:
        para_check.check_dtype_rule(None, ("float16", "float32"), "input")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60038"
    return False

def test_check_dtype_rule_error(_):
    try:
        para_check.check_dtype_rule("None", ("float16", "float32"), "input")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60005"
    return False

def test_check_dtype_rule_error_default(_):
    try:
        para_check.check_dtype_rule("None", ("float16", "float32"),)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60038"
    return False

def test_check_shape_rule_None(_):
    try:
        para_check.check_shape_rule("None",)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60037"
    return False

def test_check_shape_rule_type(_):
    try:
        para_check.check_shape_rule({"None":1},)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60037"
    return False

def test_check_shape_rule_len(_):
    try:
        para_check.check_shape_rule([1, 2, 3, 4, 5, 6, 7, 8, 9],)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60011"
    return False

def test_check_shape_rule_value_not_int(_):
    try:
        para_check.check_shape_rule(["1", 2, 3, 4, 5,],)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60037"
    return False

def test_check_shape_rule_value_out_of_range(_):
    try:
        para_check.check_shape_rule([-7, 2, 3, 4, 5,],)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60039"
    return False

def test_check_kernel_name_None(_):
    try:
        para_check.check_kernel_name(None)
    except RuntimeError as e:
        return True
    return True

def test_check_kernel_name_len(_):
    try:
        para_check.check_kernel_name("7"*300)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60039"
    return False

def test_check_kernel_name_value(_):
    try:
        para_check.check_kernel_name("xxxxx+-[")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60038"
    return False

def test_check_and_init_5hdc_reduce_support_true(_):
    return para_check.check_and_init_5hdc_reduce_support({"ori_shape":[1, 1, 1, 1], "ori_format":"NC1HWC0",
                                                          "format":"NC1HWC0", "dtype":"float16"},(1, 4))

def test_check_and_init_5hdc_reduce_support_false(_):
    return not (para_check.check_and_init_5hdc_reduce_support({"format":"xxxxx", "dtype":"float16"},(1, 4)))

def test_check_and_init_5hdc_reduce_support_value(_):
    try:
        para_check.check_and_init_5hdc_reduce_support({"format":"NC1HWC0", "dtype":"float16"},(1, 4))
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60040"
    return False

def test_is_scalar_true(_):
    return para_check.is_scalar([1,])

def test_is_scalar_false(_):
    return not para_check.is_scalar([1, 7])

def test_check_shape_size(_):
    try:
        para_check.check_shape_size((4, 2, 100),200)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E60039"
    return False

def test_check_tensor_shape_size(_):
    shape_size = para_check.check_tensor_shape_size((4, 2, 100))
    return shape_size == 800

def test_check_reduce_shape_rule(_):
    para_check.check_reduce_shape_rule((4, 2, 100))
    return True

def test_outer_check_format_None(_):
    try:
        para_check.check_format(None)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80017"
    return False

def test_outer_check_format_XXXX(_):
    try:
        para_check.check_format("XXXX")
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80015"
    return False

def test_outer_check_shape_type(_):
    try:
        para_check.check_shape(7)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80003"
    return False

def test_outer_check_shape_rank(_):
    try:
        para_check.check_shape((1, 2, 3, 4, 5, 6, 7, 8, 9))
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80012"
    return False

def test_outer_check_shape_value(_):
    try:
        para_check.check_shape((1, 2, -3,))
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80002"
    return False

def test_outer_check_shape_min_size(_):
    try:
        para_check.check_shape((1, 2, 3,), min_size=100)
    except RuntimeError as e:
        return e.args[0].get("errCode") == "E80011"
    return False

ut_case.add_cust_test_func(test_func=test_outer_check_shape_rank)
ut_case.add_cust_test_func(test_func=test_outer_check_shape_value)
ut_case.add_cust_test_func(test_func=test_outer_check_shape_min_size)

ut_case.add_cust_test_func(test_func=test_check_dtype_None)
ut_case.add_cust_test_func(test_func=test_check_dtype_not_str)
ut_case.add_cust_test_func(test_func=test_check_dtype_not_contains)
ut_case.add_cust_test_func(test_func=test_check_format_None)
ut_case.add_cust_test_func(test_func=test_check_format_XXXX)
ut_case.add_cust_test_func(test_func=test_check_elewise_shape_range_static)
ut_case.add_cust_test_func(test_func=test_check_elewise_shape_range_not_dict)
ut_case.add_cust_test_func(test_func=test_check_elewise_shape_range_dict_key)
ut_case.add_cust_test_func(test_func=test_check_elewise_shape_range_range_len)
ut_case.add_cust_test_func(test_func=test_check_elewise_shape_range_range_intersetctions)
ut_case.add_cust_test_func(test_func=test_check_elewise_shape_range_range_intersetctions_broadcast)
ut_case.add_cust_test_func(test_func=test_check_input_type)
ut_case.add_cust_test_func(test_func=test_check_input_kw_type)
ut_case.add_cust_test_func(test_func=test_check_input_inside_pass)
ut_case.add_cust_test_func(test_func=test_check_input_inside_dtype_type)
ut_case.add_cust_test_func(test_func=test_check_input_inside)
ut_case.add_cust_test_func(test_func=test_check_dtype_rule_None)
ut_case.add_cust_test_func(test_func=test_check_dtype_rule_error)
ut_case.add_cust_test_func(test_func=test_check_dtype_rule_error_default)
ut_case.add_cust_test_func(test_func=test_check_shape_rule_None)
ut_case.add_cust_test_func(test_func=test_check_shape_rule_type)
ut_case.add_cust_test_func(test_func=test_check_shape_rule_len)
ut_case.add_cust_test_func(test_func=test_check_shape_rule_value_not_int)
ut_case.add_cust_test_func(test_func=test_check_shape_rule_value_out_of_range)
ut_case.add_cust_test_func(test_func=test_check_kernel_name_None)
ut_case.add_cust_test_func(test_func=test_check_kernel_name_len)
ut_case.add_cust_test_func(test_func=test_check_kernel_name_value)
ut_case.add_cust_test_func(test_func=test_check_and_init_5hdc_reduce_support_true)
ut_case.add_cust_test_func(test_func=test_check_and_init_5hdc_reduce_support_false)
ut_case.add_cust_test_func(test_func=test_check_and_init_5hdc_reduce_support_value)
ut_case.add_cust_test_func(test_func=test_is_scalar_true)
ut_case.add_cust_test_func(test_func=test_is_scalar_false)
ut_case.add_cust_test_func(test_func=test_check_shape_size)
ut_case.add_cust_test_func(test_func=test_check_tensor_shape_size)
ut_case.add_cust_test_func(test_func=test_check_reduce_shape_rule)
ut_case.add_cust_test_func(test_func=test_outer_check_format_None)
ut_case.add_cust_test_func(test_func=test_outer_check_format_XXXX)
ut_case.add_cust_test_func(test_func=test_outer_check_shape_type)
