"""Copyright 2019 Huawei Technologies Co., Ltd

error manager conv2d.
"""
import json
from te.utils.error_manager.error_manager_util import get_error_message

def raise_err_input_value_invalid(op_name, param_name, \
                                        excepted_value, real_value):
    """
    "In op[%s], the parameter[%s] should be [%s], but actually is [%s]." %
    (op_name,param_name,excepted_value, real_value)
    :param op_name
    :param param_name
    :param excepted_value
    :param real_value
    :return
    """
    args_dict = {
        "errCode": "E80000",
        "op_name": op_name,
        "param_name": param_name,
        "excepted_value": excepted_value,
        "real_value": real_value
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)

def raise_err_inputs_dtype_not_equal(op_name, param_name1,param_name2, \
                                        param1_dtype, param2_dtype):
    """
    "In op[%s], the parameter[%s][%s] are not equal in dtype with dtype[%s][%s]." %
    (op_name,param_name1,param_name2,param1_dtype, param2_dtype)
    :param op_name
    :param param_name1
    :param param_name2
    :param param1_dtype
    :param param2_dtype
    :return
    """
    args_dict = {
        "errCode": "E80018",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "param1_dtype": param1_dtype,
        "param2_dtype": param2_dtype
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)

def raise_err_input_shpae_invalid(op_name, param_name, \
                                        error_detail):
    """
    "In op[%s], the shape of input[%s] is invalid, [%s]." %
    (op_name,param_name,error_detail)
    :param op_name
    :param param_name
    :param error_detail 
    :return
    """
    args_dict = {
        "errCode": "E80028",
        "op_name": op_name,
        "param_name": param_name,
        "error_detail": error_detail
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)

def raise_err_two_input_shpae_invalid(op_name, param_name1, \
                                        param_name2, error_detail):
    """
    "In op[%s], the shape of inputs[%s][%s] are invalid, [%s]." %
    (op_name,param_name1,param_name2,error_detail)
    :param op_name
    :param param_name1
    :param param_name2
    :param error_detail
    :return
    """
    args_dict = {
        "errCode": "E80029",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "error_detail": error_detail
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)

def raise_err_two_input_dtype_invalid(op_name, param_name1, \
                                        param_name2, error_detail):
    """
    "In op[%s], the dtype of inputs[%s][%s] are invalid, [%s]." %
    (op_name,param_name1,param_name2,error_detail)
    :param op_name
    :param param_name1
    :param param_name2
    :param error_detail
    :return
    """
    args_dict = {
        "errCode": "E80030",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "error_detail": error_detail
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)

def raise_err_two_input_format_invalid(op_name, param_name1, \
                                        param_name2, error_detail):
    """
    "In op[%s], the format of inputs[%s][%s] are invalid, [%s]." %
    (op_name,param_name1,param_name2,error_detail)
    :param op_name
    :param param_name1
    :param param_name2
    :param error_detail
    :return
    """
    args_dict = {
        "errCode": "E80031",
        "op_name": op_name,
        "param_name1": param_name1,
        "param_name2": param_name2,
        "error_detail": error_detail
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)
