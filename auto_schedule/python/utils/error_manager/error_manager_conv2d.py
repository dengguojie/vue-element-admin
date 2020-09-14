"""Copyright 2019 Huawei Technologies Co., Ltd

error manager conv2d.
"""
import json
from te.utils.error_manager.error_manager_util import get_error_message

def raise_err_input_params_not_expected(op_name, param_name, \
                                        expected_value, input_value):
    """
    The op[%s] input parameter[%s] should be [%s], actual the input is [%s] %
    (op_name,param_name,excepted_value,input_value)
    :param op_name
    :param param_name
    :param expected_value
    :param input_value
    :return
    """
    args_dict = {
        "errCode": "E60000",
        "op_name": op_name,
        "param_name": param_name,
        "expected_value": expected_value,
        "input_value": input_value
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_params_not_supported(op_name, scale_value, sqrt_mode):
    """
    In op[%s], quant model only surport scale == 1 and sqrt == 0,
    but scale is [%s], sqrt is [%s] %
    (op_name,scale_value,sqrt_mode)
    :param op_name
    :param scale_value
    :param sqrt_mode
    :return
    """
    args_dict = {
        "errCode": "E60036",
        "op_name": op_name,
        "expected_value": scale_value,
        "input_value": sqrt_mode
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_format_invalid(op_name, param_name, \
                                   expected_format_list, input_format):
    """
    The format of [%s] of op[%s] must in [%s], actual format is [%s] %
    (param_name, op_name, excepted_format_list, format)
    :param op_name
    :param param_name
    :param expected_format_list
    :param input_format
    :return
    """
    args_dict = {
        "errCode": "E60004",
        "op_name": op_name,
        "param_name": param_name,
        "expected_format_list": expected_format_list,
        "format": input_format
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_attr_range_invalid(op_name, attr_range, attr_name, value):
    """
    In op[%s], the [%s] must in range [%s], actual is [%s] %
    (op_name,range,attr_name,value)
    :param op_name
    :param attr_range
    :param attr_name
    :param value
    :return
    """
    args_dict = {
        "errCode": "E60011",
        "op_name": op_name,
        "range": attr_range,
        "attr_name": attr_name,
        "value": value
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_should_4d(op_name, param_name):
    """
    In op[%s], [%s] should be 4d list % (op_name, param_name)
    :param op_name
    :param param_name
    :return
    """
    args_dict = {
        "errCode": "E60107",
        "op_name": op_name,
        "param_name": param_name
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_specific(op_name, reason):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param reason
    :return
    """
    args_dict = {
        "errCode": "E60108",
        "op_name": op_name,
        "reason": reason
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_common(op_name, reason, value):
    """
    In op[%s], [%s],actual is [%s] % (op_name,reason,value)
    :param op_name
    :param reason
    :param value
    :return
    """
    args_dict = {
        "errCode": "E60114",
        "op_name": op_name,
        "reason": reason,
        "value": value
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_should_be_4d(op_name, param_name):
    """
    In op[%s], [%s] should be 4d list % (op_name, param_name)
    :param op_name
    :param param_name
    :return
    """
    args_dict = {
        "errCode": "E61000",
        "op_name": op_name,
        "param_name": param_name
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_specific_user(op_name, reason):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param reason
    :return
    """
    args_dict = {
        "errCode": "E61001",
        "op_name": op_name,
        "reason": reason
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_input_mem_type(op_name, input_memory_type):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param input_memory_type
    :return
    """
    args_dict = {
        "errCode": "E61500",
        "op_name": op_name,
        "input_memory_type": input_memory_type
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_output_mem_type(op_name, output_memory_type):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param output_memory_type
    :return
    """
    args_dict = {
        "errCode": "E61501",
        "op_name": op_name,
        "output_memory_type": output_memory_type
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_check_the_validity_of_variable(op_name, sence_params, \
    param_1, param_2):
    """
    The op [%s]. [%s] , the value is [%s] and [%s] %
    (op_name, sence_params, param_1, param_2)
    :param op_name
    :param sence_params
    :param param_1
    :param param_2
    :return
    """
    args_dict = {
        "errCode": "E61203",
        "op_name": op_name,
        "sence_params": sence_params,
        "param_1": param_1,
        "param_2": param_2
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_check_the_validity_of_one_variable(op_name, \
    sence_params, param_1):
    """
    The op [%s]. [%s] , the value is [%s] %
    (op_name, sence_params, param_1)
    :param op_name
    :param sence_params
    :param param_1
    :param param_2
    :return
    """
    args_dict = {
        "errCode": "E61204",
        "op_name": op_name,
        "sence_params": sence_params,
        "param_1": param_1
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_specific_input_shape(op_name, reason):
    """
    In op[%s], [%s] % (op_name,reason)
    :param op_name
    :param reason
    :return
    """
    args_dict = {
        "errCode": "E61205",
        "op_name": op_name,
        "reason": reason
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_value_or_format_invalid(op_name, param_name, \
                                      expect_value, condition):
    """
    wrong tiling in op[%s]: [%s] must be equal to [%s] when [%s] %
    (op_name, param_name, expect_value, condition)
    :param op_name
    :param param_name
    :param expect_value
    :param condition
    :return
    """
    args_dict = {
        "errCode": "E61300",
        "op_name": op_name,
        "param_name": param_name,
        "expect_value": expect_value,
        "condition": condition
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_equal_invalid(op_name, param_name_1, param_name_2):
    """
    wrong tiling in op[%s]: [%s] must be equal to [%s] %
    (op_name, param_name_1, param_name_2)
    :param op_name
    :param param_name_1
    :param param_name_2
    :return
    """
    args_dict = {
        "errCode": "E61301",
        "op_name": op_name,
        "param_name_1": param_name_1,
        "param_name_2": param_name_2
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_scene_limitation(op_name, scene, param_name, claim):
    """
    The op [%s], if it is the [%s] cut shape, the [%s] must be [%s] %
    (op_name, scene, param_name, claim)
    :param op_name
    :param scene
    :param param_name
    :return
    """
    args_dict = {
        "errCode": "E61601",
        "op_name": op_name,
        "scene": scene,
        "param_name": param_name,
        "claim": claim
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_check_type(op_name, param_name, optype_1, optype_2):
    """
    The op [%s] input parameter [%s] should be [%s] type,
    but the type you enter is [%s] %
    (op_name, param_name, optype_1, optype_2)
    :param op_name
    :param param_name
    :param optype_1
    :param optype_2
    :return
    """
    args_dict = {
        "errCode": "E61602",
        "op_name": op_name,
        "param_name": param_name,
        "optype_1": optype_1,
        "optype_2": optype_2
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_scene_equal_limitation(op_name, param_1, param_2):
    """
    The op [%s] [%s] must equal to [%s] % (op_name, param_1, param_2)
    :param op_name
    :param param_1
    :param param_2
    :return
    """
    args_dict = {
        "errCode": "E61603",
        "op_name": op_name,
        "optype_1": param_1,
        "optype_2": param_2
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)


def raise_err_contain_key_invalid(op_name, param_name, key):
    """
    The op [%s] [%s] input does not contain the [%s] key %
    (op_name,param_name,key)
    :param op_name
    :param param_name
    :param key
    :return
    """
    args_dict = {
        "errCode": "E60029",
        "op_name": op_name,
        "param_name": param_name,
        "key": key
    }
    msg = get_error_message(args_dict)
    raise RuntimeError(args_dict, msg)