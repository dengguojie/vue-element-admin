# !/usr/bin/env python
# coding=utf-8
"""
Function:
This method mainly handle the dynamic scenario.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
from . import utils

NEW_STR_KEY = ['shape_range']


def _update_shape_range_in_base_list(base_case):
    """
    dynamic shape:
    add shape_range when create case.json
    """
    if base_case.get(utils.INPUT_DESC):
        for desc_list in base_case.get(utils.INPUT_DESC):
            desc_list.update({
                utils.SHAPE_RANGE: utils.SHAPE_RANGE_DEFAULT_VALUE})
    if base_case.get(utils.OUTPUT_DESC):
        for desc_list in base_case.get(utils.OUTPUT_DESC):
            desc_list.update({
                utils.SHAPE_RANGE: utils.SHAPE_RANGE_DEFAULT_VALUE})
    return


def is_dynamic_shape_update_desc_list(dynamic_shape_support, base_case):
    """check operator exist dynamic shape."""
    if dynamic_shape_support and \
            dynamic_shape_support.get('flag') == 'true':
        _update_shape_range_in_base_list(base_case)
    return


def check_typical_shape_valid(typical_shape, json_path):
    """check typical_shape are integers and greater than 0"""
    for dim in typical_shape:
        if not isinstance(dim, int):
            utils.print_error_log(
                'The value(%s) of "typical_shape" is not int. '
                'Please modify it in file %s.' % (typical_shape, json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        if dim < 0:
            utils.print_error_log(
                'The value(%s) of "typical_shape" must be greater than 0. '
                'Please modify it in file %s.' % (typical_shape, json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)


def check_not_dynamic_shape(shape_list):
    """check whether dynamic shape, otherwise return False"""
    if not shape_list:
        return False
    # check -1 or -2 in shape_list value as a basis, return True.
    for dim in shape_list:
        if isinstance(dim, list):
            for item in dim:
                if item == utils.SHAPE_DYNAMIC_SCENARIOS_ONE \
                        or item == utils.SHAPE_DYNAMIC_SCENARIOS_TWO:
                    return True
        elif isinstance(dim, int):
            if dim == utils.SHAPE_DYNAMIC_SCENARIOS_ONE \
                    or dim == utils.SHAPE_DYNAMIC_SCENARIOS_TWO:
                return True
    return False


def add_key_in_cross_key_list(key_list):
    """add new key in cross_key_list"""
    for key in NEW_STR_KEY:
        key_list.append(key)


def set_typical_shape_in_cur_params(cur_params, tensor, current_json_path):
    """update cur_params dict"""
    shape_list = cur_params.get('shape')
    for dim in shape_list:
        if dim == utils.SHAPE_DYNAMIC_SCENARIOS_ONE \
                or dim == utils.SHAPE_DYNAMIC_SCENARIOS_TWO:
            typical_shape_list = tensor.get(utils.TYPICAL_SHAPE)
            if typical_shape_list is None:
                utils.print_error_log("Please add \"typical_shape\" filed in "
                                      "%s used for executing the operator in "
                                      "dynamic shape scenarios."
                                      % current_json_path)
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_NONE_TYPICAL_SHAPE_ERROR)
            else:
                cur_params.update({utils.TYPICAL_SHAPE: typical_shape_list[0]})
            # dynamic shape scenarios two, need to remove shape_range.
            if dim == utils.SHAPE_DYNAMIC_SCENARIOS_TWO\
                    and cur_params.get(utils.SHAPE_RANGE):
                cur_params.pop(utils.SHAPE_RANGE)
    return


def replace_shape_to_typical_shape(op_desc_dict):
    """
    if exist typical_shape and shape dim is -1 or -2,
    replace typical_shape as shape,
    return typical_shape
    Otherwise return initials shape
    """
    if op_desc_dict.get(utils.TYPICAL_SHAPE) is not None:
        typical_shape_list = op_desc_dict.get(utils.TYPICAL_SHAPE)
        if len(typical_shape_list) == 0:
            utils.print_warn_log("Please input values of typical_shape used "
                                 "for executing the operator.")
        shape_list = typical_shape_list
    else:
        shape_list = op_desc_dict.get('shape')
    return shape_list
