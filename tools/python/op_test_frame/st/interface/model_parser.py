#! /usr/bin/env python
# coding=utf-8
"""
Function:
AclOpGenerator class. This class mainly implements acl op src code generation.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
Change History: 2020-07-11 file Created
"""
try:
    import sys
    import importlib
    from interface import utils
except (ImportError,) as import_error:
    sys.exit("[model_parser]Unable to import module: %s." % str(import_error))

GET_MODEL_NODES_FUNC = 'get_model_nodes'
GET_SHAPE_FUNC = 'get_shape'
CHANGE_SHAPE_FUNC = 'change_shape'
FILE_NAME_SUFFIX = '_model_parser'

FRAMEWORK_TO_MODEL_NAME_MAP = {
    'tf': ['.pb']
}


def _get_framework_type(path):
    for (key, value) in list(FRAMEWORK_TO_MODEL_NAME_MAP.items()):
        for item in value:
            if path.endswith(item):
                return key
    utils.print_error_log(
        'The model file "%s" is invalid, only supports .pb file. '
        'Please modify it.' % path)
    raise utils.OpTestGenException(
        utils.OP_TEST_GEN_INVALID_PARAM_ERROR)


def _function_call(args, op_type, func_name):
    framework = _get_framework_type(args.model_path)
    module = importlib.import_module(
        'interface.framework.%s_model_parser' % framework)
    func = getattr(module, func_name)
    try:
        return func(args, op_type)
    except Exception as ex:
        utils.print_error_log(
            'Failed to execute "%s". %s' % (func_name, str(ex)))
        raise utils.OpTestGenException(
            utils.OP_TEST_GEN_INVALID_PARAM_ERROR)


def get_model_nodes(args, op_type):
    """
    get model nodes by framework
    :param op_type: the op type
    :param args: the argument
    :return: the model nodes
    """
    return _function_call(args, op_type, GET_MODEL_NODES_FUNC)


def get_shape(args):
    """
    get shape by framework
    :param args: the argument
    :return: the shape list
    """
    return _function_call(args, '', GET_SHAPE_FUNC)


def change_shape(args):
    """
    change shape by framework
    :param args: the argument
    :return: the shape list
    """
    return _function_call(args, '', CHANGE_SHAPE_FUNC)
