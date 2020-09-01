#!/usr/bin/env python
# coding=utf-8
"""
Function:
CaseGenerator class.
This class mainly involves the generate function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

try:
    import os
    import sys
    import json
    import time
    import copy
    from interface import utils
    from interface.model_parser import get_model_nodes
except ImportError as import_error:
    sys.exit(
        "[case_generator] Unable to import module: %s." % str(import_error))

INI_INPUT = 'input'
INI_OUTPUT = 'output'

REQUIRED_OP_INFO_KEYS = ["paramType", "name"]
PARAM_TYPE_VALID_VALUE = ["dynamic", "optional", "required"]


class CaseGenerator:
    """
    the class for design test case.
    """

    def __init__(self, args):
        self.ini_path = os.path.realpath(args.input_file)
        self.output_path = os.path.realpath(args.output_path)
        self.op_info = {}
        self.op_type = ""
        self.args = args

    def check_argument_valid(self):
        """
        check input argument valid
        """
        if not self.ini_path.endswith(".ini"):
            utils.print_error_log(
                'The file "%s" is invalid, only supports .ini file. '
                'Please modify it.' % self.ini_path)
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_PATH_ERROR)
        utils.check_path_valid(self.ini_path)
        utils.check_path_valid(self.output_path, True)

    @staticmethod
    def _parse_bool_value(value):
        new_value = value.strip().lower()
        if new_value == 'true':
            return True
        if new_value == 'false':
            return False
        raise ValueError

    @staticmethod
    def _parse_list_list_int_value(value):
        value = value.replace(' ', '')
        if not value.startswith('[[') or not value.endswith(']]'):
            raise ValueError
        # skip [[ and ]]
        value = value[2:-2].split('],[')
        new_value = []
        for item in value:
            if ']' in item or '[' in item:
                raise ValueError
            new_value.append(list(map(int, item.split(','))))
        return new_value

    def _get_default_attr_value(self, attr_type, default_value_str, attr_name):
        default_value_str = default_value_str.strip()
        default_value = None
        try:
            if attr_type == 'float':
                default_value = float(default_value_str)
            elif attr_type == 'int':
                default_value = int(default_value_str)
            elif attr_type == 'bool':
                default_value = self._parse_bool_value(default_value_str)
            elif attr_type == 'str':
                default_value = default_value_str
            elif attr_type.startswith('list'):
                if default_value_str[0] != '[' or default_value_str[-1] != ']':
                    raise ValueError
                if attr_type == 'listListInt':
                    default_value = self._parse_list_list_int_value(
                        default_value_str)
                value_list = default_value_str[1:-1].split(',')
                if attr_type == 'listInt':
                    default_value = list(map(int, value_list))
                elif attr_type == 'listFloat':
                    default_value = list(map(float, value_list))
                elif attr_type == 'listStr':
                    default_value = [x.strip() for x in value_list]
                elif attr_type == 'listBool':
                    default_value = [self._parse_bool_value(x) for x in
                                     value_list]
        except ValueError as ex:
            utils.print_error_log(
                'The default value(%s) is invalid for type(%s). Please modify '
                'the default value in "%s" attr. %s' % (
                    default_value_str, attr_type, attr_name, ex))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        return default_value

    def _parse_ini_to_json(self):
        tbe_ops_info = {}
        with open(self.ini_path) as ini_file:
            lines = ini_file.readlines()
            for index, line in enumerate(lines):
                line = line.strip()
                if line == '':
                    continue
                # such as [Add]
                if line.startswith("["):
                    if line.endswith("]"):
                        self.op_type = line[1:-1].strip()
                        self.op_info = {}
                        tbe_ops_info[self.op_type] = self.op_info
                    else:
                        utils.print_error_log(
                            'At line %d, "%s" is invalid in file %s, only '
                            'supports "[xx]". Please modify it.' % (
                                index, line, self.ini_path))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
                else:
                    key_value = line.split('=')
                    if len(key_value) != 2:
                        utils.print_error_log(
                            'At line %d, "%s" is invalid in file %s, only '
                            'supports "xx.yy=zz". Please modify it.' % (
                                index, line, self.ini_path))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
                    keys = key_value[0].split('.')
                    if len(keys) != 2:
                        utils.print_error_log(
                            'At line %d, "%s" is invalid in file %s, only '
                            'supports "xx.yy=zz". Please modify it.' % (
                                index, line, self.ini_path))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
                    key0 = keys[0].strip()
                    if key0 not in self.op_info:
                        self.op_info[key0] = {}
                    self.op_info[key0][keys[1].strip()] = key_value[1].strip()
        if len(tbe_ops_info) != 1:
            utils.print_error_log(
                'There are %d operator in file %s, only supports one operator '
                'in .ini file. Please modify it.' % (
                    len(tbe_ops_info), self.ini_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

    @staticmethod
    def _check_op_info_list_valid(value_list, support_list, op_info_key):
        if len(value_list) == 0:
            utils.print_error_log(
                'The value of "%s" is empty. Please modify it.' % op_info_key)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
        for value in value_list:
            if value == '':
                utils.print_error_log(
                    'The value of "%s" is empty. Only supports %s. Please '
                    'modify it.' % (op_info_key, support_list))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
            if value not in support_list:
                utils.print_error_log(
                    'The value(%s) of "%s" is invalid. '
                    'Only supports %s. Please modify it.' % (
                        value, op_info_key, support_list))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

    def _check_op_info(self):
        utils.print_info_log('Start to check valid for op info.')
        dtype_count = 0
        for op_info_key in self.op_info:
            if op_info_key.startswith(INI_INPUT) or op_info_key.startswith(
                    INI_OUTPUT):
                op_info = self.op_info[op_info_key]
                missing_keys = []
                # check required key is missing
                for required_key in REQUIRED_OP_INFO_KEYS:
                    if required_key not in op_info:
                        missing_keys.append(required_key)
                if len(missing_keys) > 0:
                    utils.print_error_log(
                        'The "%s" is missing: %s. Please modify it.' % (
                            op_info_key, missing_keys))
                    raise utils.OpTestGenException(
                        utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

                # check paramType valid
                self._check_op_info_list_valid(
                    [op_info["paramType"]], PARAM_TYPE_VALID_VALUE,
                    op_info_key + '.paramType')

                # check dtype valid
                current_dtype_count = 0
                if 'dtype' in op_info:
                    dtype_list = op_info["dtype"].split(",")
                    self._check_op_info_list_valid(
                        dtype_list, list(utils.DTYPE_TO_NUMPY_MAP.keys()),
                        op_info_key + '.dtype')
                    current_dtype_count = len(dtype_list)
                    if dtype_count == 0:
                        dtype_count = current_dtype_count

                    if dtype_count != current_dtype_count:
                        utils.print_error_log(
                            'The number(%d) of "%s.dtype" is not equal to the '
                            'number(%d) of the first "dtype". Please modify it.'
                            % (dtype_count, op_info_key, current_dtype_count))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
                else:
                    op_info["dtype"] = ''
                # check format valid
                if 'format' in op_info:
                    format_list = op_info["format"].split(",")
                    self._check_op_info_list_valid(
                        format_list, utils.FORMAT_LIST, op_info_key + '.format')

                    if current_dtype_count != len(format_list):
                        utils.print_error_log(
                            'The number(%d) of "%s.dtype" is not equal to the '
                            'number(%d) of "%s.format". Please modify it.' % (
                                current_dtype_count, op_info_key,
                                len(format_list), op_info_key))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
                else:
                    op_info["format"] = ''
        utils.print_info_log('Finish to check valid for op info.')

    def _make_attr(self, key, value):
        name = key[len('attr_'):]
        if 'type' not in value:
            utils.print_error_log(
                'The "%s" is missing "type". Please modify it.' % key)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
        self._check_op_info_list_valid(
            [value['type']], list(utils.ATTR_TYPE_MAP.keys()), key + '.type')
        default_value = None
        if 'defaultValue' in value:
            default_value = self._get_default_attr_value(
                value['type'], value['defaultValue'], name)
        return {'name': name,
                'type': utils.ATTR_TYPE_MAP[value['type']],
                'value': default_value}

    def _check_desc_valid(self, base_case, key):
        if len(base_case[key]) == 0:
            utils.print_error_log(
                'The number of %s is zero in file %s. Please modify it.'
                % (key[:-5], self.ini_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

    def _generate_base_case(self):
        base_case = {'case_name': 'Test_' + self.op_type.replace('/', '_')
                                  + '_001',
                     'op': self.op_type, 'input_desc': [], 'output_desc': []}
        for (key, value) in list(self.op_info.items()):
            if key.startswith(INI_INPUT):
                input_format = [] if len(value['format']) == 0 else \
                    list(set(value['format'].split(',')))
                input_dtype = [] if len(value['dtype']) == 0 else \
                    list(set(value['dtype'].split(',')))
                input_desc = {'format': input_format, 'type': input_dtype,
                              'shape': [], 'data_distribute': ['uniform'],
                              'value_range': [[0.1, 1.0]]}
                base_case['input_desc'].append(input_desc)
            elif key.startswith(INI_OUTPUT):
                output_format = [] if len(value['format']) == 0 else \
                    list(set(value['format'].split(',')))
                output_dtype = [] if len(value['dtype']) == 0 else \
                    list(set(value['dtype'].split(',')))
                output_desc = {'format': output_format, 'type': output_dtype,
                               'shape': []}
                base_case['output_desc'].append(output_desc)
            elif key.startswith("attr_"):
                if 'attr' not in base_case:
                    base_case['attr'] = []
                base_case['attr'].append(self._make_attr(key, value))
        self._check_desc_valid(base_case, 'input_desc')
        self._check_desc_valid(base_case, 'output_desc')
        # generate base case from model
        if self.args.model_path != "":
            return self._generate_base_case_from_model(base_case)
        return [base_case]

    def _update_input_output_from_model(self, base_case, node):
        layer_name = node['layer'].replace('/', '_')
        try:
            # when the length not equal, skip to next layer
            if len(node['input_shape']) != len(
                    base_case['input_desc']):
                utils.print_warn_log(
                    'The \"%s\" layer is skipped, because its number of '
                    'inputs(%d) is different from '
                    'that(%d) specified in the .ini file.Ignore if you have '
                    'change the "placeholder" shape.'
                    % (layer_name, len(node['input_shape']),
                       len(base_case['input_desc']),
                       ))
                return None
            if len(node['output_shape']) != len(
                    base_case['output_desc']):
                utils.print_warn_log(
                    'The \"%s\" layer is skipped, because its number of '
                    'outputs(%d) is different from '
                    'that(%d) specified in the .ini file. Ignore if you have '
                    'changed the "Placeholder" shape.'
                    % (len(node['output_shape']),
                       layer_name,
                       len(base_case['output_desc']),))
                return None

            new_base_case = {
                'case_name': 'Test_%s' % layer_name,
                'op': copy.deepcopy(base_case['op']),
                'input_desc': copy.deepcopy(base_case['input_desc']),
                'output_desc': copy.deepcopy(base_case['output_desc'])}
            for (index, shape) in enumerate(node['input_shape']):
                self._check_shape_valid(shape, node['layer'])
                new_base_case['input_desc'][index]['shape'].append(shape)
            for (index, shape) in enumerate(node['output_shape']):
                self._check_shape_valid(shape, node['layer'])
                new_base_case['output_desc'][index]['shape'].append(shape)
            for (index, dtype) in enumerate(node['input_dtype']):
                new_base_case['input_desc'][index]['type'] = dtype
            for (index, dtype) in enumerate(node['output_dtype']):
                new_base_case['output_desc'][index]['type'] = dtype
        except KeyError as e:
            utils.print_error_log("Failed to create case. %s" % e)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
        return new_base_case

    @staticmethod
    def _check_shape_valid(shape, layer_name):
        for dim in shape:
            if not isinstance(dim, int):
                utils.print_error_log(
                    'The input shape(%s) of layer(%s) is invalid. Please '
                    'check the model or try to change the "placeholder" '
                    'shape to fix the problem.' % (shape, layer_name))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
            if dim <= 0:
                utils.print_error_log(
                    'The input shape(%s) of layer(%s) must be greater than 0. '
                    'Please check the model or try to change the "placeholder"'
                    ' shape to fix the problem.' % (shape, layer_name))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _update_attr_from_model(self, base_case, node, new_base_case):
        if 'attr' in base_case and 'attr' in node:
            new_attr_list = []
            for (index, attr) in enumerate(base_case['attr']):
                for item in node['attr']:
                    if item['name'] == attr['name']:
                        new_attr = {'name': item['name'], 'type': attr['type'],
                                    'value': item['value']}
                        utils.check_attr_value_valid(new_attr)
                        new_attr_list.append(new_attr)
                    if item['name'] == "data_format":
                        format_value = item['value'].replace('\"', "").strip()
                        self._update_format_from_model(format_value,
                                                       new_base_case)
            if len(new_attr_list) > 0:
                new_base_case['attr'] = new_attr_list

    @staticmethod
    def _update_format_from_model(format_value, new_base_case):
        for input_format in new_base_case['input_desc']:
            if 'format' in input_format:
                input_format['format'] = [format_value]
        for output_format in new_base_case['output_desc']:
            if 'format' in output_format:
                output_format['format'] = [format_value]

    def _generate_base_case_from_model(self, base_case):
        nodes = get_model_nodes(self.args, self.op_type)
        if len(nodes) == 0:
            utils.print_error_log(
                "There is no \"%s\" operator in the tf model. " % self.op_type)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

        base_case_list = list()
        for i, node in enumerate(nodes):
            # update input and output
            new_base_case = self._update_input_output_from_model(base_case,
                                                                 node)
            if new_base_case:
                # update attr
                self._update_attr_from_model(base_case, node, new_base_case)
                base_case_list.append(new_base_case)
        return base_case_list

    def generate(self):
        """
        generate case.json from .ini file
        """
        # check path valid
        self.check_argument_valid()

        # parse ini to json
        self._parse_ini_to_json()

        # check json valid
        self._check_op_info()
        base_case = self._generate_base_case()
        file_name = '%s_case_%s.json' \
                    % (self.op_type.replace('/', '_'),
                       time.strftime("%Y%m%d%H%M%S",
                                     time.localtime(time.time())))
        json_path = os.path.join(self.output_path, file_name)

        try:
            with os.fdopen(os.open(json_path, utils.WRITE_FLAGS,
                                   utils.WRITE_MODES), 'w+') as file_object:
                file_object.write(
                    json.dumps(base_case, sort_keys=False, indent=4))
        except IOError as io_error:
            utils.print_error_log(
                'Failed to generate file %s. %s' % (json_path, str(io_error)))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_WRITE_FILE_ERROR)
        utils.print_info_log(
            "Generate test case file %s successfully." % json_path)
