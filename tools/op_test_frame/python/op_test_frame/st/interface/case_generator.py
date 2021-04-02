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
    import re
    import importlib
    from . import utils
    from . import dynamic_handle
    from .model_parser import get_model_nodes
except ImportError as import_error:
    sys.exit(
        "[case_generator] Unable to import module: %s." % str(import_error))

INI_INPUT = 'input'
INI_OUTPUT = 'output'
PY_INPUT_OUTPUT = ['inputs', 'outputs']
OP_NAME = 'op_name'
REQUIRED_OP_INFO_KEYS = ["paramType", "name"]
PARAM_TYPE_VALID_VALUE = ["dynamic", "optional", "required"]


class CaseGenerator:
    """
    the class for design test case.
    """

    def __init__(self, args):
        self.input_file_path = os.path.realpath(args.input_file)
        self.output_path = os.path.realpath(args.output_path)
        self.op_info = {}
        self.op_type = ""
        self.args = args

    def check_argument_valid(self):
        """
        check input argument valid
        """
        if os.path.splitext(self.input_file_path)[-1] \
                not in utils.INPUT_SUFFIX_LIST:
            utils.print_error_log(
                'The file "%s" is invalid, only supports .ini or .py file. '
                'Please modify it.' % self.input_file_path)
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_PATH_ERROR)
        utils.check_path_valid(self.input_file_path)
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
                if default_value_str == "[]":
                    return []
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
            utils.print_warn_log(
                'The default value(%s) is invalid for type(%s). Please modify '
                'the default value in "%s" attr. %s' % (
                    default_value_str, attr_type, attr_name, ex))
        return default_value

    def _parse_ini_to_json(self):
        tbe_ops_info = {}
        with open(self.input_file_path) as ini_file:
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
                                index, line, self.input_file_path))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
                else:
                    key_value = line.split('=')
                    if len(key_value) != 2:
                        utils.print_error_log(
                            'At line %d, "%s" is invalid in file %s, only '
                            'supports "xx.yy=zz". Please modify it.' % (
                                index, line, self.input_file_path))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
                    keys = key_value[0].split('.')
                    if len(keys) != 2:
                        utils.print_error_log(
                            'At line %d, "%s" is invalid in file %s, only '
                            'supports "xx.yy=zz". Please modify it.' % (
                                index, line, self.input_file_path))
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
                    len(tbe_ops_info), self.input_file_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

    def _parse_py_to_json(self):
        expect_func_file = self.input_file_path
        sys.path.append(os.path.dirname(expect_func_file))
        py_file = os.path.basename(expect_func_file)
        module_name, _ = os.path.splitext(py_file)
        utils.print_info_log("Start to import {} in {}.".format(
            module_name, py_file))
        class_name = "{}op_info".format(module_name.rstrip("impl"))
        try:
            params = importlib.import_module(module_name)
            mindspore_ops_info = getattr(params, class_name)
        except NameError as name_error:
            utils.print_error_log(
                '%s in %s, please modify it.' % (
                    name_error, self.input_file_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        except ValueError as value_error:
            utils.print_error_log(
                '%s in %s, please modify it.' % (
                    value_error, self.input_file_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)

        if mindspore_ops_info.get(OP_NAME) is None:
            utils.print_warn_log("The op_name is null, please modify it.")
        self.op_type = mindspore_ops_info.get(OP_NAME)
        if mindspore_ops_info.get('attr'):
            attr_name = 'attr_{}'.format(mindspore_ops_info['attr'][0]['name'])
            self.op_info[attr_name] = mindspore_ops_info['attr'][0]
        count_input = 0
        for key in PY_INPUT_OUTPUT:
            op_info = mindspore_ops_info.get(key)
            for index in op_info:
                if not index.get('name'):
                    utils.print_error_log(
                        'This %s of this operator is null, '
                        'please modify it.' % key)
                    raise utils.OpTestGenException(
                        utils.OP_TEST_GEN_INVALID_DATA_ERROR)
                key0 = index.get('name')
                info = "{}{}".format(key[:-1], count_input)
                if info not in self.op_info:
                    self.op_info[info] = {}
                self.op_info[info]['name'] = key0
                self.op_info[info]['paramType'] = index.get('param_type')
                self.op_info[info]['shape'] = index.get('shape')

                dtype = []
                for value in mindspore_ops_info.get('dtype_format'):
                    if not value[count_input]:
                        utils.print_error_log(
                            'The dtype_format of this opeartor is null, '
                            'please modify it')
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_INVALID_DATA_ERROR)
                    dtype.append(value[count_input][0])
                dtypes = ','.join(dtype)
                self.op_info[info]['dtype'] = dtypes
                count_input += 1

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
                utils.print_warn_log(
                    'The value(%s) of "%s" is invalid. '
                    'Only supports %s. Please modify it.' % (
                        value, op_info_key, support_list))

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
                    if self.input_file_path.endswith(".py"):
                        self._check_op_info_list_valid(
                            dtype_list,
                            list(utils.DTYPE_TO_MINDSPORE_MAP.keys()),
                            op_info_key + '.dtype')
                    else:
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
                        format_list, list(utils.FORMAT_ENUM_MAP.keys()),
                        op_info_key + '.format')

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
        if not self._is_aicpu_op():
            if 'type' not in value:
                utils.print_error_log(
                    'The "%s" is missing "type". Please modify it.' % key)
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)
            self._check_op_info_list_valid(
                [value.get('type')], 
                list(utils.ATTR_TYPE_MAP.keys()), key + '.type')
        # defaultValue is None
        data_type = None
        if value.get('type') is not None:
            data_type = utils.ATTR_TYPE_MAP.get(value.get('type'))
        default_value = None
        if 'defaultValue' in value and data_type is not None:
            default_value = self._get_default_attr_value(
                value.get('type'), value.get('defaultValue'), name)
        return {'name': name,
                'type': data_type,
                'value': default_value}

    def _check_desc_valid(self, base_case, key):
        if len(base_case[key]) == 0:
            utils.print_error_log(
                'The number of %s is zero in file %s. Please modify it.'
                % (key[:-5], self.input_file_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR)

    @staticmethod
    def _delete_comments(content):
        no_comments_content = re.sub(r'//.*|/\*(\s|.)*?\*/', '', content)
        return no_comments_content.strip()

    @staticmethod
    def _check_op_proto_path_valid(path):
        if not os.path.exists(path):
            utils.print_warn_log(
                'The path {} does not exist. Please check whether '
                'the path exists.'.format(path))
            return False
        if not os.access(path, os.R_OK):
            utils.print_warn_log('The path {} does not have permission to '
                                 'read. Please check the path permission.'
                                 .format(path))
            return False
        return True

    def _parse_name_from_op_proto_file(self, op_key, start_str):
        # regular match to get input/output/attr name.
        head_file = utils.fix_name_lower_with_under(self.op_type) + '.h'
        try:
            # parse sample as INPUT(x, TensorType({DT_FLOAT}))
            return re.findall(r'[(](.*?)[,]', start_str)[0]
        except IndexError as index_error:
            utils.print_warn_log(
                "Can't parse operator information of %s in %s,"
                "please check." % (op_key, head_file))
            return

    def _parse_aicpu_input_output_info(self, line, input_count, output_count):
        for op_key in utils.IN_OUT_OP_KEY_MAP.keys():
            if line.startswith(op_key):
                # find keyword.
                start_str = line[line.find(op_key):]
                # match name.
                name = self._parse_name_from_op_proto_file(op_key, start_str)
                if name is None:
                    break
                # match type.
                input_tensor_type = start_str[
                                    start_str.find("{") + 1:start_str.find("}")]

                op_info_key = ''
                # record the input information of key
                if utils.IN_OUT_OP_KEY_MAP.get(op_key) == 'input':
                    input_count += 1
                    op_info_key = '%s%s' % (
                        utils.IN_OUT_OP_KEY_MAP.get(op_key), input_count)
                # record the output information of key
                if utils.IN_OUT_OP_KEY_MAP.get(op_key) == 'output':
                    output_count += 1
                    op_info_key = '%s%s' % (
                        utils.IN_OUT_OP_KEY_MAP.get(op_key), output_count)
                if op_info_key not in self.op_info:
                    self.op_info[op_info_key] = {}
                if op_info_key:
                    input_tensor_type = input_tensor_type.replace(
                        utils.NEW_LINE_MARK, utils.EMPTY)
                    input_tensor_type = input_tensor_type.replace(
                        utils.SPACE, utils.EMPTY)
                    dtype_list = list(set(input_tensor_type.split(
                        utils.COMMA)))
                    self._check_op_info_list_valid(dtype_list, list(
                        utils.DTYPE_TO_TYPE_MAP.keys()), INI_INPUT + '.dtype')
                    self.op_info[op_info_key]['name'] = name
                    self.op_info[op_info_key][
                        'dtype'] = input_tensor_type.replace(' ', '')
                    self.op_info[op_info_key]['format'] = ''
        return input_count, output_count

    def _parse_aicpu_attr(self, line):
        for op_key in utils.AICPU_ATTR_LIST:
            if line.startswith(op_key):
                # find keyword.
                attr_str = line[line.find(op_key):]
                # match name.
                name = self._parse_name_from_op_proto_file(op_key, attr_str)
                if name is None:
                    break
                attr_name = 'attr_%s' % name
                if attr_name not in self.op_info:
                    self.op_info[attr_name] = {}
                self.op_info[attr_name]['name'] = name
                # obtain type txt.
                type_attrs = attr_str[attr_str.find(",") + 1:].strip()
                # match type consider different attr.
                if op_key == 'ATTR':
                    # example: .ATTR(attr1, Float, 0.001)
                    try:
                        attr_tensor_type = type_attrs.split(',', 1)[0]
                        defaultvalue = type_attrs.split(',', 1)[1]
                    except IndexError as index_error:
                        utils.print_warn_log(
                            "Can't parse operator information of %s, "
                            "please check." % attr_name)
                        return
                    format_default_value = utils.format_dict_to_list(
                        defaultvalue)
                    default_value = format_default_value.replace(
                        utils.NEW_LINE_MARK, utils.EMPTY)
                    self.op_info[attr_name]['defaultValue'] = \
                        default_value.replace(
                            utils.QUOTATION_MARK, utils.EMPTY)
                else:
                    # example: .REQUIRED_ATTR(attr2, Int)
                    attr_tensor_type = type_attrs.split(')')[0]
                # check invalid for attr type.
                self._check_op_info_list_valid(
                    [attr_tensor_type],
                    list(utils.OP_PROTO_PARSE_ATTR_TYPE_MAP.keys()),
                    'attr_' + name + '.type')
                self.op_info[attr_name]['type'] = \
                    utils.OP_PROTO_PARSE_ATTR_TYPE_MAP.get(attr_tensor_type)

    def _parse_aicpu_op_proto(self, file_path):
        # read op_name.h as an txt.
        op_name_h_text = utils.read_file(file_path)
        if op_name_h_text:
            # find keyword 'REG_OP'.
            reg_op_info_str = op_name_h_text[op_name_h_text.find('REG_OP'):]
            reg_op_info_str = reg_op_info_str.strip()
            if reg_op_info_str.startswith('REG_OP'):
                # delete code comments in op_name.h and format.
                no_comments_content = self._delete_comments(reg_op_info_str)
                new_line = no_comments_content.replace('\n', utils.EMPTY)\
                    .replace('\r', utils.EMPTY) \
                    .replace('\t', utils.EMPTY)
                pattern = re.compile(utils.SPACE)
                line = pattern.sub(utils.EMPTY, new_line)
                line_point_list = line.split(').')
                # record the number of input and output.
                input_count = 0
                output_count = 0
                for line in line_point_list:
                    line = line.strip()
                    # parse op_name and type of input or output.
                    input_count, output_count = \
                        self._parse_aicpu_input_output_info(
                            line, input_count, output_count)
                    # parse op_name and type of attr.
                    self._parse_aicpu_attr(line)
                return True
            else:
                return False
        return False

    def _find_aicpu_op_proto_path(self):
        # search op_name.h upward through the ini file
        op_proto_path = os.path.realpath(
            os.path.join(os.path.dirname(self.input_file_path),
                         '../../../op_proto'))
        op_proto_file_name = utils.fix_name_lower_with_under(self.op_type)\
                             + '.h'
        op_proto_file_path = os.path.join(op_proto_path, op_proto_file_name)
        # check invalid for the op_name.h file.
        is_file_exist = self._check_op_proto_path_valid(op_proto_file_path)

        is_parse_success = False
        if is_file_exist:
            utils.print_info_log("Start parse %s to obtain operator "
                                 "information." % op_proto_file_path)
            # parse aicpu op_name.h file to obtain op_name and type.
            is_parse_success = self._parse_aicpu_op_proto(op_proto_file_path)
            if is_parse_success:
                utils.print_info_log("Finish parse %s." % op_proto_file_path)
        if not is_file_exist or not is_parse_success:
            utils.print_warn_log("There are unavailable operator information "
                                 "in %s." % op_proto_file_path)
            return False
        return True

    def _generate_aicpu_op_desc(self, value, base_case, op_key):
        varible_name = value.get('name')
        op_format = [] if len(value['format']) == 0 else \
            list(set(value['format'].split(',')))

        if len(value['dtype']) == 0:
            op_dtype = []
        else:
            dtype_list = list(set(value['dtype'].split(',')))
            trans_dtype_list = []
            for dtype_key in dtype_list:
                if dtype_key in utils.DTYPE_TO_TYPE_MAP.keys():
                    trans_dtype_list.append(
                        utils.DTYPE_TO_TYPE_MAP.get(dtype_key))
            op_dtype = trans_dtype_list

        if op_key == 'input_desc':
            op_desc = {'format': op_format,
                       'type': op_dtype,
                       'shape': [],
                       'data_distribute': ['uniform'],
                       'value_range': [[0.1, 1.0]]}
        if op_key == 'output_desc':
            op_desc = {'format': op_format,
                       'type': op_dtype,
                       'shape': []}
        if varible_name is not None:
            op_desc.update({'name': varible_name})
        base_case[op_key].append(op_desc)

    def _generate_aicpu_base_case(self):
        # find aicpu op_proto op_name.h file, if it existed, parse it.
        is_find_op_proto_path = self._find_aicpu_op_proto_path()
        base_case = {'case_name': 'Test_' + self.op_type.replace('/', '_')
                                  + '_001',
                     'op': self.op_type,
                     'input_desc': [],
                     'output_desc': []}
        if is_find_op_proto_path:
            for (key, value) in list(self.op_info.items()):
                if key.startswith(INI_INPUT):
                    self._generate_aicpu_op_desc(value, base_case, 'input_desc')
                elif key.startswith(INI_OUTPUT):
                    self._generate_aicpu_op_desc(value, base_case,
                                                 'output_desc')
                elif key.startswith("attr_"):
                    if 'attr' not in base_case:
                        base_case['attr'] = []
                    base_case['attr'].append(self._make_attr(key, value))
        if not base_case['input_desc']:
            input_desc = {'format': [], 'type': [],
                          'shape': [], 'data_distribute': ['uniform'],
                          'value_range': [[0.1, 1.0]]}
            base_case['input_desc'].append(input_desc)
        if not base_case['output_desc']:
            output_desc = {'format': [], 'type': [],
                           'shape': []}
            base_case['output_desc'].append(output_desc)
        # generate base case from model
        if self.args.model_path != "":
            return self._generate_base_case_from_model(base_case, True)
        return [base_case]

    def _generate_input_desc(self, value, base_case):
        input_name = value.get('name')
        if value.get('paramType') == 'optional':
            input_format = ["UNDEFINED"]
            input_dtype = ["UNDEFINED"]
        else:
            if value.get('paramType') == 'dynamic':
                # dynamic input has "name": input_name0
                input_name = "".join([input_name, '0'])
            input_format = [] if len(value['format']) == 0 else \
                list(set(value['format'].split(',')))
            input_dtype = [] if len(value['dtype']) == 0 else \
                list(set(value['dtype'].split(',')))

        if self.input_file_path.endswith(".py"):
            input_desc = {'type': input_dtype,
                          'shape': [],
                          'data_distribute': ['uniform'],
                          'value_range': [[0.1, 1.0]]}
        else:
            input_desc = {'format': input_format,
                          'type': input_dtype,
                          'shape': [],
                          'data_distribute': ['uniform'],
                          'value_range': [[0.1, 1.0]]}
            input_desc.update({'name': input_name})
        base_case['input_desc'].append(input_desc)

    def _generate_aicore_base_case(self):
        if self.input_file_path.endswith(".py"):
            base_case = {'case_name': 'Test_' + self.op_type.replace('/', '_')
                                      + '_001',
                         'st_mode': 'ms_python_train',
                         'op': self.op_type,
                         'input_desc': [], 'output_desc': []}
        else:
            base_case = {'case_name': 'Test_' + self.op_type.replace('/', '_')
                                      + '_001',
                         'op': self.op_type,
                         'input_desc': [], 'output_desc': []}

        for (key, value) in list(self.op_info.items()):
            if key.startswith(INI_INPUT):
                self._generate_input_desc(value, base_case)
            elif key.startswith(INI_OUTPUT):
                output_name = value.get('name')
                output_format = [] if len(value['format']) == 0 else \
                    list(set(value['format'].split(',')))
                output_dtype = [] if len(value['dtype']) == 0 else \
                    list(set(value['dtype'].split(',')))
                if self.input_file_path.endswith(".py"):
                    output_desc = {'type': output_dtype,
                                   'shape': []}
                else:
                    output_desc = {'format': output_format,
                                   'type': output_dtype,
                                   'shape': []}
                    output_desc.update({'name': output_name})
                base_case['output_desc'].append(output_desc)
            elif key.startswith("attr_"):
                if 'attr' not in base_case:
                    base_case['attr'] = []
                base_case['attr'].append(self._make_attr(key, value))
        self._check_desc_valid(base_case, 'output_desc')

        # generate base case from model
        if self.args.model_path != "":
            return self._generate_base_case_from_model(base_case)
        return [base_case]

    def _update_aicpu_io_from_model(self, node):
        # The node won't be None and must has the keys like 'layer'...
        layer_name = node.get('layer').replace('/', '_')
        new_base_case = {'case_name': 'Test_' + layer_name,
                         'op': self.op_type,
                         'input_desc': [],
                         'output_desc': []}
        for (index, dtype) in enumerate(node.get('input_dtype')):
            input_desc = {'format': ['ND'],
                          'type': dtype,
                          'shape': node.get('input_shape')[index],
                          'data_distribute': ['uniform'],
                          'value_range': [[0.1, 1.0]]}
            new_base_case['input_desc'].append(input_desc)
        for (index, dtype) in enumerate(node.get('output_dtype')):
            output_desc = {'format': ['ND'],
                           'type': dtype,
                           'shape': node.get('output_shape')[index]}
            new_base_case['output_desc'].append(output_desc)
        return new_base_case

    def _update_aicore_io_from_model(self, base_case, node):
        # The node won't be None and must has the keys like 'layer'...
        # The base_case won't be None and must has the keys like 'input..'
        layer_name = node.get('layer').replace('/', '_')
        try:
            # when the length not equal, skip to next layer
            if len(node.get('input_shape')) != len(
                    base_case.get('input_desc')):
                utils.print_warn_log(
                    'The \"%s\" layer is skipped, because its number of '
                    'inputs(%d) is different from '
                    'that(%d) specified in the .ini file.Ignore if you have '
                    'change the "placeholder" shape.'
                    % (layer_name, len(node.get('input_shape')),
                       len(base_case.get('input_desc')),
                       ))
                return None
            if len(node.get('output_shape')) != len(
                    base_case.get('output_desc')):
                utils.print_warn_log(
                    'The \"%s\" layer is skipped, because its number of '
                    'outputs(%d) is different from '
                    'that(%d) specified in the .ini file. Ignore if you have '
                    'changed the "Placeholder" shape.'
                    % (len(node.get('output_shape')),
                       layer_name,
                       len(base_case.get('output_desc')),))
                return None

            new_base_case = {
                'case_name': 'Test_%s' % layer_name,
                'op': copy.deepcopy(base_case.get('op')),
                'input_desc': copy.deepcopy(base_case.get('input_desc')),
                'output_desc': copy.deepcopy(base_case.get('output_desc'))}
            for (index, shape) in enumerate(node.get('input_shape')):
                self._check_shape_valid(shape, node.get('layer'))
                new_base_case['input_desc'][index]['shape'].append(shape)
            for (index, shape) in enumerate(node['output_shape']):
                self._check_shape_valid(shape, node.get('layer'))
                new_base_case['output_desc'][index]['shape'].append(shape)
            for (index, dtype) in enumerate(node['input_dtype']):
                new_base_case['input_desc'][index]['type'] = dtype
            for (index, dtype) in enumerate(node['output_dtype']):
                new_base_case['output_desc'][index]['type'] = dtype
        except KeyError as error:
            utils.print_error_log("Failed to create case. %s" % error)
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

    def _update_aicpu_attr_from_model(self, node, new_base_case):
        # The node won't be None and must has the key 'attr'.
        # IF the node is empty, node:{'attr':[]}
        if node.get('attr'):
            for item in node.get('attr'):
                if item.get('name') == "data_format":
                    format_value = item.get('value').replace('\"', "").strip()
                    self._update_format_from_model(format_value,
                                                   new_base_case)

    def _get_info_from_model(self, item, attr, new_attr_list, new_base_case):
        """get information from model"""
        node_attr_name = item['name']
        base_attr_name = attr.get('name')
        if node_attr_name == base_attr_name:
            attr_value = item.get('value')
            # format attr_value when attr's type is 'string'.
            if attr.get('type') == 'string':
                attr_format_value = item.get('value').replace(
                    utils.QUOTATION_MARK, utils.EMPTY)
                attr_value = attr_format_value.replace(
                    utils.SPACE, utils.EMPTY)
            new_attr = {'name': node_attr_name,
                        'type': attr['type'],
                        'value': attr_value}
            utils.check_attr_value_valid(new_attr)
            new_attr_list.append(new_attr)
        if node_attr_name == "data_format":
            format_value = item['value'].replace('\"', "").strip()
            self._update_format_from_model(format_value,
                                           new_base_case)

    def _update_aicore_attr_from_model(self, base_case, node, new_base_case):
        if 'attr' in base_case and 'attr' in node:
            new_attr_list = []
            for (_, attr) in enumerate(base_case.get('attr')):
                for item in node.get('attr'):
                    self._get_info_from_model(
                        item, attr, new_attr_list, new_base_case)
            if len(new_attr_list) > 0:
                new_base_case['attr'] = new_attr_list

    @staticmethod
    def _update_format_from_model(format_value, new_base_case):
        for input_format in new_base_case.get('input_desc'):
            if 'format' in input_format:
                input_format['format'] = [format_value]
        for output_format in new_base_case.get('output_desc'):
            if 'format' in output_format:
                output_format['format'] = [format_value]

    def _generate_base_case_from_model(self, base_case, is_aicpu=False):
        nodes = get_model_nodes(self.args, self.op_type)
        if len(nodes) == 0:
            utils.print_warn_log(
                "\"%s\" operator not found. Failed to get the "
                "operator info from the model. Please check the model." %
                self.op_type)
            utils.print_info_log(
                "Continue generating the case json file based on the .ini "
                "file.")
            return [base_case]
        base_case_list = list()
        for _, node in enumerate(nodes):
            # update input and output
            if is_aicpu:
                new_base_case = self._update_aicpu_io_from_model(node)
                # the new_base_case won't be None
                self._update_aicpu_attr_from_model(node, new_base_case)
                base_case_list.append(new_base_case)
            else:
                new_base_case = self._update_aicore_io_from_model(base_case,
                                                                  node)
                if new_base_case:
                    self._update_aicore_attr_from_model(base_case, node,
                                                        new_base_case)
                    base_case_list.append(new_base_case)
        if not base_case_list:
            utils.print_warn_log("No match for the layer in the model. Failed "
                                 "to get the operator info from the model. "
                                 "Please check the model and .ini file.")
            utils.print_info_log(
                "Continue generating the case json file based on the .ini "
                "file.")
            return [base_case]
        return base_case_list

    def _is_aicpu_op(self):
        # in the aicpu ini file , "opInfo.kernelSo=libaicpu_kernels.so"
        # in the aicore ini file, "input0.name=x1"
        if self.op_info.get('opInfo'):
            if self.op_info.get('opInfo').get('kernelSo'):
                return True
        return False

    def generate(self):
        """
        generate case.json from .ini or .py file
        """
        # check path valid
        self.check_argument_valid()
        if self.input_file_path.endswith(".ini"):
            # parse ini to json
            self._parse_ini_to_json()
        elif self.input_file_path.endswith(".py"):
            # parse .py to json
            self._parse_py_to_json()

        if self._is_aicpu_op():
            # generate aicpu base case of case.json
            base_case = self._generate_aicpu_base_case()
        else:
            # check json valid
            self._check_op_info()
            base_case = self._generate_aicore_base_case()
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
