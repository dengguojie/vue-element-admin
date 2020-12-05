#!/usr/bin/env python
# coding=utf-8
"""
Function:
CaseDesign class
This class mainly involves parse
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

try:
    import os
    import sys
    import itertools
    from . import utils
    from . import st_report
    from . import op_st_case_info
except ImportError as import_error:
    sys.exit("[case_design] Unable to import module: %s." % str(import_error))

OP = 'op'
INPUT_DESC = 'input_desc'
OUTPUT_DESC = 'output_desc'
ATTR = 'attr'
CASE_NAME = 'case_name'
ST_MODE = 'st_mode'
REQUIRED_KEYS = [OP, INPUT_DESC, OUTPUT_DESC, CASE_NAME]
ATTR_REQUIRED_KEYS = ["name", "type", "value"]
SUPPORT_TYPE_LIST = list(utils.ATTR_TYPE_MAP.values())

INPUT_CROSS_LIST = ['format', 'type', 'shape', 'data_distribute', 'value_range']
OUTPUT_CROSS_LIST = ['format', 'type', 'shape']
MS_INPUT_CROSS_LIST = ['type', 'shape', 'data_distribute', 'value_range']
MS_OUTPUT_CROSS_LIST = ['type', 'shape']


class CaseDesign:
    """
    the class for design test case.
    """

    def __init__(self, json_path_list, case_name_list, report):
        self.json_path_list = json_path_list.split(',')
        if case_name_list == 'all':
            self.case_name_list = None
        else:
            self.case_name_list = case_name_list.split(',')
        self.current_json_path = ''
        self.case_name_to_json_file_map = {}
        self.multi = False
        self.report = report

    def check_argument_valid(self):
        """
        check input json file valid
        """
        json_file_list = []
        for json_path in self.json_path_list:
            json_path = os.path.realpath(json_path)
            if not json_path.endswith(".json"):
                utils.print_error_log(
                    'The file "%s" is invalid, only supports .json file. '
                    'Please modify it.' % json_path)
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_PATH_ERROR)
            utils.check_path_valid(json_path)
            json_file_list.append(json_path)
        self.json_path_list = json_file_list

    def _check_key_exist(self, json_obj, key, tensor):
        if key not in json_obj:
            utils.print_error_log(
                'There is no key "%s" for "%s". Please modify it in file %s.'
                % (key, tensor, self.current_json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_range_value_valid(self, range_value):
        if len(range_value) != 2:
            utils.print_error_log('The value(%s) of "range_value" is not [min,'
                                  'max]. Please modify it in file %s.'
                                  % (range_value, self.current_json_path))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for value in range_value:
            if not isinstance(value, float) and not isinstance(value, int):
                utils.print_error_log(
                    'The value(%s) of "range_value" is not int or float'
                    '. Please modify it in file %s.'
                    % (range_value, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        if range_value[1] < range_value[0]:
            utils.print_error_log(
                'In %s the maximum value is less than the minimum value '
                'Please modify it in file %s.' % (
                    range_value, self.current_json_path))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_list_list_valid(self, json_obj, key, tensor):
        self._check_key_exist(json_obj, key, tensor)
        value_list = []
        if isinstance(json_obj[key], list):
            check_type = None
            for item in json_obj[key]:
                if isinstance(item, (tuple, list)):
                    current_type = list
                else:
                    current_type = int
                if check_type is None:
                    check_type = current_type
                else:
                    if check_type != current_type:
                        utils.print_error_log(
                            'The value (%s) is invalid. The key "%s" for "%s" '
                            'only supports [] or [[]]. Please modify it '
                            'in file %s.' % (json_obj[key], key, tensor,
                                             self.current_json_path))
                        raise utils.OpTestGenException(
                            utils.OP_TEST_GEN_INVALID_DATA_ERROR)
            if check_type == list:
                value_list = json_obj[key]
            else:
                value_list.append(json_obj[key])
        else:
            utils.print_error_log(
                'The value (%s) is invalid. The key "%s" for "%s" only '
                'supports [] or [[]]. Please modify it in file %s.' % (
                    json_obj[key], key, tensor, self.current_json_path))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        return value_list

    def _check_list_str_valid(self, json_obj, key, support_list, tensor):
        self._check_key_exist(json_obj, key, tensor)
        value_list = []
        if isinstance(json_obj[key], str):
            value_list.append(json_obj[key])
        elif isinstance(json_obj[key], (tuple, list)):
            value_list = json_obj[key]
        else:
            utils.print_error_log(
                'The value (%s) is invalid. The key "%s" for "%s" only '
                'supports string or [string]. Please modify it in file %s.'
                % (json_obj[key], key, tensor, self.current_json_path))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        if len(value_list) == 0:
            utils.print_error_log(
                'The value of "%s" for "%s" is empty. Only supports %s.'
                ' Please modify it in file %s.' %
                (key, tensor, support_list, self.current_json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for item in value_list:
            if not isinstance(item, str):
                utils.print_error_log(
                    'The value (%s) is invalid. The key "%s" for "%s" only'
                    ' supports string or [string]. Please modify it in file %s.'
                    % (item, key, tensor, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
            if item == '':
                utils.print_error_log(
                    'The value(%s) of "%s" for "%s" contains empty string. '
                    'Only supports %s. Please modify it in file %s.' %
                    (json_obj[key], key, tensor, support_list,
                     self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
            if item not in support_list:
                utils.print_error_log(
                    'The value(%s) of "%s" for "%s" does not support. '
                    'Only supports %s. Please modify it in file %s.' %
                    (item, key, tensor, support_list, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        return value_list

    def _check_name_type_valid(self, attr, key):
        if not isinstance(attr[key], str):
            utils.print_error_log(
                'The value (%s) is invalid. The key "%s" for "attr" only '
                'supports string. Please modify it in file %s.'
                % (attr[key], key, self.current_json_path))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        if attr[key] == "":
            utils.print_error_log(
                'The value of "%s" for "attr" is empty. Please modify '
                'it in file %s.' % (key, self.current_json_path))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        if key == 'type' and attr[key] not in SUPPORT_TYPE_LIST:
            utils.print_error_log(
                'The value(%s) of "type" does not support. Only supports %s. '
                'Please modify it in file %s.' % (attr[key], SUPPORT_TYPE_LIST,
                                                  self.current_json_path))
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_attr_valid(self, json_obj):
        if ATTR not in json_obj:
            return []
        if not isinstance(json_obj[ATTR], (tuple, list)):
            utils.print_error_log(
                'The value (%s) of "attr" is not list. Please modify it in file'
                ' %s.' % (json_obj[ATTR], self.current_json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        name_list = []
        for attr in json_obj[ATTR]:
            self._check_required_key_valid(attr, ATTR_REQUIRED_KEYS, 'attr')
            self._check_name_type_valid(attr, 'name')
            if attr['name'] not in name_list:
                name_list.append(attr['name'])
            else:
                utils.print_error_log(
                    'The %s already exists. Please modify or remove the '
                    'redundant key in file %s.'
                    % (attr['name'], self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
            self._check_name_type_valid(attr, 'type')
            utils.check_attr_value_valid(attr)
        return json_obj[ATTR]

    def _check_shape_valid(self, shape):
        for dim in shape:
            if not isinstance(dim, int):
                utils.print_error_log(
                    'The value(%s) of "shape" is not int. Please modify it in '
                    'file %s.' % (shape, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
            if dim <= 0:
                utils.print_error_log(
                    'The value(%s) of "shape" must be greater than 0. Please '
                    'modify it in file %s.' % (shape, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _make_input_desc_list(self, json_obj):
        input_desc_list = []
        if len(json_obj[INPUT_DESC]) == 0:
            utils.print_error_log(
                'The value of "input_desc" is empty. Please modify it in '
                'file %s.' % self.current_json_path)
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for input_desc in json_obj[INPUT_DESC]:
            format_list = self._check_list_str_valid(
                input_desc, 'format', utils.FORMAT_LIST, INPUT_DESC)
            type_list = self._check_list_str_valid(
                input_desc, 'type', list(utils.DTYPE_TO_NUMPY_MAP.keys()),
                INPUT_DESC)
            shape_list = self._check_list_list_valid(
                input_desc, 'shape', INPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            if 'data_distribute' in input_desc:
                data_distribute_list = self._check_list_str_valid(
                    input_desc, 'data_distribute',
                    utils.DATA_DISTRIBUTION_LIST, INPUT_DESC)
            else:
                data_distribute_list = ['uniform']
            if 'value_range' in input_desc:
                value_range_list = self._check_list_list_valid(
                    input_desc, 'value_range', INPUT_DESC)
                for item in value_range_list:
                    self._check_range_value_valid(item)
            else:
                value_range_list = [[0.1, 1.0]]
            one_input_desc = {'format': format_list, 'type': type_list,
                              'shape': shape_list,
                              'value_range': value_range_list,
                              'data_distribute': data_distribute_list}
            input_desc_list.append(one_input_desc)
            for item in one_input_desc.values():
                if len(item) > 1:
                    self.multi = True
        return input_desc_list

    def _make_input_desc_list_ms(self, json_obj):
        input_desc_list = []
        if len(json_obj[INPUT_DESC]) == 0:
            utils.print_error_log(
                'The value of "input_desc" is empty. Please modify it in '
                'file %s.' % self.current_json_path)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for input_desc in json_obj[INPUT_DESC]:
            type_list = self._check_list_str_valid(
                input_desc, 'type', list(utils.DTYPE_TO_MINDSPORE_MAP.keys()),
                INPUT_DESC)
            shape_list = self._check_list_list_valid(
                input_desc, 'shape', INPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            if 'data_distribute' in input_desc:
                data_distribute_list = self._check_list_str_valid(
                    input_desc, 'data_distribute',
                    utils.DATA_DISTRIBUTION_LIST, INPUT_DESC)
            else:
                data_distribute_list = ['uniform']
            if 'value_range' in input_desc:
                value_range_list = self._check_list_list_valid(
                    input_desc, 'value_range', INPUT_DESC)
                for item in value_range_list:
                    self._check_range_value_valid(item)
            else:
                value_range_list = [[0.1, 1.0]]
            one_input_desc = {'type': type_list,
                              'shape': shape_list,
                              'value_range': value_range_list,
                              'data_distribute': data_distribute_list}
            input_desc_list.append(one_input_desc)
            for item in one_input_desc.values():
                if len(item) > 1:
                    self.multi = True
        return input_desc_list

    def _make_output_desc_list(self, json_obj):
        output_desc_list = []
        if len(json_obj[OUTPUT_DESC]) == 0:
            utils.print_error_log(
                'The value of "output_desc" is empty. Please modify it in '
                'file %s.' % self.current_json_path)
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for output_desc in json_obj[OUTPUT_DESC]:
            format_list = self._check_list_str_valid(
                output_desc, 'format', utils.FORMAT_LIST, OUTPUT_DESC)
            type_list = self._check_list_str_valid(
                output_desc, 'type', list(utils.DTYPE_TO_NUMPY_MAP.keys()),
                OUTPUT_DESC)
            shape_list = self._check_list_list_valid(
                output_desc, 'shape', OUTPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            one_output_desc = {'format': format_list, 'type': type_list,
                               'shape': shape_list}
            output_desc_list.append(one_output_desc)
            for item in one_output_desc.values():
                if len(item) > 1:
                    self.multi = True
        return output_desc_list

    def _make_output_desc_list_ms(self, json_obj):
        output_desc_list = []
        if len(json_obj[OUTPUT_DESC]) == 0:
            utils.print_error_log(
                'The value of "output_desc" is empty. Please modify it in '
                'file %s.' % self.current_json_path)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for output_desc in json_obj[OUTPUT_DESC]:
            type_list = self._check_list_str_valid(
                output_desc, 'type', list(utils.DTYPE_TO_MINDSPORE_MAP.keys()),
                OUTPUT_DESC)
            shape_list = self._check_list_list_valid(
                output_desc, 'shape', OUTPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            one_output_desc = {'type': type_list,
                               'shape': shape_list}
            output_desc_list.append(one_output_desc)
            for item in one_output_desc.values():
                if len(item) > 1:
                    self.multi = True
        return output_desc_list

    def _check_value_number_valid(self, input_desc_list, output_desc_list):
        key = 'format'
        count = len(input_desc_list[0][key])
        self._check_number_match(key, count, input_desc_list)
        self._check_number_match(key, count, output_desc_list)
        key = 'type'
        count = len(input_desc_list[0][key])
        self._check_number_match(key, count, input_desc_list)
        self._check_number_match(key, count, output_desc_list)
        key = 'shape'
        count = len(input_desc_list[0][key])
        self._check_number_match(key, count, input_desc_list)
        self._check_number_match(key, count, output_desc_list)

    def _check_value_number_valid_ms(self, input_desc_list, output_desc_list):
        key = 'type'
        count = len(input_desc_list[0][key])
        self._check_number_match(key, count, input_desc_list)
        self._check_number_match(key, count, output_desc_list)
        key = 'shape'
        count = len(input_desc_list[0][key])
        self._check_number_match(key, count, input_desc_list)
        self._check_number_match(key, count, output_desc_list)

    def _check_number_match(self, key, count, desc_list):
        for item in desc_list:
            if count != len(item[key]):
                utils.print_error_log(
                    'The number of "%s" is different for "input_desc"'
                    ' and "output_desc". Please modify it in file %s.'
                    % (key, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    @staticmethod
    def _cross_tensor(tensor_list, cross_key_list):
        total_case_list = []
        for tensor in tensor_list:
            cross_list = []
            for key in cross_key_list:
                cross_list.append(tensor[key])
            cross_list = [list(x) for x in itertools.product(*cross_list)]
            case_list = []
            for case in cross_list:
                cur_params = {cross_key_list[x]: case[x] for x, _ in
                              enumerate(cross_key_list)}
                case_list.append(cur_params)
            total_case_list.append(case_list)
        return total_case_list

    def _check_required_key_valid(self, json_obj, required_key_list, tensor):
        if ST_MODE in json_obj:
            required_key_list.append(ST_MODE)
        missing_keys = []
        for key in required_key_list:
            if key not in json_obj:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            utils.print_error_log(
                'The "%s" is missing key: %s. Please modify it in file %s.' % (
                    tensor, missing_keys, self.current_json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def generate_cases(self):
        """
        Generate test case by json file
        :return: the list of test case
        """
        case_idx = 1
        total_case_in_file = []
        for json_path in self.json_path_list:
            utils.print_info_log('Start to create sub test cases for %s.'
                                 % json_path)
            self.current_json_path = json_path
            # load json file
            json_object = utils.load_json_file(json_path)
            # parse json object
            for json_obj in json_object:
                self.multi = False
                self._check_required_key_valid(json_obj, REQUIRED_KEYS, 'case')
                # skip the case name not in case_name_list
                if self.case_name_list and \
                        json_obj[CASE_NAME] not in self.case_name_list:
                    continue
                if json_obj[CASE_NAME] in self.case_name_to_json_file_map:
                    utils.print_error_log(
                        'The case name "%s" already exists. Please modify or '
                        'remove the redundant case name in file %s.'
                        % (json_obj[CASE_NAME], self.current_json_path))
                    raise utils.OpTestGenException(
                        utils.OP_TEST_GEN_INVALID_DATA_ERROR)
                self.case_name_to_json_file_map[json_obj[CASE_NAME]] = json_path

                if ST_MODE in json_obj:
                    input_desc_list = self._make_input_desc_list_ms(json_obj)
                    output_desc_list = self._make_output_desc_list_ms(json_obj)
                    attr_list = self._check_attr_valid(json_obj)
                    self._check_value_number_valid_ms(input_desc_list,
                                                      output_desc_list)
                    input_case_list = self._cross_tensor(
                        input_desc_list, MS_INPUT_CROSS_LIST)
                    output_case_list = self._cross_tensor(
                        output_desc_list, MS_OUTPUT_CROSS_LIST)
                else:
                    input_desc_list = self._make_input_desc_list(json_obj)
                    output_desc_list = self._make_output_desc_list(json_obj)
                    attr_list = self._check_attr_valid(json_obj)
                    self._check_value_number_valid(
                        input_desc_list, output_desc_list)
                    input_case_list = self._cross_tensor(
                        input_desc_list, INPUT_CROSS_LIST)
                    output_case_list = self._cross_tensor(
                        output_desc_list, OUTPUT_CROSS_LIST)
                count = len(input_case_list[0])
                prefix = json_obj[CASE_NAME].replace('/', '_') + '_'
                if self.multi:
                    if ST_MODE in json_obj:
                        prefix += 'sub_'
                    else:
                        prefix += 'sub_case_'
                else:
                    prefix += 'case_'
                pyfile, function = self._check_expect_output_param(json_obj)
                for index in range(count):
                    if ST_MODE in json_obj:
                        case = {OP: json_obj[OP], ST_MODE: json_obj[ST_MODE],
                                INPUT_DESC: [], OUTPUT_DESC: [],
                                'case_name': prefix + '%d' % case_idx}
                    else:
                        case = {OP: json_obj[OP],
                                INPUT_DESC: [], OUTPUT_DESC: [],
                                'case_name': prefix + '%03d' % case_idx}
                    if len(attr_list) > 0:
                        case[ATTR] = attr_list
                    for input_case in input_case_list:
                        case[INPUT_DESC].append(input_case[index])
                    output_index = index
                    if index >= len(output_case_list[0]):
                        output_index = index % len(output_case_list)
                    for output_case in output_case_list:
                        case[OUTPUT_DESC].append(output_case[output_index])
                    self._parse_expect_output_param(case, pyfile, function)
                    case_idx += 1
                    total_case_in_file.append(case)
                    # deal with report
                    case_info = op_st_case_info.OpSTCase(case['case_name'],
                                                         case)
                    st_case_trace = op_st_case_info.OpSTCaseTrace(case_info)
                    case_rpt = st_report.OpSTCaseReport(st_case_trace)
                    self.report.add_case_report(case_rpt)
                utils.print_info_log('Create %d sub test cases for %s.'
                                     % (count, json_obj[CASE_NAME]))
        return total_case_in_file

    @staticmethod
    def _check_expect_output_param(json_obj):
        expect_str = json_obj.get("calc_expect_func_file")
        pyfile = None
        function = None
        if expect_str:
            length = len(expect_str.split(":"))
            if length == 2:
                pyfile, function = expect_str.split(":")
                utils.print_info_log("The expect data generate python file:%s." % pyfile)
                utils.check_path_valid(pyfile)
            elif length == 1:
                pyfile = expect_str
                function = json_obj.get(OP)
                utils.print_info_log("The expect data generate python file:%s." % pyfile)
                utils.check_path_valid(pyfile)
            else:
                utils.print_warn_log("The value of calc_expect_func_file is '%s', is invalid! If no need to compare"
                                     "output data,ignore." % expect_str)
        else:
            expect_str = json_obj.get("calc_expect_func")
            if not expect_str:
                utils.print_warn_log("There is no expect output function in "
                                     "the case json, if no need to compare "
                                     "output data, ignore.")
        return pyfile, function

    @staticmethod
    def _parse_expect_output_param(case, pyfile, function):
        if not pyfile or not function:
            return
        if not pyfile and function:
            case.update({"calc_expect_func": function})
        if pyfile and function:
            case.update({"calc_expect_func_file": pyfile})
            case.update({"calc_expect_func_file_func": function})
        return

    def design(self):
        """
        Design test case by json file.
        :return: the test case list
        """
        # check json path valid
        self.check_argument_valid()

        # design sub test case by json file
        case_list = self.generate_cases()

        if len(case_list) == 0:
            case_info = 'all'
            if self.case_name_list:
                case_info = str(self.case_name_list)
            utils.print_error_log(
                'There is no case to generate for %s. Please modify the case '
                'name argument.' % case_info)
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        return case_list
