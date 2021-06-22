#!/usr/bin/env python
# coding=utf-8
"""
Function:
SubCaseDesignFuzz class
This class mainly involves parse
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

try:
    import os
    import sys
    import importlib
    import copy
    from . import utils
    from . import dynamic_handle
    from . import case_design as CD
    from . import subcase_design as SD
except ImportError as import_error:
    sys.exit("[subcase_design_fuzz] Unable to import module: %s." % str(import_error))


FUZZ_CASE_NUM = 'fuzz_case_num'
FUZZ_FUNCTION = 'fuzz_branch'
MAX_FUZZ_CASE_NUM = 2000


class SubCaseDesignFuzz(SD.SubCaseDesign):
    """
    the class for design test subcase by fuzz.
    """

    def check_fuzz_valid(self, json_obj):
        """
        check number match
        :param json_obj: the json_obj of json object
        :return: fuzz_function
        """
        fuzz_impl_path = json_obj.get(CD.FUZZ_IMPL)
        dir_path = os.path.dirname(self.current_json_path)
        real_fuzz_path = os.path.join(dir_path, fuzz_impl_path)
        if os.path.splitext(real_fuzz_path)[-1] != utils.PY_FILE:
            utils.print_error_log(
                'The fuzz file "%s" is invalid, only supports .py file. '
                'Please modify it.' % fuzz_impl_path)
            raise utils.OpTestGenException(utils.OP_TEST_GEN_INVALID_PATH_ERROR)
        utils.check_path_valid(real_fuzz_path)
        # get fuzz function from fuzz file
        sys.path.append(os.path.dirname(real_fuzz_path))
        fuzz_file = os.path.basename(real_fuzz_path)
        module_name, _ = os.path.splitext(fuzz_file)
        utils.print_info_log("Start to import %s in %s." % (module_name,
                                                            real_fuzz_path))
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, FUZZ_FUNCTION):
                utils.print_error_log('%s has no attribute "%s"' % (real_fuzz_path,
                                                                    FUZZ_FUNCTION))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_PARAM_ERROR)
            fuzz_function = getattr(module, FUZZ_FUNCTION)
        except Exception as ex:
            utils.print_error_log(
                'Failed to execute function "%s" in %s. %s' % (
                    FUZZ_FUNCTION, fuzz_file, str(ex)))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_PARAM_ERROR)
        return fuzz_function

    def _check_fuzz_case_num_valid(self, json_obj):
        fuzz_case_num = json_obj.get(FUZZ_CASE_NUM)
        if isinstance(fuzz_case_num, int):
            if 0 < fuzz_case_num <= MAX_FUZZ_CASE_NUM:
                return fuzz_case_num
            utils.print_error_log(
                'The "%s" is invalid in %s, only supports 1~%s. '
                'Please modify it.' % (FUZZ_CASE_NUM,
                                       self.current_json_path,
                                       MAX_FUZZ_CASE_NUM))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_PARAM_ERROR)
        utils.print_error_log(
            'The "%s" is invalid in %s, only supports integer. '
            'Please modify it.' % (FUZZ_CASE_NUM, self.current_json_path))
        raise utils.OpTestGenException(
            utils.OP_TEST_GEN_INVALID_PARAM_ERROR)

    # pylint: disable=too-many-arguments
    def _check_fuzz_value_valid(self, json_obj, key, support_list, param_type,
                                fuzz_dict, required=True):
        if required:
            self._check_key_exist(json_obj, key, param_type)
        json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                            fuzz_dict)
        if isinstance(json_obj.get(key), (tuple, list)):
            if len(json_obj.get(key)) != 1:
                utils.print_error_log(
                    'The fuzz case, each of the fields can be configured with '
                    'one profile only. Please modify %s in file %s.' % (key, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
            param_value = json_obj.get(key)[0]
        else:
            param_value = json_obj.get(key)
        if support_list is not None and param_value not in support_list:
            utils.print_error_log(
                'The value of "%s" for "%s" does not support. '
                'Only supports %s. Please modify it in file %s.' %
                (key, param_type, support_list, self.current_json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        return param_value

    def _check_fuzz_shape_valid(self, json_obj, key, param_type, fuzz_dict,
                                required=True):
        if required:
            self._check_key_exist(json_obj, key, param_type)
        json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                            fuzz_dict)
        shape_value = json_obj.get(key)
        if isinstance(shape_value, list):
            self._check_shape_valid(shape_value)
            return shape_value
        utils.print_error_log(
            'The value (%s) is invalid. The key "%s" for "%s" only '
            'supports [] in fuzz case. Please modify it in file %s.' % (
                shape_value, key, param_type, self.current_json_path))
        raise utils.OpTestGenException(
            utils.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_fuzz_data_distribute_valid(self, json_obj, key, param_type,
                                          fuzz_dict):
        if key in json_obj and utils.VALUE not in json_obj:
            data_distribute_value = self._check_fuzz_value_valid(
                json_obj, key, SD.WHITE_LISTS.data_distribution_list, param_type,
                fuzz_dict)
        else:
            data_distribute_value = 'uniform'
        return data_distribute_value

    def _check_fuzz_value_range_valid(self, json_obj, key, param_type,
                                      fuzz_dict):
        if key in json_obj and utils.VALUE not in json_obj:
            self._check_key_exist(json_obj, key, param_type)
            json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                                fuzz_dict)
            value_range = json_obj.get(key)
            if isinstance(value_range, list):
                if len(value_range) == 1 and isinstance(value_range[0], list):
                    value_range = value_range[0]
                self._check_range_value_valid(value_range)
            else:
                utils.print_error_log(
                    'The value (%s) is invalid. The key "%s" for "%s" only '
                    'supports [] in fuzz case. Please modify it in file %s.' % (
                        value_range, key, param_type, self.current_json_path))
                raise utils.OpTestGenException(
                    utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        else:
            value_range = [0.1, 1.0]
        return value_range

    def _make_desc_list_ms_fuzz(self, json_obj, fuzz_dict, desc_type):
        desc_list = []
        if len(json_obj[desc_type]) == 0:
            utils.print_error_log(
                'The value of "%s" is empty. Please modify it in '
                'file %s.' % (desc_type, self.current_json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for desc_obj in json_obj[desc_type]:
            type_value = self._check_fuzz_value_valid(
                desc_obj, 'type', SD.WHITE_LISTS.mindspore_type_list,
                desc_type, fuzz_dict)
            shape_value = self._check_fuzz_shape_valid(
                desc_obj, 'shape', desc_type, fuzz_dict)
            if desc_type == CD.INPUT_DESC:
                data_distribute = self._check_fuzz_data_distribute_valid(
                    desc_obj, 'data_distribute', desc_type, fuzz_dict)
                value_range = self._check_fuzz_value_range_valid(
                    desc_obj, 'value_range', desc_type, fuzz_dict)
                one_desc = {'type': type_value,
                            'shape': shape_value,
                            'value_range': value_range,
                            'data_distribute': data_distribute}
                # check whether has value.
                desc_obj = self._replace_fuzz_param(
                    desc_obj, utils.VALUE, desc_type, fuzz_dict)
                self._deal_with_value(desc_obj, one_desc)
            else:
                one_desc = {'type': type_value,
                            'shape': shape_value}
            desc_list.append(one_desc)
        return desc_list

    def _deal_with_ori_filed_data_fuzz(self, json_obj, param_type, fuzz_dict,
                                       one_input_desc):
        ori_shape_value = None
        ori_format_value = self._check_fuzz_value_valid(
            json_obj, 'ori_format', None, param_type, fuzz_dict, required=False)
        if json_obj.get('ori_shape') is not None:
            ori_shape_value = self._check_fuzz_shape_valid(
                json_obj, 'ori_shape', param_type, fuzz_dict, required=False)
        if ori_format_value and ori_shape_value:
            one_input_desc['ori_format'] = ori_format_value
            one_input_desc['ori_shape'] = ori_shape_value
        return one_input_desc

    def _check_fuzz_shape_range_valid(self, json_obj, key, param_type,
                                      fuzz_dict):
        json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                            fuzz_dict)
        shape_range = self._check_list_list_valid(json_obj, key, param_type)
        for item in shape_range:
            self._check_range_value_valid(item, for_shape_range=True)
        return shape_range

    def _deal_with_dynamic_shape_fuzz(self, json_obj, param_type, fuzz_dict,
                                      one_input_desc):
        if not dynamic_handle.check_not_dynamic_shape(json_obj.get('shape')):
            return one_input_desc
        typical_shape = self._check_fuzz_shape_valid(
            json_obj, utils.TYPICAL_SHAPE, param_type, fuzz_dict)
        dynamic_handle.check_typical_shape_valid(typical_shape,
                                                 self.current_json_path)
        one_input_desc[utils.TYPICAL_SHAPE] = typical_shape
        shape_range = self._check_fuzz_shape_range_valid(
            json_obj, utils.SHAPE_RANGE, param_type, fuzz_dict)
        one_input_desc[utils.SHAPE_RANGE] = shape_range
        return one_input_desc

    def _make_desc_list_fuzz(self, json_obj, fuzz_dict, desc_type):
        desc_list = []
        if len(json_obj[desc_type]) == 0:
            if desc_type == CD.INPUT_DESC:
                utils.print_warn_log(
                    'The value of "input_desc" is empty.')
                return desc_list
            utils.print_error_log(
                'The value of "%s" is empty. Please modify it in '
                'file %s.' % (desc_type, self.current_json_path))
            raise utils.OpTestGenException(
                utils.OP_TEST_GEN_INVALID_DATA_ERROR)
        for desc_obj in json_obj[desc_type]:
            format_value = self._check_fuzz_value_valid(
                desc_obj, 'format', list(SD.WHITE_LISTS.format_map.keys()),
                desc_type, fuzz_dict)
            type_value = self._check_fuzz_value_valid(
                desc_obj, 'type', SD.WHITE_LISTS.type_list,
                desc_type, fuzz_dict)
            shape_value = self._check_fuzz_shape_valid(
                desc_obj, 'shape', desc_type, fuzz_dict)
            if desc_type == CD.INPUT_DESC:
                data_distribute = self._check_fuzz_data_distribute_valid(
                    desc_obj, 'data_distribute', desc_type, fuzz_dict)
                value_range = self._check_fuzz_value_range_valid(
                    desc_obj, 'value_range', desc_type, fuzz_dict)
                one_desc = {'format': format_value, 'type': type_value,
                            'shape': shape_value,
                            'value_range': value_range,
                            'data_distribute': data_distribute}
                # check whether has value.
                desc_obj = self._replace_fuzz_param(
                    desc_obj, utils.VALUE, desc_type, fuzz_dict)
                self._deal_with_value(desc_obj, one_desc)
            else:
                one_desc = {'format': format_value, 'type': type_value,
                            'shape': shape_value}
            # check whether has ori_format and ori_shape.
            one_desc = self._deal_with_ori_filed_data_fuzz(
                desc_obj, desc_type, fuzz_dict, one_desc)
            # check whether the shape is dynamic.
            if desc_obj.get(utils.TYPICAL_SHAPE) is not None:
                one_desc = self._deal_with_dynamic_shape_fuzz(
                    desc_obj, desc_type, fuzz_dict, one_desc)
            desc_list.append(one_desc)
        return desc_list

    def subcase_generate(self):
        """
        generate subcase by cross
        :return: the test case list
        """
        fuzz_function = self.check_fuzz_valid(self.json_obj)
        loop_num = self._check_fuzz_case_num_valid(self.json_obj)
        prefix = self.json_obj[CD.CASE_NAME].replace('/', '_') + '_fuzz_case_'
        pyfile, function = self._check_expect_output_param(self.json_obj)
        repeat_case_num = 0
        for _ in range(loop_num):
            ori_json = copy.deepcopy(self.json_obj)
            fuzz_dict = fuzz_function()
            if self.json_obj.get(CD.ST_MODE) == "ms_python_train":
                input_desc_list = self._make_desc_list_ms_fuzz(self.json_obj,
                                                               fuzz_dict,
                                                               CD.INPUT_DESC)
                output_desc_list = self._make_desc_list_ms_fuzz(self.json_obj,
                                                                fuzz_dict,
                                                                CD.OUTPUT_DESC)
            else:
                input_desc_list = self._make_desc_list_fuzz(self.json_obj,
                                                            fuzz_dict,
                                                            CD.INPUT_DESC)
                output_desc_list = self._make_desc_list_fuzz(self.json_obj,
                                                             fuzz_dict,
                                                             CD.OUTPUT_DESC)
            attr_list = self._check_attr_valid(self.json_obj, fuzz_dict)
            case = {CD.OP: self.json_obj[CD.OP],
                    CD.INPUT_DESC: input_desc_list,
                    CD.OUTPUT_DESC: output_desc_list,
                    'case_name': prefix + '%03d' % self.case_idx}
            if self.json_obj.get(CD.ST_MODE) == "ms_python_train":
                case[CD.ST_MODE] = "ms_python_train"
            if len(attr_list) > 0:
                case[CD.ATTR] = attr_list
            self.case_idx, self.total_case_list, repeat_case_num = \
                self._add_case_to_total_case(case, self.case_idx, pyfile,
                                             function, self.total_case_list,
                                             repeat_case_num)
            self.json_obj = ori_json
        if repeat_case_num > 0:
            utils.print_info_log(
                '%d fuzz test cases is repeated in %s, will '
                'not generate testcase.' % (repeat_case_num,
                                            self.json_obj[CD.CASE_NAME]))
        utils.print_info_log('Create %d fuzz test cases for %s.'
                             % (loop_num - repeat_case_num,
                                self.json_obj[CD.CASE_NAME]))
        return self.total_case_list
