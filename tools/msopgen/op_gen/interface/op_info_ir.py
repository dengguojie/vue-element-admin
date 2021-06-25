#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for IR operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import sys
import xlrd

from . import utils
from .op_info import OpInfo
from .arg_parser import ArgParser

IR_DEFAULT_SHEET_NAME = 'Op'
IR_TEMPLATE_HEADER = ['Op', 'Classify', 'Name', 'Type', 'TypeRange',
                      'Required', 'Doc', 'Attr_Default_value', 'Format']
IR_TEMPLATE_FIRST_ROW = 3
IR_TEMPLATE_VALID_NCLOS = 9
IR_TEMPLATE_OP_CLO = 0
IR_TEMPLATE_CLASSIFY_CLO = 1
IR_TEMPLATE_NAME_CLO = 2
IR_TEMPLATE_TYPE_CLO = 3
IR_TEMPLATE_TYPE_RANGE_CLO = 4
IR_TEMPLATE_REQUIRED_CLO = 5
IR_TEMPLATE_ATTR_DEFAULT_CLO = 7
IR_TEMPLATE_FORMAT_CLO = 8
INPUT_NAME = 'INPUT'
DYNAMIC_INPUT_NAME = 'DYNAMIC_INPUT'
OUTPUT_NAME = 'OUTPUT'
DYNAMIC_OUTPUT_NAME = 'DYNAMIC_OUTPUT'
ATTR_NAME = 'ATTR'
REQUIRED_ATTR_NAME = 'REQUIRED_ATTR'


class IrRow:
    """
    CLass for IR row.
    """

    def __init__(self, row):
        if len(row) >= IR_TEMPLATE_VALID_NCLOS:
            self.classify = row[IR_TEMPLATE_CLASSIFY_CLO]
            self.name = row[IR_TEMPLATE_NAME_CLO]
            self.op_type = row[IR_TEMPLATE_TYPE_CLO]
            self.type_range = row[IR_TEMPLATE_TYPE_RANGE_CLO]
            self.required = row[IR_TEMPLATE_REQUIRED_CLO]
            self.attr_default = row[IR_TEMPLATE_ATTR_DEFAULT_CLO]
            self.op_format = row[IR_TEMPLATE_FORMAT_CLO]
        else:
            utils.print_error_log("The row information is not enough.")
            sys.exit(utils.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)

    def get_name(self):
        """
        get name
        """
        return self.name

    def get_op_type(self):
        """
        get op type
        """
        return self.op_type


class IROpInfo(OpInfo):
    """
    CLass for IR OP Info.
    """

    def __init__(self, argument: ArgParser):
        super().__init__()
        self.op_path = argument.input_path
        self.gen_flag = argument.gen_flag
        self.output_path = argument.output_path
        if self.gen_flag:
            self.choose_op = argument.op_type
        else:
            self.mi_cmd = argument.mi_cmd

    def parse(self):
        """
        Parse the IR excel, store the parse result in OpInfo attribute
        """
        if self.gen_flag:
            self._parse_xls_to_info()
            self._check_input_output_info()
        else:
            if self.mi_cmd == utils.INPUT_ARGUMENT_CMD_MI_QUERY:
                self._parse_xls_to_json()

    def _parse_xls_to_info(self):
        utils.print_info_log("Start to parse the ir template:%s" %
                             self.op_path)
        sheet_data = self._get_sheet_data(self.op_path)
        ir_map = self._get_ir_map(sheet_data)
        op_names = list(ir_map.keys())
        op_name = self._choose_op(op_names)
        if not op_name:
            utils.print_error_log("Failed to get op type.")
            sys.exit(utils.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)
        ir_info = ir_map.get(op_name)
        if not ir_info:
            utils.print_error_log("Failed to get op info for '%s'. Please "
                                  "check the excel." % op_name)
            sys.exit(utils.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)
        self.op_type = op_name
        self.fix_op_type = utils.fix_name_lower_with_under(op_name)
        for ir_row in ir_info:
            classify = ir_row.classify.upper()
            if classify == INPUT_NAME:
                param_type = self._mapping_ini_param_type(ir_row.required)
                input_map = self._add_input_output('input', ir_row, param_type)
                if input_map is None:
                    utils.print_error_log("The attr types in the .xlsx file "
                                          "are unsupported. Please check the "
                                          "input or output types.")
                    raise utils.MsOpGenException(
                        utils.MS_OP_GEN_INVALID_PARAM_ERROR)
                self.parsed_input_info.update(input_map)
            elif classify == DYNAMIC_INPUT_NAME:
                param_type = utils.PARAM_TYPE_DYNAMIC
                input_map = self._add_input_output('input', ir_row, param_type)
                self.parsed_input_info.update(input_map)
            elif classify == OUTPUT_NAME:
                param_type = self._mapping_ini_param_type(ir_row.required)
                output_map = self._add_input_output('output', ir_row,
                                                    param_type)
                self.parsed_output_info.update(output_map)
            elif classify == DYNAMIC_OUTPUT_NAME:
                param_type = utils.PARAM_TYPE_DYNAMIC
                output_map = self._add_input_output('output', ir_row,
                                                    param_type)
                self.parsed_output_info.update(output_map)
            elif classify == ATTR_NAME:
                self._add_attr(ir_row.name, ir_row.op_type,
                               ir_row.attr_default)
            elif classify == REQUIRED_ATTR_NAME:
                self._add_attr(ir_row.name, ir_row.op_type,
                               ir_row.attr_default)
            else:
                utils.print_warn_log("Classify value is invalid: " + classify)

    def _parse_xls_to_json(self):
        sheet_data = self._get_sheet_data(self.op_path)
        ir_map = self._get_ir_map(sheet_data)
        op_names = list(ir_map.keys())
        json_data = {}
        json_data.setdefault(IR_DEFAULT_SHEET_NAME, [])
        for op_name in op_names:
            json_data[IR_DEFAULT_SHEET_NAME].append({"OP": op_name})
        _, ir_file_name = os.path.split(self.op_path)
        json_path = os.path.join(self.output_path, ir_file_name + ".json")
        utils.write_json_file(json_path, json_data)

    def _get_ir_map(self, sheet_data):
        if sheet_data is None:
            utils.print_error_log("Get sheet data fails.")
            sys.exit(utils.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)
        if not self._check_sheet_data(sheet_data):
            utils.print_error_log("Check sheet format failed.")
            sys.exit(utils.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)
        ir_map = self._read_sheet_data(sheet_data)
        return ir_map

    def _get_sheet_data(self, ir_file):
        ir_template, sheet_names = self._get_sheets(ir_file)
        if IR_DEFAULT_SHEET_NAME not in sheet_names:
            utils.print_error_log(" There is no sheet named \"Op\" in IR "
                                  "Excel. Please check!")
            raise utils.MsOpGenException(
                utils.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)
        sheet_data = ir_template.sheet_by_name(IR_DEFAULT_SHEET_NAME)
        return sheet_data

    @staticmethod
    def _get_sheets(ir_file):
        try:
            ir_template = xlrd.open_workbook(ir_file)
            sheet_names = ir_template.sheet_names()
        except OSError as err:
            utils.print_error_log("Load excel failed ,%s " % str(err.args[0]))
            raise utils.MsOpGenException(utils.MS_OP_GEN_READ_FILE_ERROR)
        return ir_template, sheet_names

    @staticmethod
    def _check_sheet_data(sheet_data):
        # The second row is header, a valid sheet has at least 2 rows
        if sheet_data.nrows < IR_TEMPLATE_FIRST_ROW:
            utils.print_warn_log(
                "The sheet '%s' data is invalid, at least %s rows "
                "needed" % (sheet_data.name, IR_TEMPLATE_FIRST_ROW))
            return False
        if sheet_data.nrows == IR_TEMPLATE_FIRST_ROW:
            utils.print_warn_log(
                "There is no content in the sheet '%s'" % sheet_data.name)
            return False
        headers = sheet_data.row_values(1)
        if len(headers) < IR_TEMPLATE_VALID_NCLOS:
            utils.print_warn_log("The '%s' number of headers is invalid: '%s'"
                                 % (sheet_data.name, str(len(headers))))
            return False
        for i in range(0, IR_TEMPLATE_VALID_NCLOS):
            header = str(headers[i]).strip()
            if header != IR_TEMPLATE_HEADER[i]:
                utils.print_warn_log(
                    "The header of the sheet '%s' is invalid: %s,"
                    "should be like %s " % (
                        sheet_data.name, header, IR_TEMPLATE_HEADER))
                return False
        return True

    @staticmethod
    def _del_with_merged_cell(sheet_data, col_span):
        if sheet_data.merged_cells:
            for item in sheet_data.merged_cells:
                # item contain the scope of the merged cell, e.g. [0, 3, 0, 3]
                for row in range(item[0], item[1]):
                    for col in range(item[2], item[3]):
                        # only the first block has value, file the value of
                        # first block in other blocks
                        if (row, col) != (item[0], item[2]):
                            col_span.update({(row, col): (item[0], item[2])})

    def _read_sheet_data(self, sheet_data):
        rows = sheet_data.nrows
        # Get merged cell in the sheet
        col_span = {}
        self._del_with_merged_cell(sheet_data, col_span)
        ir_map = {}
        for i in range(IR_TEMPLATE_FIRST_ROW, rows):
            row = []
            for j in range(IR_TEMPLATE_VALID_NCLOS):
                # if the block is merged block, fetch value from col span
                if col_span.get((i, j)):
                    op_value = str(
                        sheet_data.cell_value(*col_span.get((i, j)))).strip()
                else:
                    op_value = str(sheet_data.cell_value(i, j)).strip()
                if j == 0:
                    # if the op name is invalid , skip the row
                    if utils.check_name_valid(
                            op_value) == utils.MS_OP_GEN_NONE_ERROR:
                        row.append(op_value)
                    else:
                        break
                else:
                    row.append(op_value)
            if row and len(row[0]) > 0:
                ir_row = IrRow(row)
                if row[0] in ir_map:
                    ir_map[row[0]].append(ir_row)
                else:
                    ir_row_list = [ir_row]
                    ir_map[row[0]] = ir_row_list
        return ir_map

    def _choose_op(self, op_names):
        if self.choose_op != "":
            utils.print_info_log("Start to parse '%s' in the ir template."
                                 % self.choose_op)
            if self.choose_op not in op_names:
                utils.print_error_log(
                    "Failed to find '%s' in the excel. Please check that the "
                    "value for '-op' is valid."
                    % self.choose_op)
                sys.exit(utils.MS_OP_GEN_INVALID_PARAM_ERROR)
            return self.choose_op
        if len(op_names) > 1:
            utils.print_info_log("There is more than one operator in sheet:")
            i = 1
            for op_name in op_names:
                print(i, op_name)
                i += 1
            while True:
                op_number = input('Input the number of the op:')
                if op_number.isdigit():
                    op_number = int(op_number)
                    if op_number < 1 or op_number > len(op_names):
                        utils.print_warn_log(
                            "The input is out of range, please retype!")
                    else:
                        op_name = op_names[op_number - 1]
                        utils.print_info_log("You have chosen: " + op_name)
                        return op_name
                else:
                    utils.print_warn_log(
                        "The input is not a number, please retype!")
        elif len(op_names) == 0:
            utils.print_error_log("There is no op info to read.")
            return None
        else:
            utils.print_info_log("Start to parse the op: " + op_names[0])
            return op_names[0]

    def _add_input_output(self, prefix, ir_row, param_type):
        ir_name = ir_row.name.strip()
        ir_type = ir_row.type_range.strip()
        types = ir_type.split(",")
        # init ir type from TypeRange of IR excel
        ir_type_list = []
        for type_ir in types:
            converted_type = self._mapping_input_output_type(type_ir.strip(), ir_name)
            if converted_type:
                ir_type_list += converted_type.split(",")
        if not ir_type_list:
            utils.print_warn_log("The %s ir type is invalid: %s" % (
                prefix, types))
            return None
        # init  format from Format of IR excel
        op_format = []
        if ir_row.op_format:
            op_format += ir_row.op_format.split(",")
        else:
            op_format = ",".join("ND" for _ in ir_type_list)
            op_format = op_format.split(",")
        utils.print_info_log("One %s is handled: %s" % (prefix, ir_name))
        return {ir_name: {
            utils.INFO_IR_TYPES_KEY: ir_type_list,
            utils.INFO_PARAM_TYPE_KEY: param_type,
            utils.INFO_PARAM_FORMAT_KEY: op_format}}

    def _check_input_output_info(self):
        if not self.parsed_input_info:
            utils.print_warn_log("There is no input in ir excel. Please "
                                 "check the input or output type. If you "
                                 "aren't having problems, just ignore the "
                                 "warning.")
            return
        if not self.parsed_output_info:
            utils.print_warn_log("There is no output in ir excel. Please "
                                 "check the input or output type. If you "
                                 "aren't having problems, just ignore the "
                                 "warning.")
            return
        # check input ir type and format
        first_count = 0
        first_name = ""
        io_map = self.parsed_input_info.copy()
        io_map.update(self.parsed_output_info)
        for (name, value) in io_map.items():
            ir_type_count = len(value[utils.INFO_IR_TYPES_KEY])
            format_count = len(value[utils.INFO_PARAM_FORMAT_KEY])
            if first_count == 0:
                first_count = ir_type_count
                first_name = name
            else:
                if ir_type_count != first_count:
                    utils.print_warn_log("The number(%d) of %s type range is "
                                         "different from that(%d) of %s. "
                                         "Please check the input numbers in "
                                         "'TypeRange'."
                                         % (ir_type_count, name,
                                            first_count, first_name))
                if format_count != first_count:
                    utils.print_warn_log("The number(%d) of %s format is "
                                         "different from that(%d) of %s. "
                                         "Please check the input numbers in "
                                         "'Format'." % (format_count, name,
                                                        first_count,
                                                        first_name))

    def _add_attr(self, name, op_type, default_value):
        name = name.strip()
        op_type = op_type.strip()
        attr_type = self._mapping_attr_type(op_type)
        if not attr_type:
            utils.print_warn_log("Attr op_type is invalid: %s " % op_type)
            return
        self.parsed_attr_info.append([name, attr_type, self._parse_bool_value(
            default_value)])
        utils.print_info_log("One attr is handled: " + name)

    @staticmethod
    def _parse_bool_value(value):
        new_value = value.strip().lower()
        if new_value == 'true':
            return 'true'
        if new_value == 'false':
            return "false"
        return value

    @staticmethod
    def _mapping_ini_param_type(param_type):
        if param_type in utils.PARAM_TYPE_MAP_INI:
            # the return wont't be none
            return utils.PARAM_TYPE_MAP_INI.get(param_type)
        utils.print_error_log("The '%s' config in ir excel is "
                              "invalid." % param_type)
        sys.exit(utils.MS_OP_GEN_PARSER_EXCEL_FILE_ERROR)

    @staticmethod
    def _mapping_input_output_type(ir_type, ir_name):
        file_type = utils.INPUT_FILE_XLSX
        return utils.CheckFromConfig().trans_io_dtype(ir_type, ir_name,
                                                      file_type)

    @staticmethod
    def _mapping_attr_type(attr_type):
        file_type = utils.INPUT_FILE_XLSX
        return utils.CheckFromConfig().trans_ir_attr_type(attr_type, file_type)

    def get_op_path(self):
        """
        get op path
        """
        return self.op_path
