#!/usr/bin/env python
# coding=utf-8

"""
Function:
This file mainly involves class for IR JSON operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import sys

from . import utils
from .op_info import OpInfo
from .arg_parser import ArgParser
from .const_manager import ConstManager


class JsonIROpInfo(OpInfo):
    """
    CLass for IR OP Info from Json.
    """

    def __init__(self: any, argument: ArgParser) -> None:
        super().__init__()
        self.op_path = argument.input_path
        self.gen_flag = argument.gen_flag
        self.output_path = argument.output_path
        if self.gen_flag:
            self.choose_op = argument.op_type
        else:
            self.mi_cmd = argument.mi_cmd

    def parse(self: any) -> None:
        """
        Parse the IR json, store the parse result in OpInfo attribute
        """
        if self.gen_flag:
            self._parse_json_to_info()
            self._check_input_output_info()
        else:
            if self.mi_cmd == ConstManager.INPUT_ARGUMENT_CMD_MI_QUERY:
                self._parse_template_to_json()

    def _parse_json_to_info(self: any) -> None:
        utils.print_info_log("Start to parse the ir template:%s" %
                             self.op_path)
        json_data = utils.read_json_file(self.op_path)
        self._check_json_data(json_data)
        if isinstance(json_data, dict):
            json_data = [json_data]
        ir_map = self._read_json_data(json_data)
        ir_info = self._choose_op_for_generate(ir_map)
        input_list = ir_info.get("input_list")
        output_list = ir_info.get("output_list")
        attr_list = ir_info.get("attr_list")
        if input_list is not None:
            self._add_input_output_from_json("input_desc", input_list)
        else:
            utils.print_warn_log("The \"input_desc\" value is invalid or "
                                 "no \"input_desc\" exists in the map.")
        if output_list is not None:
            self._add_input_output_from_json("output_desc", output_list)
        else:
            utils.print_warn_log("The \"output_desc\" value is invalid or "
                                 "no \"output_desc\" exists in the map.")
        if attr_list is not None:
            self._add_attr_from_json(attr_list)
        else:
            utils.print_warn_log("The \"attr\" value is invalid or no \"attr\" "
                                 "exists in the map.")

    def _check_input_output_info(self: any) -> None:
        if not self.parsed_input_info:
            utils.print_warn_log("There is no input in the IR json. Please "
                                 "check the input or output type. If you "
                                 "do not have this problem, ignore the "
                                 "warning.")
            return
        if not self.parsed_output_info:
            utils.print_warn_log("There is no output in the IR json. Please "
                                 "check the input or output type. If you "
                                 "aren't having problems, ignore the "
                                 "warning.")
            return
        # check input ir type and format
        first_count = 0
        first_name = ""
        io_map = self.parsed_input_info.copy()
        io_map.update(self.parsed_output_info)
        for (name, value) in io_map.items():
            ir_type_count = len(value[ConstManager.INFO_IR_TYPES_KEY])
            format_count = len(value[ConstManager.INFO_PARAM_FORMAT_KEY])
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

    def _parse_template_to_json(self: any) -> None:
        json_data = utils.read_json_file(self.op_path)
        self._check_json_data(json_data)
        if isinstance(json_data, dict):
            json_data = [json_data]
        ir_map = self._read_json_data(json_data)
        op_names = list(ir_map.keys())
        json_data = {}
        json_data.setdefault("Op", [])
        for op_name in op_names:
            json_data["Op"].append({"OP": op_name})
        _, ir_file_name = os.path.split(self.op_path)
        json_path = os.path.join(self.output_path, ir_file_name + ".json")
        utils.write_json_file(json_path, json_data)

    def _check_json_data(self: any, json_data: any) -> None:
        if not isinstance(json_data, (list, dict)):
            utils.print_error_log("Data in %s should be List or Dict."
                                  % self.op_path)
            raise utils.MsOpGenException(ConstManager.MS_OP_GEN_JSON_DATA_ERROR)
        if isinstance(json_data, list) and len(json_data) < 1:
            utils.print_error_log("There is an operator definition map in %s. "
                                  "Please check." % self.op_path)
            raise utils.MsOpGenException(ConstManager.MS_OP_GEN_JSON_DATA_ERROR)

    def _read_json_data(self: any, json_data: any) -> dict:
        ir_map = {}
        for json_map in json_data:
            op_map = {}
            if json_map.get("op") is None:
                utils.print_warn_log("The map in %s does not have the \"op\" key."
                                     "Please check." % self.op_path)
            else:
                op_name = json_map.get("op")
                if utils.check_name_valid(
                        op_name) == ConstManager.MS_OP_GEN_NONE_ERROR:
                    op_map["input_list"] = json_map.get("input_desc")
                    op_map["output_list"] = json_map.get("output_desc")
                    op_map["attr_list"] = json_map.get("attr")
                    if op_name in ir_map.keys():
                        utils.print_warn_log("There are some maps with "
                                             "duplicate \"op\" : \"%s\" "
                                             "in %s. The last one is to be "
                                             "used." % (op_name, self.op_path))
                    ir_map[op_name] = op_map
        return ir_map

    def _choose_op_for_generate(self: any, ir_map: dict) -> any:
        op_names = list(ir_map.keys())
        op_name = self._choose_op(op_names)
        if not op_name:
            utils.print_error_log("Failed to obtain the op type.")
            sys.exit(ConstManager.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)
        ir_info = ir_map.get(op_name)
        if not ir_info:
            utils.print_error_log("Failed to obtain op info for '%s'. Please "
                                  "check the json." % op_name)
            sys.exit(ConstManager.MS_OP_GEN_INVALID_SHEET_PARSE_ERROR)
        self.op_type = op_name
        self.fix_op_type = utils.fix_name_lower_with_under(op_name)
        return ir_info

    def _choose_op(self: any, op_names: list) -> str:
        if self.choose_op != "":
            utils.print_info_log("Start to parse '%s' in the json ir template."
                                 % self.choose_op)
            if self.choose_op not in op_names:
                utils.print_error_log(
                    "Failed to find '%s' in json. Please check "
                    "that the value for '-op' is valid."
                    % self.choose_op)
                sys.exit(ConstManager.MS_OP_GEN_INVALID_PARAM_ERROR)
            return self.choose_op
        if len(op_names) > 1:
            utils.print_info_log("There is more than one operator in "
                                 "the .json file:")
            i = 1
            for op_name in op_names:
                print(i, op_name)
                i += 1
            while True:
                op_number = input('Input the number of the ops:')
                if op_number.isdigit():
                    op_number = int(op_number)
                    if op_number < 1 or op_number > len(op_names):
                        utils.print_warn_log(
                            "The input is out of range, please retype one.")
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

    def _add_input_output_from_json(self: any, prefix: str, input_output_list: any) -> None:
        if isinstance(input_output_list, list):
            for input_output_map in input_output_list:
                self._update_input_output_info(prefix, input_output_map)
        else:
            utils.print_warn_log("\"%s\" in the map should be list" % prefix)

    def _update_input_output_info(self: any, prefix: str, input_output_map: any) -> None:
        if isinstance(input_output_map, dict):
            input_output_name = input_output_map.get("name")
            if input_output_name is None or \
                    not isinstance(input_output_name, str):
                utils.print_warn_log("The input or output name is "
                                     "None or invalid. please check.")
                return
            input_output_name = input_output_name.strip()
            types = input_output_map.get("type")
            if types is None or not isinstance(types, (list, str)):
                utils.print_warn_log("The input or output type is "
                                     "None or invalid. please check.")
                return
            if isinstance(types, str):
                types = [types]
            ir_type_list = self._init_ir_type(prefix, input_output_name, types)
            op_format = self._init_op_format(input_output_map, prefix,
                                             input_output_name, ir_type_list)
            param_type = self._init_param_type(input_output_map,
                                               input_output_name)
            self._update_parsed_info(prefix, input_output_name, ir_type_list,
                                     [param_type, op_format])
        else:
            utils.print_warn_log(
                "Every value in the \"%s\" list should be dict" % prefix)

    def _init_ir_type(self: any, prefix: str, input_output_name: any, types: list) -> list:
        ir_type_list = []
        for ir_type in types:
            converted_type = self._mapping_input_output_type(
                ir_type.strip(), input_output_name)
            if converted_type:
                ir_type_list += converted_type.split(",")
        if not ir_type_list:
            utils.print_warn_log("The %s ir type is invalid: %s" %
                                 (prefix, types))
            utils.print_error_log("The input or output type in the json "
                                  "file is not supported. Please check the "
                                  "input or output type.")
            raise utils.MsOpGenException(
                ConstManager.MS_OP_GEN_INVALID_PARAM_ERROR)
        return ir_type_list

    @staticmethod
    def _mapping_input_output_type(ir_type: str, ir_name: str) -> any:
        file_type = ConstManager.INPUT_FILE_JSON
        return utils.CheckFromConfig().trans_io_dtype(ir_type, ir_name,
                                                      file_type)

    @staticmethod
    def _init_op_format(input_output_map: dict, prefix: str, input_output_name: str,
                        ir_type_list: list) -> any:
        op_format = input_output_map.get("format")
        if not isinstance(op_format, (list, str)):
            op_format = None
        if isinstance(op_format, str):
            op_format = [op_format]
        if op_format is None or len(op_format) == 0:
            utils.print_warn_log("The format value is None or invalid, which will "
                                 "be automatically filled with ND according to the "
                                 "number of types")
            op_format = ",".join("ND" for _ in ir_type_list)
            op_format = op_format.split(",")
        if len(op_format) != len(ir_type_list):
            utils.print_warn_log("The number of types does not match that of "
                                 "formats. please check.")
        utils.print_info_log("One %s is handled: %s" %
                             (prefix, input_output_name))
        return op_format

    @staticmethod
    def _init_param_type(input_output_map: dict, input_output_name: str) -> str:
        param_type = input_output_map.get("param_type")
        if param_type not in ConstManager.INPUT_OUTPUT_PARAM_TYPE:
            param_type = ConstManager.PARAM_TYPE_REQUIRED
            utils.print_warn_log("The param_type of %s is invalid or None, "
                                 "Assign it the default value \"required\"" %
                                 input_output_name)
        return param_type

    def _update_parsed_info(self: any, prefix: str, input_output_name: str, ir_type_list: list,
                            type_format: list) -> None:
        param_type = type_format[0]
        op_format = type_format[1]
        if prefix == "input_desc":
            if input_output_name in self.parsed_input_info:
                utils.print_warn_log("The input name \"%s\" is duplicate.  "
                                     "The last one is to be used!" %
                                     input_output_name)
            self.parsed_input_info.update({input_output_name: {
                ConstManager.INFO_IR_TYPES_KEY: ir_type_list,
                ConstManager.INFO_PARAM_TYPE_KEY: param_type,
                ConstManager.INFO_PARAM_FORMAT_KEY: op_format}})
        else:
            if input_output_name in self.parsed_output_info:
                utils.print_warn_log("The out name \"%s\" is duplicate.  The "
                                     "last one is to be used!" %
                                     input_output_name)
            self.parsed_output_info.update({input_output_name: {
                ConstManager.INFO_IR_TYPES_KEY: ir_type_list,
                ConstManager.INFO_PARAM_TYPE_KEY: param_type,
                ConstManager.INFO_PARAM_FORMAT_KEY: op_format}})

    def _add_attr_from_json(self: any, attr_list: any) -> None:
        if isinstance(attr_list, list):
            for attr_map in attr_list:
                self._update_attr_info(attr_map)
        else:
            utils.print_warn_log("Attr in the map should be a list.")

    def _update_attr_info(self: any, attr_map: dict) -> None:
        attr_name = attr_map.get("name")
        if attr_name is None or not isinstance(attr_name, str):
            utils.print_warn_log("The attr_name name is None or invalid. "
                                 "Please check!")
            return
        attr_name = attr_name.strip()
        op_type = attr_map.get("type")
        if op_type is None or not isinstance(op_type, str):
            utils.print_warn_log("The op_type name is None or invalid. "
                                 "Please check!")
            return
        op_type = op_type.strip()
        attr_type = self._mapping_attr_type(op_type)
        if not attr_type:
            utils.print_warn_log("Attr op_type is invalid: %s " % op_type)
            return
        default_value = attr_map.get("default_value")
        if isinstance(default_value, str):
            default_value = default_value.strip()
        if isinstance(default_value, bool):
            default_value = self._parse_bool_value_for_json(default_value)
        param_type = attr_map.get("param_type")
        if param_type not in ConstManager.ATTR_PARAM_TYPE:
            param_type = ConstManager.PARAM_TYPE_REQUIRED
            utils.print_warn_log("The param_type of %s is invalid or None. "
                                 "Assign it the default value \"required\"" %
                                 attr_name)
        self.parsed_attr_info.append([attr_name, attr_type, default_value,
                                      param_type])
        utils.print_info_log("One attr is handled: " + attr_name)

    @staticmethod
    def _mapping_attr_type(attr_type: str) -> any:
        file_type = ConstManager.INPUT_FILE_JSON
        return utils.CheckFromConfig().trans_ir_attr_type(attr_type, file_type)

    @staticmethod
    def _parse_bool_value_for_json(value: any) -> str:
        if value:
            return 'true'
        return "false"
