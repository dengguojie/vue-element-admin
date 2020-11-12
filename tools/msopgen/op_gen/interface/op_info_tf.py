#!/usr/bin/python3
# coding=utf-8

"""
Function:
This file mainly involves class for IR operator info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import re
    import sys
    from . import utils
    from .op_info import OpInfo
except (ImportError,) as import_error:
    sys.exit("[ERROR][op_info_tf]Unable to import module: %s." % str(
        import_error))

TF_INPUT_OUTPUT_DTYPE_MAP = {
    "float": "DT_FLOAT",
    "DT_FLOAT": "DT_FLOAT",
    "half": "DT_FLOAT16",
    "DT_HALF": "DT_FLOAT16",
    "double": "DT_DOUBLE",
    "DT_DOUBLE": "DT_DOUBLE",
    "int8": "DT_INT8",
    "DT_INT8": "DT_INT8",
    "int16": "DT_INT16",
    "DT_INT16": "DT_INT16",
    "int32": "DT_INT32",
    "DT_INT32": "DT_INT32",
    "int64": "DT_INT64",
    "DT_INT64": "DT_INT64",
    "uint8": "DT_UINT8",
    "DT_UINT8": "DT_UINT8",
    "uint16": "DT_UINT16",
    "DT_UINT16": "DT_UINT16",
    "uint32": "DT_UINT32",
    "DT_UINT32": "DT_UINT32",
    "uint64": "DT_UINT64",
    "DT_UINT64": "DT_UINT64",
    "qint8": "DT_QINT8",
    "DT_QINT8": "DT_QINT8",
    "qint16": "DT_QINT16",
    "DT_QINT16": "DT_QINT16",
    "qint32": "DT_QINT32",
    "DT_QINT32": "DT_QINT32",
    "quint8": "DT_QUINT8",
    "DT_QUINT8": "DT_QUINT8",
    "quint16": "DT_QUINT16",
    "DT_QUINT16": "DT_QUINT16",
    "complex64": "DT_COMPLEX64",
    "DT_COMPLEX64": "DT_COMPLEX64",
    "complex128": "DT_COMPLEX128",
    "DT_COMPLEX128": "DT_COMPLEX128",
    "bool": "DT_BOOL",
    "DT_BOOL": "DT_BOOL",
    "string": "DT_STRING",
    "DT_STRING": "DT_STRING",
    "resource": "DT_RESOURCE",
    "DT_RESOURCE": "DT_RESOURCE",
    "numbertype": " DT_FLOAT, DT_DOUBLE,DT_INT64, DT_INT32, DT_UINT8, "
                  "DT_UINT16, DT_INT16, DT_INT8, DT_COMPLEX64,DT_COMPLEX128,"
                  "DT_QINT8,DT_QINT32, DT_FLOAT16, DT_UINT32,DT_UINT64",
    "realnumbertype": "DT_FLOAT, DT_DOUBLE,  DT_INT32, DT_INT64, DT_UINT8, "
                      "DT_INT16, DT_INT8, DT_UINT16, DT_FLOAT16, DT_UINT32, "
                      "DT_UINT64",
    "quantizedtype": "DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32"
}


class TFOpInfo(OpInfo):
    """
    CLass representing operator info.
    """

    def __init__(self, op_path):
        super().__init__()
        self.op_path = op_path
        self.attr_info = []
        self.input_info = []
        self.output_info = []
        self.type_attr = {}
        self.op_type = None

    @staticmethod
    def _parse_name_info(line):
        if ":" not in line:
            utils.print_warn_log(
                " The name info \"" + line + "\" error ,can not find \":\".")
            return '', ''
        return line.split(":", 1)

    def parse(self):
        """
        Function Description:
            parse tensorflow operator
        Parameter:
            tf_file: tensorflow operator file
        Return Value:
        """
        utils.print_info_log("Start to parse the tensorflow ir: %s" %
                             self.op_path)
        txt = utils.read_file(self.op_path)
        input_info_lines = []
        output_info_lines = []
        attr_info_lines = []
        op_name = ""
        new_line = txt.replace('\n', utils.EMPTY).replace('\r', utils.EMPTY) \
            .replace('\t', utils.EMPTY)
        pattern = re.compile(utils.SPACE)
        line = pattern.sub(utils.EMPTY, new_line)
        line = line.replace('\"\"', utils.EMPTY).replace('\\', utils.EMPTY)
        line_point_list = line.split(".")
        for info_str in line_point_list:
            if info_str is None or len(info_str) == 0:
                continue
            if "REGISTER_OP" in info_str:
                match_list = utils.get_content_from_double_quotes(info_str)
                if match_list:
                    utils.print_info_log(
                        "The op type is %s." % str(match_list[0]))
                    op_name = match_list[0]
                else:
                    op_name = ""
                continue
            if info_str.startswith("Input") or info_str.startswith("Output") \
                    or info_str.startswith("Attr"):
                match_list = utils.get_content_from_double_quotes(info_str)
                if not match_list:
                    utils.print_warn_log("Parsing by (\"key:value\") error, "
                                         "continue.")
                    continue
                if info_str.startswith("Input"):
                    input_info_lines.append(match_list[0])
                elif info_str.startswith("Output"):
                    output_info_lines.append(match_list[0])
                elif info_str.startswith("Attr"):
                    attr_info_lines.append(match_list[0])
                else:
                    continue
        self._init_op_info(op_name, input_info_lines, output_info_lines,
                           attr_info_lines)

    def _init_op_info(self, op_name, input_info_lines, output_info_lines,
                      attr_info_lines):
        if not op_name:
            utils.print_warn_log(
                "Cannot parse the op type, please check the op type.")
        if not input_info_lines and not output_info_lines:
            utils.print_warn_log(
                "There is no input and output information, please check the "
                "input and output.")
        self.op_type = op_name
        self.fix_op_type = utils.fix_name_lower_with_under(op_name)
        for input_line in input_info_lines:
            utils.print_info_log("One input line is handled: %s" % input_line)
            name, info = self._parse_name_info(input_line)
            self._add_input(name, info)
        for output_line in output_info_lines:
            utils.print_info_log("One output line is handled: %s" %
                                 output_line)
            name, info = self._parse_name_info(output_line)
            self._add_output(name, info)
        for attr_line in attr_info_lines:
            utils.print_info_log("One attribute line is handled: %s" %
                                 attr_line)
            name, info = self._parse_name_info(attr_line)
            self._add_attr(name, info)

        self._generate_input_info()
        self._generate_output_info()
        self._generate_attr_info()

    def _add_input(self, op_name, op_info):
        op_name = utils.fix_name_lower_with_under(op_name.strip())
        op_info = op_info.strip()
        self.input_info.append([op_name, op_info])
        # for dynamic type(eg:N*T), the key store in type_attr should to
        # remove the "N*", only "T"
        dynamic_type = self._get_dynamic_input_output_type(op_info)
        if dynamic_type:
            self.type_attr[dynamic_type] = 0
        else:
            self.type_attr[op_info] = 0

    def _add_output(self, op_name, op_info):
        op_name = utils.fix_name_lower_with_under(op_name.strip())
        op_info = op_info.strip()
        self.output_info.append([op_name, op_info])
        # for dynamic type(eg:N*T), the key store in type_attr should to
        # remove the "N*", only "T"
        dynamic_type = self._get_dynamic_input_output_type(op_info)
        if dynamic_type:
            self.type_attr[dynamic_type] = 0
        else:
            self.type_attr[op_info] = 0

    def _add_attr(self, op_name, op_info):
        op_name = op_name.strip()
        op_info = op_info.strip()
        if op_name in self.type_attr:
            self.type_attr[op_name] = op_info
        else:
            self.attr_info.append([op_name, op_info])

    def _generate_type_info(self, types, name):
        attr_info = {}
        if types.startswith("{"):
            if "}" not in types:
                utils.print_error_log(
                    "The attr type '%s' error ,can not find '}'." % types)
                return ""
            type_info = types[1:types.index("}")]
            types = type_info.split(",")
            attr_info["types"] = [
                self._mapping_input_output_type(t.strip(), name) for t in
                types]
            if "=" in types:
                default_type = types[types.index("="):]
                attr_info["default_type"] = default_type
            attr_info["types"] = [x for x in attr_info.get("types") if x != ""]
            return ",".join(attr_info.get("types"))
        return self._mapping_input_output_type(types.strip(), name)

    @staticmethod
    def _get_dynamic_input_output_type(value):
        if "N*" in value:
            return value.replace("N*", "")
        return ""

    def _generate_input_info(self):
        for name, value in self.input_info:
            # parse dynamic input/output
            dynamic_type = self._get_dynamic_input_output_type(value)
            if dynamic_type:
                param_type = utils.PARAM_TYPE_DYNAMIC
                ir_type = dynamic_type
            else:
                param_type = utils.PARAM_TYPE_REQUIRED
                ir_type = value
            # mapping ir type list
            if self.type_attr.get(ir_type) != 0:
                ir_types = self._generate_type_info(
                    self.type_attr.get(ir_type),
                    name)
            else:
                ir_types = self._mapping_input_output_type(ir_type, name)
            ir_type_list = ir_types.split(',')
            # update op_info.parsed_input_info
            self.parsed_input_info.update({name: {
                utils.INFO_IR_TYPES_KEY: ir_type_list,
                utils.INFO_PARAM_TYPE_KEY: param_type}})

    def _generate_output_info(self):
        for name, value in self.output_info:
            # parse dynamic input/output
            dynamic_type = self._get_dynamic_input_output_type(value)
            if dynamic_type:
                param_type = utils.PARAM_TYPE_DYNAMIC
                ir_type = dynamic_type
            else:
                param_type = utils.PARAM_TYPE_REQUIRED
                ir_type = value
            # mapping ir type list
            if self.type_attr.get(ir_type) != 0:
                ir_types = self._generate_type_info(
                    self.type_attr.get(ir_type),
                    name)
            else:
                ir_types = self._mapping_input_output_type(ir_type, name)
            ir_type_list = ir_types.split(',')
            # update op_info.parsed_input_info
            self.parsed_output_info.update({name: {
                utils.INFO_IR_TYPES_KEY: ir_type_list,
                utils.INFO_PARAM_TYPE_KEY: param_type}})

    def _generate_attr_info(self):
        for name, value in self.attr_info:
            if self._check_dynamic_io_attr_info(name, value):
                utils.print_info_log("The attr '%s:%s' is belong to dynamic "
                                     "input/output, do not parse."
                                     % (name, value))
                return
            attr_name = utils.fix_name_lower_with_under(name)
            if "=" in value:
                attr_splits = value.split("=")
                attr_type = self._mapping_attr_type(attr_splits[0].strip())
                default_value = attr_splits[1].strip()
                self.parsed_attr_info.append(
                    [attr_name, attr_type, default_value])
            else:
                attr_type = self._mapping_attr_type(value.strip())
                self.parsed_attr_info.append([attr_name, attr_type])

    @staticmethod
    def _check_dynamic_io_attr_info(name, value):
        if name is 'N' and ('>' or '<' in value):
            return True
        else:
            return False

    @staticmethod
    def _mapping_attr_type(tf_type):
        if tf_type in utils.TF_ATTR_TYPE_MAP:
            return utils.TF_ATTR_TYPE_MAP.get(tf_type)
        utils.print_warn_log("The attr type '%s'  in "
                             "the .txt file is unsupported. Please check "
                             "the input or output type. If you aren't "
                             "having problems, just ignore the warning."
                             % tf_type)
        return ""

    @staticmethod
    def _mapping_input_output_type(tf_type, name):
        # mapping from tf type to D enum
        if tf_type in TF_INPUT_OUTPUT_DTYPE_MAP:
            return TF_INPUT_OUTPUT_DTYPE_MAP.get(tf_type)
        utils.print_warn_log("The '%s' type '%s' in "
                             "the .txt file is unsupported. Please "
                             "check. If you aren't having problems, "
                             "just ignore the warning." % (name, tf_type))
        return ""
