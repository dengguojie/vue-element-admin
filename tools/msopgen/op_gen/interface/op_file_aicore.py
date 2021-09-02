#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves class for generating aicore operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import os
from .op_file import OPFile
from .op_tmpl import OPTmpl
from . import utils


class OpFileAiCore(OPFile):
    """
    CLass for generate aicore op files
    """

    CFG_INFO_TYPE_MAP = {
        'DT_FLOAT': 'float',
        'DT_FLOAT16': 'float16',
        'DT_FLOAT32': 'float32',
        'DT_INT8': 'int8',
        'DT_INT16': 'int16',
        'DT_INT32': 'int32',
        'DT_INT64': 'int64',
        'DT_UINT8': 'uint8',
        'DT_UINT16': 'uint16',
        'DT_UINT32': 'uint32',
        'DT_UINT64': 'uint64',
        'DT_BOOL': 'bool',
        "DT_COMPLEX64": "complex64",
        "DT_COMPLEX128": "complex128",
        "DT_DOUBLE": "double"
    }

    def _generate_cmake_lists(self):
        tbe_dir = os.path.join(self.output_path, 'tbe')
        if os.path.exists(tbe_dir):
            return
        utils.make_dirs(tbe_dir)
        template_path = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            utils.OP_TEMPLATE_TBE_PATH)
        utils.copy_template(template_path, tbe_dir, True)

    def _generate_op_params_for_check(self):
        op_params_for_check = ""
        for op_input in self.op_info.parsed_input_info.values():
            input_type = op_input.get("param_type").upper()
            input_for_check = "para_check.%s_INPUT, " % input_type
            op_params_for_check = "{}{}".format(op_params_for_check, input_for_check)
        for op_output in self.op_info.parsed_output_info.values():
            output_type = op_output.get("param_type").upper()
            output_for_check = "para_check.%s_OUTPUT, " % output_type
            op_params_for_check = "{}{}".format(op_params_for_check, output_for_check)
        for op_attr in self.op_info.parsed_attr_info:
            attr_type = op_attr[1].strip()
            check_from_config = utils.CheckFromConfig()
            attr_type = check_from_config.trans_check_attr_type(attr_type)
            if utils.PARAM_TYPE_OPTIONAL in op_attr:
                opt_attr_for_check = "para_check.OPTION_ATTR_%s, " % attr_type
                op_params_for_check = "{}{}".format(op_params_for_check, opt_attr_for_check)
            else:
                required_attr_for_check = "para_check.REQUIRED_ATTR_%s, " % attr_type
                op_params_for_check = "{}{}".format(op_params_for_check, required_attr_for_check)
        op_params_for_check += "para_check.KERNEL_NAME"
        return op_params_for_check

    def generate_impl(self):
        """
        Function Description:
        generate operator implementation.
        Parameter:
        Return Value:
        """
        self._generate_cmake_lists()
        op_params_for_check = self._generate_op_params_for_check()
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty. Failed to generate "
                                 "impl files. Please check.")
            return
        # 1.make head string
        head_str = OPTmpl.PY_HEAD
        # 2.make [op_type]_compute()
        op_input = ", ".join(list(self.op_info.parsed_input_info))
        op_output = ", ".join(list(self.op_info.parsed_output_info))
        if len(self.op_info.parsed_attr_info) == 0:
            head_str += OPTmpl.PY_COMPUTE_WITHOUT_ATTR.format(
                name=self.op_info.fix_op_type,
                input_name=op_input,
                output=op_output)
        else:
            attr = ", ".join(a[0] for a in self.op_info.parsed_attr_info)
            head_str += OPTmpl.PY_COMPUTE_WITH_ATTR.format(
                name=self.op_info.fix_op_type,
                input_name=op_input,
                output=op_output,
                attr=attr)
        head_str += OPTmpl.PY_COMPUTE_END.format(input_name=op_input)
        # 3.make [op_type]()
        if len(self.op_info.parsed_attr_info) == 0:
            head_str += OPTmpl.PY_DEF_WITHOUT_ATTR.format(
                op_params=op_params_for_check,
                name=self.op_info.fix_op_type,
                input_name=op_input,
                output=op_output)
        else:
            attr = ", ".join(a[0] for a in self.op_info.parsed_attr_info)
            head_str += OPTmpl.PY_DEF_WITH_ATTR.format(
                op_params=op_params_for_check,
                name=self.op_info.fix_op_type,
                input_name=op_input,
                output=op_output,
                attr=attr)
        for name in self.op_info.parsed_input_info:
            head_str += OPTmpl.PY_PLACEHOLDER.format(name=name)
        input_data = ", ".join("data_" + x
                               for x in self.op_info.parsed_input_info)
        output_data = ", ".join(y for y in self.op_info.parsed_output_info)
        if len(self.op_info.parsed_attr_info) == 0:
            head_str += OPTmpl.PY_RES_WITHOUT_ATTR.format(
                name=self.op_info.fix_op_type,
                input_data=input_data,
                output_data=output_data)
        else:
            attr = ", ".join(a[0] for a in
                             self.op_info.parsed_attr_info)
            head_str += OPTmpl.PY_RES_WIT_ATTR.format(
                name=self.op_info.fix_op_type,
                input_data=input_data,
                output_data=output_data,
                attr=attr)
        head_str += OPTmpl.PY_TARGET_CCE
        head_str += OPTmpl.PY_BUILD.format(input_data=input_data,
                                            left_braces=utils.LEFT_BRACES,
                                            right_braces=utils.RIGHT_BRACES)
        # create py_dir
        py_dir = os.path.join(self.output_path, utils.IMPL_DIR)
        py_path = os.path.join(py_dir, self.op_info.fix_op_type +
                               utils.IMPL_SUFFIX)
        utils.make_dirs(py_dir)
        utils.write_files(py_path, head_str)

    def generate_info_cfg(self):
        """
        Function Description:
        generate operator info config file
        Parameter:
        Return Value:
        """
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty. Failed to generate "
                                 "the info config file. Please check.")
            return
        # 1.make [OpType], eg:[Add]
        new_str = OPTmpl.INI_OP.format(op_type=self.op_info.op_type)
        # 2.make input string
        new_str += self._generate_input_output_info_cfg(
            self.op_info.parsed_input_info, OPTmpl.INI_INPUT)
        # 3.make output string
        new_str += self._generate_input_output_info_cfg(
            self.op_info.parsed_output_info, OPTmpl.INI_OUTPUT)
        # 4.make attr string
        if len(self.op_info.parsed_attr_info) > 0:
            attr_info = ", ".join(x[0] for x in self.op_info.parsed_attr_info)
            new_str += OPTmpl.INI_ATTR_LIST.format(attr_info=attr_info)
            for attr in self.op_info.parsed_attr_info:
                new_str = self._generate_attr_aicore(attr, new_str)

        # 5.make bin file string
        new_str += OPTmpl.INI_BIN_FILE.format(name=self.op_info.fix_op_type)
        self._make_info_cfg_file(new_str)

    def _generate_attr_aicore(self, attr, new_str):
        attr_type = self._mapping_attr_type_for_ini(attr[1])
        new_str += OPTmpl.INI_ATTR_TYPE_VALUE.format(name=attr[0],
                                                      type=attr_type)
        if len(attr) == 4:
            new_str += OPTmpl.INI_ATTR_PARAM_TYPE.format(
                name=attr[0],
                paramType=attr[3]
            )
            if attr[2]:
                new_str += OPTmpl.INI_ATTR_DEFAULT_VALUE.format(
                    name=attr[0],
                    defaultValue=attr[2]
                )
        elif len(attr) == 3 and attr[2] != "":
            new_str += OPTmpl.INI_ATTR_PARAM_TYPE.format(
                name=attr[0],
                paramType=utils.PARAM_TYPE_OPTIONAL
            )
            new_str += OPTmpl.INI_ATTR_DEFAULT_VALUE.format(
                name=attr[0],
                defaultValue=attr[2]
            )
        else:
            new_str += OPTmpl.INI_ATTR_PARAM_TYPE.format(
                name=attr[0],
                paramType=utils.PARAM_TYPE_REQUIRED
            )
        return new_str

    def _generate_input_output_info_cfg(self, parsed_info, template_string):
        new_str = ""
        for (index, name) in enumerate(parsed_info):
            ir_types = list(x for x in
                            parsed_info[name][utils.INFO_IR_TYPES_KEY] if
                            x != "")
            ini_types = list(self._mapping_info_cfg_type(x) for x in ir_types)
            ini_types = list(x for x in ini_types if x != "")
            ini_types = ",".join(ini_types)

            # pram_type, when generator from tf ir, default param is 'required'
            param_type = parsed_info[name][utils.INFO_PARAM_TYPE_KEY]
            # format, the default format is 'ND'
            if utils.INFO_PARAM_FORMAT_KEY in parsed_info[name]:
                op_format = ",".join(
                    parsed_info[name][utils.INFO_PARAM_FORMAT_KEY])
            else:
                op_format = ",".join("ND" for _ in ini_types.split(','))
            new_str += template_string.format(index=index,
                                              name=name,
                                              dtype=ini_types,
                                              format=op_format,
                                              paramType=param_type)
        return new_str

    def _make_info_cfg_file(self, new_str):
        for unit in self.compute_unit:
            compute_unit_parse_list = unit.split("-", 1)
            info_dir = os.path.join(self.output_path, 'tbe', 'op_info_cfg')
            utils.make_dirs(info_dir)
            core_type_dir = os.path.join(info_dir, compute_unit_parse_list[0])
            utils.make_dirs(core_type_dir)
            soc_dir = os.path.join(core_type_dir, compute_unit_parse_list[1])
            utils.make_dirs(soc_dir)
            # create dir and write ini file
            info_path = os.path.join(soc_dir, self.op_info.fix_op_type +
                                     ".ini")
            utils.write_files(info_path, new_str)

    @staticmethod
    def _mapping_attr_type_for_ini(attr_type):
        attr_type = attr_type.strip()
        return utils.CheckFromConfig().trans_ini_attr_type(attr_type)

    @staticmethod
    def _mapping_info_cfg_type(op_type):
        op_type = op_type.strip()
        if op_type in OpFileAiCore.CFG_INFO_TYPE_MAP:
            return OpFileAiCore.CFG_INFO_TYPE_MAP.get(op_type)
        utils.print_warn_log("The input/output type '%s' "
                             "is not supported by the .ini file. "
                             "Please check. If "
                             "you do not have this problem, ignore "
                             "the warning." % op_type)
        return ""
