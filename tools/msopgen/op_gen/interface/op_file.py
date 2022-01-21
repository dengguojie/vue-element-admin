#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves base class for operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import os
from abc import ABCMeta
from abc import abstractmethod
from . import utils
from .op_tmpl import OPTmpl
from .arg_parser import ArgParser
from .op_info_parser import OpInfoParser
from .const_manager import ConstManager


class OPFile(metaclass=ABCMeta):
    """
    CLass for generate op files
    """

    def __init__(self: any, argument: any) -> None:
        self.mode = argument.mode
        self.output_path = argument.output_path
        self.fmk_type = argument.framework
        self.compute_unit = argument.compute_unit
        self.op_info = OpInfoParser(argument).op_info

    def generate(self: any) -> None:
        """
        Function Description:
        generate project or only generator an operator according to mode
        """
        if self.mode == ConstManager.GEN_OPERATOR:
            if os.path.isdir(os.path.join(
                    self.output_path, ConstManager.PROJ_MS_NAME)):
                utils.print_error_log("MindSpore operators cannot be added to "
                                      "a non-MindSpore operator project.")
                raise utils.MsOpGenException(
                    ConstManager.MS_OP_GEN_INVALID_PARAM_ERROR)
            utils.print_info_log("Start to add a new operator.")
            self._new_operator()
        else:
            utils.print_info_log("Start to generate a new project.")
            self._generate_project()

    def _generate_project(self: any) -> None:
        template_path = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            ConstManager.OP_TEMPLATE_PATH)
        utils.copy_template(template_path, self.output_path)
        self._new_operator()

    def _new_operator(self: any) -> None:
        self.generate_impl()
        self._generate_plugin()
        self.generate_info_cfg()
        self._generate_op_proto()

    def _generate_plugin(self: any) -> None:
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty. Failed to generate "
                                 "plugin files. Please check.")
            return
        if self.fmk_type == "caffe":
            plugin_dir = os.path.join(self.output_path, 'framework',
                                      'caffe_plugin')
            self._generate_caffe_plugin_cpp(plugin_dir, "caffe")
            self._generate_caffe_plugin_cmake_list(plugin_dir)
            custom_proto_path = os.path.join(self.output_path, 'custom.proto')
            utils.write_files(custom_proto_path, OPTmpl.CAFFE_CUSTOM_PROTO)
        elif self.fmk_type == "tensorflow" or self.fmk_type == "tf":
            plugin_dir = os.path.join(self.output_path, 'framework',
                                      'tf_plugin')
            self._generate_tf_plugin_cpp(plugin_dir, "tensorflow")
            self._generate_tf_plugin_cmake_list(plugin_dir)
        elif self.fmk_type == "onnx":
            plugin_dir = os.path.join(self.output_path, 'framework',
                                      'onnx_plugin')
            self._generate_onnx_plugin_cpp(plugin_dir, "onnx")
            self._generate_onnx_plugin_cmake_list(plugin_dir)
        elif self.fmk_type == "pytorch":
            return

    @staticmethod
    def _generate_caffe_plugin_cmake_list(plugin_dir: str) -> None:
        # create and write
        cmake_list_path = os.path.join(plugin_dir, "CMakeLists.txt")
        if os.path.exists(cmake_list_path):
            return
        utils.make_dirs(plugin_dir)
        utils.write_files(cmake_list_path, OPTmpl.CAFFE_PLUGIN_CMAKLIST)

    @staticmethod
    def _generate_tf_plugin_cmake_list(plugin_dir: str) -> None:
        # create and write
        cmake_list_path = os.path.join(plugin_dir, "CMakeLists.txt")
        if os.path.exists(cmake_list_path):
            return
        utils.make_dirs(plugin_dir)
        utils.write_files(cmake_list_path, OPTmpl.PLUGIN_CMAKLIST)

    def _generate_onnx_plugin_cmake_list(self: any, plugin_dir: str) -> None:
        # create and write
        if self.mode == ConstManager.GEN_PROJECT:
            cmake_list_path = os.path.join(plugin_dir, "CMakeLists.txt")
            utils.make_dirs(plugin_dir)
            utils.write_files(cmake_list_path, OPTmpl.ONNX_PLUGIN_CMAKLIST)

    def _generate_caffe_plugin_cpp(self: any, plugin_dir: str, prefix: str) -> None:
        p_str = OPTmpl.CAFFE_PLUGIN_CPP.format(left_braces=ConstManager.LEFT_BRACES,
                                                name=self.op_info.op_type,
                                                fmk_type=prefix.upper(),
                                                right_braces=ConstManager.RIGHT_BRACES)
        # create dir and write
        plugin_path = os.path.join(plugin_dir, prefix + "_" +
                                   self.op_info.fix_op_type + "_plugin.cc")
        utils.make_dirs(plugin_dir)
        utils.write_files(plugin_path, p_str)

    def _generate_tf_plugin_cpp(self: any, plugin_dir: str, prefix: str) -> None:
        p_str = OPTmpl.TF_PLUGIN_CPP.format(left_braces=ConstManager.LEFT_BRACES,
                                             name=self.op_info.op_type,
                                             fmk_type=prefix.upper(),
                                             right_braces=ConstManager.RIGHT_BRACES)
        # create dir and write
        plugin_path = os.path.join(plugin_dir, prefix + "_" +
                                   self.op_info.fix_op_type + "_plugin.cc")
        utils.make_dirs(plugin_dir)
        utils.write_files(plugin_path, p_str)

    def _generate_onnx_plugin_cpp(self: any, plugin_dir: str, prefix: str) -> None:
        p_str = OPTmpl.ONNX_PLUGIN_CPP.format(left_braces=ConstManager.LEFT_BRACES,
                                               name=self.op_info.op_type,
                                               fmk_type=prefix.upper(),
                                               right_braces=ConstManager.RIGHT_BRACES)
        # create dir and write
        plugin_path = os.path.join(plugin_dir, self.op_info.fix_op_type
                                   + "_plugin.cc")
        utils.make_dirs(plugin_dir)
        utils.write_files(plugin_path, p_str)

    def _generate_op_proto(self: any) -> None:
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty. Failed to generate "
                                 "op proto files. Please check.")
            return
        self._generate_ir_h()
        self._generate_ir_cpp()

    def _generate_ir_h(self: any) -> None:
        head_str = OPTmpl.IR_H_HEAD.format(
            left_braces=ConstManager.LEFT_BRACES,
            op_type_upper=self.op_info.fix_op_type.upper(),
            op_type=self.op_info.op_type)
        # generate input
        for (name, value) in self.op_info.parsed_input_info.items():
            if value[ConstManager.INFO_PARAM_TYPE_KEY] == ConstManager.PARAM_TYPE_DYNAMIC:
                template_str = OPTmpl.IR_H_DYNAMIC_INPUT
            else:
                template_str = OPTmpl.IR_H_INPUT
            input_type = ",".join(value[ConstManager.INFO_IR_TYPES_KEY])
            head_str += template_str.format(name=name, type=input_type)
        # generate output
        for (name, value) in self.op_info.parsed_output_info.items():
            if value[ConstManager.INFO_PARAM_TYPE_KEY] == ConstManager.PARAM_TYPE_DYNAMIC:
                template_str = OPTmpl.IR_H_DYNAMIC_OUTPUT
            else:
                template_str = OPTmpl.IR_H_OUTPUT
            output_type = ",".join(value[ConstManager.INFO_IR_TYPES_KEY])
            head_str += template_str.format(name=name, type=output_type)
        # generate attr
        for attr in self.op_info.parsed_attr_info:
            head_str = self._generate_attr(attr, head_str)
        head_str += OPTmpl.IR_H_END.format(
            op_type=self.op_info.op_type,
            right_braces=ConstManager.RIGHT_BRACES,
            op_type_upper=self.op_info.fix_op_type.upper())
        ir_h_dir = os.path.join(self.output_path, "op_proto")
        ir_h_path = os.path.join(ir_h_dir, self.op_info.fix_op_type + ".h")
        # create and write
        utils.make_dirs(ir_h_dir)
        utils.write_files(ir_h_path, head_str)

    def _generate_attr(self: any, attr: list, head_str: str) -> str:
        attr_name = utils.fix_name_lower_with_under(attr[0])
        attr_type = attr[1]
        if (len(attr) == 4 and attr[3] == "optional") or (len(attr) == 3 and attr[2] != ""):
            default_value = self._deal_with_default_value(attr_type,
                                                          attr[2])
            head_str += OPTmpl.IR_H_ATTR_WITH_VALUE.format(
                name=attr_name,
                type=attr_type,
                value=default_value)
        else:
            head_str += OPTmpl.IR_H_ATTR_WITHOUT_VALUE.format(
                name=attr_name,
                type=attr_type)
        return head_str

    @staticmethod
    def _deal_with_default_value(attr_type: str, default_value: any) -> any:
        if attr_type.startswith("List"):
            if isinstance(default_value, list):
                default_value = str(default_value).replace('[', '{') \
                    .replace(']', '}')
        return default_value

    def _generate_ir_cpp(self: any) -> None:
        cpp_str = OPTmpl.IR_CPP_HEAD.format(
            fix_op_type=self.op_info.fix_op_type,
            op_type=self.op_info.op_type,
            left_braces=ConstManager.LEFT_BRACES,
            right_braces=ConstManager.RIGHT_BRACES)
        ir_cpp_dir = os.path.join(self.output_path, "op_proto")
        ir_cpp_path = os.path.join(ir_cpp_dir, self.op_info.fix_op_type +
                                   ".cc")
        utils.make_dirs(ir_cpp_dir)
        utils.write_files(ir_cpp_path, cpp_str)

    @abstractmethod
    def generate_impl(self: any) -> None:
        """
        Function Description:
        generate operator implementation.
        Parameter:
        Return Value:
        """

    @abstractmethod
    def generate_info_cfg(self: any) -> None:
        """
        Function Description:
        generate operator info config file
        Parameter:
        Return Value:
        """
