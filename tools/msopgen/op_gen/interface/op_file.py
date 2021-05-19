#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves base class for operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import os
    import sys
    from abc import ABCMeta
    from abc import abstractmethod
    from . import utils
    from . import op_tmpl
    from .arg_parser import ArgParser
    from .op_info_parser import OpInfoParser
except (ImportError,) as import_error:
    sys.exit("[ERROR][op_file] Unable to import module: %s." % str(
        import_error))


class OPFile(metaclass=ABCMeta):
    """
    CLass for generate op files
    """

    def __init__(self, argument: ArgParser):
        self.mode = argument.mode
        self.output_path = argument.output_path
        self.fmk_type = argument.framework
        self.compute_unit = argument.compute_unit
        self.op_info = OpInfoParser(argument).op_info

    def generate(self):
        """
        Function Description:
            generate project or only generator an operator according to mode
        """
        if self.mode == utils.GenModeType.GEN_OPERATOR:
            if os.path.isdir(os.path.join(
                    self.output_path, utils.PROJ_MS_NAME)):
                utils.print_error_log("Mindspore operator cannot be added to "
                                      "a non-mindspore operator project!")
                raise utils.MsOpGenException(
                    utils.MS_OP_GEN_INVALID_PARAM_ERROR)
            utils.print_info_log("Start to add a new operator.")
            self._new_operator()
        else:
            utils.print_info_log("Start to generator a new project.")
            self._generate_project()

    def _generate_project(self):
        template_path = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            utils.OP_TEMPLATE_PATH)
        utils.copy_template(template_path, self.output_path)
        self._new_operator()

    def _new_operator(self):
        self.generate_impl()
        self._generate_plugin()
        self.generate_info_cfg()
        self._generate_op_proto()

    def _generate_plugin(self):
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty, failed to generate "
                                 "plugin files. Please check.")
            return
        if self.fmk_type == "caffe":
            plugin_dir = os.path.join(self.output_path, 'framework',
                                      'caffe_plugin')
            self._generate_caffe_plugin_cpp(plugin_dir, "caffe")
            self._generate_caffe_plugin_cmake_list(plugin_dir)
            custom_proto_path = os.path.join(self.output_path, 'custom.proto')
            utils.write_files(custom_proto_path, op_tmpl.CAFFE_CUSTOM_PROTO)
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

    def _generate_caffe_plugin_cmake_list(self, plugin_dir):
        # create and write
        if self.mode == utils.GenModeType.GEN_PROJECT:
            cmake_list_path = os.path.join(plugin_dir, "CMakeLists.txt")
            utils.make_dirs(plugin_dir)
            utils.write_files(cmake_list_path, op_tmpl.CAFFE_PLUGIN_CMAKLIST)

    def _generate_tf_plugin_cmake_list(self, plugin_dir):
        # create and write
        if self.mode == utils.GenModeType.GEN_PROJECT:
            cmake_list_path = os.path.join(plugin_dir, "CMakeLists.txt")
            utils.make_dirs(plugin_dir)
            utils.write_files(cmake_list_path, op_tmpl.PLUGIN_CMAKLIST)

    def _generate_onnx_plugin_cmake_list(self, plugin_dir):
        # create and write
        if self.mode == utils.GenModeType.GEN_PROJECT:
            cmake_list_path = os.path.join(plugin_dir, "CMakeLists.txt")
            utils.make_dirs(plugin_dir)
            utils.write_files(cmake_list_path, op_tmpl.ONNX_PLUGIN_CMAKLIST)

    def _generate_caffe_plugin_cpp(self, plugin_dir, prefix):
        p_str = op_tmpl.CAFFE_PLUGIN_CPP.format(left_braces=utils.LEFT_BRACES,
                                                name=self.op_info.op_type,
                                                fmk_type=prefix.upper(),
                                                right_braces=utils.RIGHT_BRACES)
        # create dir and write
        plugin_path = os.path.join(plugin_dir, prefix + "_" +
                                   self.op_info.fix_op_type + "_plugin.cc")
        utils.make_dirs(plugin_dir)
        utils.write_files(plugin_path, p_str)

    def _generate_tf_plugin_cpp(self, plugin_dir, prefix):
        p_str = op_tmpl.TF_PLUGIN_CPP.format(left_braces=utils.LEFT_BRACES,
                                             name=self.op_info.op_type,
                                             fmk_type=prefix.upper(),
                                             right_braces=utils.RIGHT_BRACES)
        # create dir and write
        plugin_path = os.path.join(plugin_dir, prefix + "_" +
                                   self.op_info.fix_op_type + "_plugin.cc")
        utils.make_dirs(plugin_dir)
        utils.write_files(plugin_path, p_str)

    def _generate_onnx_plugin_cpp(self, plugin_dir, prefix):
        p_str = op_tmpl.ONNX_PLUGIN_CPP.format(left_braces=utils.LEFT_BRACES,
                                               name=self.op_info.op_type,
                                               fmk_type=prefix.upper(),
                                               right_braces=utils.RIGHT_BRACES)
        # create dir and write
        plugin_path = os.path.join(plugin_dir, self.op_info.fix_op_type
                                   + "_plugin.cc")
        utils.make_dirs(plugin_dir)
        utils.write_files(plugin_path, p_str)

    def _generate_op_proto(self):
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty, failed to generate "
                                 "op proto files. Please check.")
            return
        self._generate_ir_h()
        self._generate_ir_cpp()

    def _generate_ir_h(self):
        head_str = op_tmpl.IR_H_HEAD.format(
            left_braces=utils.LEFT_BRACES,
            op_type_upper=self.op_info.fix_op_type.upper(),
            op_type=self.op_info.op_type)
        # generate input
        for (name, value) in self.op_info.parsed_input_info.items():
            if value[utils.INFO_PARAM_TYPE_KEY] == utils.PARAM_TYPE_DYNAMIC:
                template_str = op_tmpl.IR_H_DYNAMIC_INPUT
            else:
                template_str = op_tmpl.IR_H_INPUT
            input_type = ",".join(value[utils.INFO_IR_TYPES_KEY])
            head_str += template_str.format(name=name, type=input_type)
        # generate output
        for (name, value) in self.op_info.parsed_output_info.items():
            if value[utils.INFO_PARAM_TYPE_KEY] == utils.PARAM_TYPE_DYNAMIC:
                template_str = op_tmpl.IR_H_DYNAMIC_OUTPUT
            else:
                template_str = op_tmpl.IR_H_OUTPUT
            output_type = ",".join(value[utils.INFO_IR_TYPES_KEY])
            head_str += template_str.format(name=name, type=output_type)
        # generate attr
        for attr in self.op_info.parsed_attr_info:
            head_str = self._generate_attr(attr, head_str)
        head_str += op_tmpl.IR_H_END.format(
            op_type=self.op_info.op_type,
            right_braces=utils.RIGHT_BRACES,
            op_type_upper=self.op_info.fix_op_type.upper())
        ir_h_dir = os.path.join(self.output_path, "op_proto")
        ir_h_path = os.path.join(ir_h_dir, self.op_info.fix_op_type + ".h")
        # create and write
        utils.make_dirs(ir_h_dir)
        utils.write_files(ir_h_path, head_str)

    def _generate_attr(self, attr, head_str):
        attr_name = utils.fix_name_lower_with_under(attr[0])
        attr_type = attr[1]
        if len(attr) == 4:
            if attr[3] == "optional":
                default_value = self._deal_with_default_value(attr_type,
                                                              attr[2])
                head_str += op_tmpl.IR_H_ATTR_WITH_VALUE.format(
                    name=attr_name,
                    type=attr_type,
                    value=default_value)
            else:
                head_str += op_tmpl.IR_H_ATTR_WITHOUT_VALUE.format(
                    name=attr_name,
                    type=attr_type)
        elif len(attr) == 3 and attr[2] != "":
            default_value = self._deal_with_default_value(attr_type,
                                                          attr[2])
            head_str += op_tmpl.IR_H_ATTR_WITH_VALUE.format(name=attr_name,
                                                            type=attr_type,
                                                            value=default_value
                                                            )
        else:
            head_str += op_tmpl.IR_H_ATTR_WITHOUT_VALUE.format(
                name=attr_name,
                type=attr_type)
        return head_str

    @staticmethod
    def _deal_with_default_value(attr_type, default_value):
        if attr_type.startswith("List"):
            if isinstance(default_value, list):
                default_value = str(default_value).replace('[', '{') \
                    .replace(']', '}')
        return default_value

    def _generate_ir_cpp(self):
        cpp_str = op_tmpl.IR_CPP_HEAD.format(
            fix_op_type=self.op_info.fix_op_type,
            op_type=self.op_info.op_type,
            left_braces=utils.LEFT_BRACES,
            right_braces=utils.RIGHT_BRACES)
        ir_cpp_dir = os.path.join(self.output_path, "op_proto")
        ir_cpp_path = os.path.join(ir_cpp_dir, self.op_info.fix_op_type +
                                   ".cc")
        utils.make_dirs(ir_cpp_dir)
        utils.write_files(ir_cpp_path, cpp_str)

    @abstractmethod
    def generate_impl(self):
        """
        Function Description:
            generate operator implementation.
        Parameter:
        Return Value:
        """

    @abstractmethod
    def generate_info_cfg(self):
        """
        Function Description:
            generate operator info config file
        Parameter:
        Return Value:
        """
