#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves class for generating mindspore operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import os
from .op_file import OPFile
from .op_tmpl import OPTmpl
from . import utils


class OpFileMindSpore(OPFile):
    """
    CLass for generate MindSpore op files
    """

    def generate(self):
        """
        Function Description:
        generate MindSpore project or only generator an MindSpore operator
        according to mode
        """
        if self.mode == utils.GEN_OPERATOR:
            if self.fmk_type in utils.FMK_MS:
                if os.path.isdir(os.path.join(self.output_path,
                                              utils.PROJ_MS_NAME)):
                    utils.print_info_log("Start to add a new MindSpore "
                                         "operator.")

                if not os.path.isdir(os.path.join(self.output_path,
                                                  utils.PROJ_MS_NAME)):
                    utils.print_error_log("The new MindSpore operator cannot be"
                                          " added to another operator project.")
                    raise utils.MsOpGenException(
                        utils.MS_OP_GEN_INVALID_PARAM_ERROR)
        else:
            utils.print_info_log("Start to generate a new MindSpore project.")
        # generate mindspore operator
        self._new_operator()

    def _new_operator(self):
        self.generate_impl()
        self._generate_op_proto()

    def _generate_mindspore_path(self):
        ms_dir = os.path.join(self.output_path, utils.PROJ_MS_NAME)
        utils.make_dirs(ms_dir)

    def _parse_attr_info(self):
        attr_list = []
        for attr_info in self.op_info.parsed_attr_info:
            attr_str = []
            if len(attr_info) == utils.OP_INFO_WITH_PARAM_TYPE_LEN \
                    or len(attr_info) == utils.OP_INFO_WITH_FORMAT_LEN:
                attr_name = attr_info[0]
                attr_type = attr_info[1]
                attr_type_format = utils.CheckFromConfig().trans_ini_attr_type(
                    attr_type)
                param_type = "required"
                if len(attr_info) == utils.OP_INFO_WITH_FORMAT_LEN:
                    param_type = attr_info[3]
                attr_str = OPTmpl.PY_MS_ATTR_WITHOUT_VALUE_INFO.format(
                    attr_name=attr_name,
                    param_type=param_type,
                    attr_type=attr_type_format)
            else:
                if len(attr_info) > 0:
                    attr_name = attr_info[0]
                    utils.print_warn_log(
                        "The attr:'%s' in the .txt file cannot be parsed."
                        % attr_name)
            attr_list.append(attr_str)
        return attr_list

    def _parse_input_output(self, var_list):
        # parse inputs
        input_list = []
        for input_name in self.op_info.parsed_input_info:
            var_list.append(input_name)
            str_name = OPTmpl.PY_MS_INPUT_INFO.format(input_name=input_name)
            input_list.append(str_name)
        # parse outputs
        output_list = []
        for output_name in self.op_info.parsed_output_info:
            var_list.append(output_name)
            str_output_name = OPTmpl.PY_MS_OUTPUT_INFO.format(
                output_name=output_name)
            output_list.append(str_output_name)
        return input_list, output_list

    def _parse_op_info(self, head_str):
        var_list = []
        # parse attr information
        attr_valid_list = []
        attr_list = self._parse_attr_info()
        for attr_item in attr_list:
            # remove attr when it is empty list
            if attr_item:
                attr_valid_list.append(attr_item)
        input_list, output_list = self._parse_input_output(var_list)
        # parse dtype_format
        input_attr_list = self.op_info.parsed_input_info.get(var_list[0])
        ir_type_list = input_attr_list.get("ir_type_list")
        data_types_list = []
        for dtype_format in ir_type_list:
            type_list = []
            for _ in range(len(var_list)):
                if dtype_format == '':
                    type_list.append("")
                    break
                type_list.append(OPTmpl.PY_MS_DATA_TYPE.format(
                    data_type=dtype_format))
            data_type_join = ', '.join(type_list)
            data_types_list.append(OPTmpl.PY_MS_DTYPE_FORMAT.format(
                data_types_join=data_type_join))
        if attr_valid_list:
            head_str += OPTmpl.PY_MS_OP_WITH_ATTR_INFO.format(
                name=self.op_info.fix_op_type,
                up_name=self.op_info.op_type,
                attrs='\n    '.join(attr_list),
                inputs='\n    '.join(input_list),
                outputs='\n    '.join(output_list),
                data_types='\n    '.join(data_types_list))
        else:
            head_str += OPTmpl.PY_MS_OP_WITHOUT_ATTR_INFO.format(
                name=self.op_info.fix_op_type,
                up_name=self.op_info.op_type,
                inputs='\n    '.join(input_list),
                outputs='\n    '.join(output_list),
                data_types='\n    '.join(data_types_list))
        return head_str

    def generate_impl(self):
        """
        Function Description:
        generate mindspore operator implementation.
        Parameter:
        Return Value:
        """
        self._generate_mindspore_path()
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty. Failed to generate "
                                 "impl files. Please check.")
            return
        # 1.make head string
        head_str = OPTmpl.PY_MS_HEAD
        # 2.make [op_type]_compute()
        op_input = ", ".join(list(self.op_info.parsed_input_info))
        op_input_x = list(self.op_info.parsed_input_info)[0]
        op_output = ", ".join(list(self.op_info.parsed_output_info))
        head_str += OPTmpl.PY_MS_COMPUTE.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            output=op_output)
        # 3.parse op_info
        head_str = self._parse_op_info(head_str)

        # 4.make op_info_register
        tvm_placeholder_list = []
        datas_list = []
        for data_count in range(len(list(self.op_info.parsed_input_info))):
            tvm_placeholder_list.append(OPTmpl.PY_MS_OP_INFO_REGISTER_TVM.format(
                data_count=data_count+1))
            datas_list.append("data{num}".format(num=data_count+1))
        tvm_placeholder_join = '\n    '.join(tvm_placeholder_list)
        datas_join = ', '.join(datas_list)
        head_str += OPTmpl.PY_MS_OP_INFO_REGISTER.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            input_x=op_input_x,
            output=op_output,
            tvm_placeholder=tvm_placeholder_join,
            datas_join=datas_join)
        head_str += OPTmpl.PY_MS_OP_INFO_REGISTER_CONFIG.format(datas_join=datas_join)

        # create mindspore directory
        py_dir = os.path.join(self.output_path, utils.MS_IMPL_DIR)
        py_path = os.path.join(py_dir, self.op_info.fix_op_type
                               + utils.IMPL_NAME
                               + utils.IMPL_SUFFIX)
        utils.make_dirs(py_dir)
        utils.write_files(py_path, head_str)

    def _generate_op_proto(self):
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty. Failed to generate "
                                 "op proto files. Please check.")
            return
        template_path = os.path.join(self.output_path, utils.MS_PROTO_PATH)
        utils.make_dirs(template_path)
        self._generate_ms_proto()

    def _generate_ms_proto(self):
        input_list = list(self.op_info.parsed_input_info)
        op_input = ", ".join(input_list)
        op_output = ", ".join(list(self.op_info.parsed_output_info))
        data_shape_list = []
        data_dtype_list = []
        for input_count in range(len(input_list)):
            data_shape_list.append("data{num}_shape".format(num=input_count+1))
            data_dtype_list.append("data{num}_dtype".format(num=input_count+1))
        data_shapes = ", ".join(data_shape_list)
        data_dtypes = ", ".join(data_dtype_list)
        # 1.make mindspore proto string
        ms_proto_str = OPTmpl.PY_MS_PROTO_HEAD.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            output=op_output,
            data_shapes=data_shapes,
            data_dtypes=data_dtypes)
        # create ms_proto_dir
        ms_proto_dir = os.path.join(self.output_path, "op_proto")
        ms_proto_path = os.path.join(ms_proto_dir, self.op_info.fix_op_type +
                                     utils.IMPL_SUFFIX)
        utils.make_dirs(ms_proto_dir)
        utils.write_files(ms_proto_path, ms_proto_str)

    @staticmethod
    def generate_info_cfg():
        """
        Function Description:
        generate operator info config file
        Parameter:
        Return Value:""
        """
        return ""
