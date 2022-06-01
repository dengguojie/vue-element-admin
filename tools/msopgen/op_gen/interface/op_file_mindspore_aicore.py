#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves class for generating mindspore aicore operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import os
from op_gen.interface.op_file import OPFile
from op_gen.interface.op_tmpl import OPTmpl
from op_gen.interface import utils
from op_gen.interface.const_manager import ConstManager


class OpFileMindSporeAiCore(OPFile):
    """
    CLass for generate MindSpore op files
    """
    @staticmethod
    def _parse_op_desc(data_desc: any, parsed_op_info: any) -> any:
        op_desc_list = []
        for index, key_name in enumerate(parsed_op_info):
            if parsed_op_info.get(key_name).get(
                    ConstManager.INFO_PARAM_TYPE_KEY) is None:
                param_type = ConstManager.PARAM_TYPE_REQUIRED
            else:
                param_type = parsed_op_info.get(key_name).get(
                    ConstManager.INFO_PARAM_TYPE_KEY)
            op_desc = OPTmpl.PY_MS_DATA_DESC_INFO.format(
                data_desc=data_desc,
                index=index,
                key_name=key_name,
                param_type=param_type)
            op_desc_list.append(op_desc)
        op_desc_str = ConstManager.NEXT_LINE.join(op_desc_list)
        return op_desc_str

    @staticmethod
    def _get_op_info_dtype_format(parsed_op_info: any) -> list:
        input_info_list = []
        for _, data_info in parsed_op_info.items():
            ir_type_list = data_info.get(ConstManager.INFO_IR_TYPES_KEY)
            type_list = list()
            for dtype_format in ir_type_list:
                if dtype_format == '':
                    type_list.append("")
                    break
                type_list.append(OPTmpl.PY_MS_DATA_TYPE.format(
                    data_type=dtype_format))
            input_info_list.append(type_list)

        def get_index_elements(dtype_list):
            return dtype_list[index]

        all_dtype_format = []
        for index in range(0, len(input_info_list[0])):
            dtype_format_list = list(map(get_index_elements, input_info_list))
            all_dtype_format.append(", ".join(dtype_format_list))
        return all_dtype_format

    def generate_info_cfg(self: any) -> str:
        """
        Function Description:
        generate operator info config file
        Parameter:
        Return Value:""
        """
        return ""

    def generate_impl(self: any) -> None:
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
        # generate operator impl string.
        ms_impl_str = self._generate_op_impl_str()
        # create mindspore directory
        py_dir = os.path.join(self.output_path, ConstManager.MS_IMPL_DIR)
        py_name = '{}{}{}'.format(self.op_info.fix_op_type,
                                  ConstManager.IMPL_NAME,
                                  ConstManager.IMPL_SUFFIX)
        py_path = os.path.join(py_dir, py_name)
        utils.make_dirs(py_dir)
        utils.write_files(py_path, ms_impl_str)

    def generate(self: any) -> None:
        """
        Function Description:
        generate MindSpore project or only generator an MindSpore aicore operator
        according to mode
        """
        if self.mode == ConstManager.GEN_OPERATOR:
            if self.fmk_type in ConstManager.FMK_MS:
                if os.path.isdir(os.path.join(self.output_path,
                                              ConstManager.PROJ_MS_NAME)):
                    utils.print_info_log("Start to add a new mindspore aicore "
                                         "operator.")

                if not os.path.isdir(os.path.join(self.output_path,
                                                  ConstManager.PROJ_MS_NAME)):
                    utils.print_error_log(
                        "The new operator of mindspore aicore operator cannot be"
                        " added to another operator project.")
                    raise utils.MsOpGenException(
                        ConstManager.MS_OP_GEN_INVALID_PARAM_ERROR)
        else:
            utils.print_info_log("Start to generate a new mindspore aicore project.")
        # generate mindspore operator
        self._new_operator()

    def _new_operator(self: any) -> None:
        self.generate_impl()
        self._generate_op_proto()

    def _generate_mindspore_path(self: any) -> None:
        ms_dir = os.path.join(self.output_path, ConstManager.PROJ_MS_NAME)
        utils.make_dirs(ms_dir)

    def _parse_attr_info(self: any) -> list:
        attr_list = []
        for attr_info in self.op_info.parsed_attr_info:
            attr_str = []
            if len(attr_info) == ConstManager.OP_INFO_WITH_PARAM_TYPE_LEN \
                    or len(attr_info) == ConstManager.OP_INFO_WITH_FORMAT_LEN:
                attr_name = attr_info[0]
                attr_type = attr_info[1]
                attr_type_format = utils.CheckFromConfig().trans_ini_attr_type(
                    attr_type)
                param_type = ConstManager.PARAM_TYPE_REQUIRED
                if len(attr_info) == ConstManager.OP_INFO_WITH_FORMAT_LEN:
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
            if attr_str:
                attr_list.append(attr_str)
        return attr_list

    def _get_data_types_list(self: any) -> list:
        all_dtype_format_list = []
        input_type_format = self._get_op_info_dtype_format(
            self.op_info.parsed_input_info)
        output_type_format = self._get_op_info_dtype_format(
            self.op_info.parsed_output_info)
        dtype_format_list = list(
            map(lambda x, y: ', '.join([x, y]), input_type_format,
                output_type_format))
        for dtype_format in dtype_format_list:
            all_dtype_format_list.append(OPTmpl.PY_MS_DTYPE_FORMAT.format(
                data_types_join=dtype_format))
        return all_dtype_format_list

    def _get_op_info_register(self: any) -> any:
        # parse attr information
        attr_list = self._parse_attr_info()
        # get input_desc with attr info
        if attr_list:
            inputs_desc = self._parse_op_desc('input',
                                              self.op_info.parsed_input_info)
            input_str = '{}{}{}'.format(
                ConstManager.NEXT_LINE.join(attr_list), ConstManager.NEXT_LINE,
                inputs_desc)
        else:
            input_str = self._parse_op_desc('input',
                                            self.op_info.parsed_input_info)
        # parse dtype_format
        data_types_list = self._get_data_types_list()
        op_info = OPTmpl.PY_MS_OP_INFO.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            inputs=input_str,
            outputs=self._parse_op_desc('output',
                                        self.op_info.parsed_output_info),
            data_types=ConstManager.NEXT_LINE.join(data_types_list))
        return op_info

    def _generate_op_impl_str(self: any) -> any:
        # 1.make head string
        ms_impl_str = OPTmpl.PY_MS_HEAD
        # 2.format [op_type]_compute()
        op_input = ", ".join(list(self.op_info.parsed_input_info))
        op_input_x = list(self.op_info.parsed_input_info)[0]
        op_output = ", ".join(list(self.op_info.parsed_output_info))
        ms_impl_str += OPTmpl.PY_MS_COMPUTE.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            output=op_output)
        # 3.get operator information
        ms_impl_str += self._get_op_info_register()
        # 4.format operator implement function
        tvm_placeholder_list = []
        input_list = []
        for count_input in range(len(list(self.op_info.parsed_input_info))):
            tvm_placeholder_list.append(
                OPTmpl.PY_MS_OP_INFO_REGISTER_TVM.format(
                    index=count_input))
            input_list.append("input{}".format(count_input))
        ms_impl_str += OPTmpl.PY_MS_OP_INFO_REGISTER.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            input_x=op_input_x,
            output=op_output,
            tvm_placeholder=ConstManager.NEXT_LINE.join(tvm_placeholder_list),
            all_inputs=', '.join(input_list))
        ms_impl_str += OPTmpl.PY_MS_OP_INFO_REGISTER_CONFIG.format(
            all_inputs=', '.join(input_list))
        return ms_impl_str

    def _generate_op_proto(self: any) -> None:
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty. Failed to generate "
                                 "op proto files. Please check.")
            return
        # generate mindspore proto string
        ms_proto_str = self._generate_ms_proto()
        # create ms_proto_dir
        ms_proto_dir = os.path.join(self.output_path,
                                    ConstManager.MS_PROTO_PATH)
        utils.make_dirs(ms_proto_dir)
        proto_file_name = '{}{}'.format(self.op_info.fix_op_type, ConstManager.IMPL_SUFFIX)
        ms_proto_path = os.path.join(ms_proto_dir, proto_file_name)
        utils.make_dirs(ms_proto_dir)
        utils.write_files(ms_proto_path, ms_proto_str)

    def _generate_ms_proto(self: any) -> any:
        input_list = list(self.op_info.parsed_input_info)
        op_output = list(self.op_info.parsed_output_info)
        input_shape_list = []
        input_dtype_list = []
        first_input_shape = ''
        first_input_dtype = ''
        for index, input_name in enumerate(input_list):
            if index == 0:
                first_input_shape = "{}_shape".format(input_name)
                first_input_dtype = "{}_dtype".format(input_name)
            input_shape_list.append("{}_shape".format(input_name))
            input_dtype_list.append("{}_dtype".format(input_name))
        input_shapes = ", ".join(input_shape_list)
        input_dtypes = ", ".join(input_dtype_list)
        # 1.make mindspore proto string
        ms_proto_str = OPTmpl.PY_MS_PROTO_HEAD.format(
            init_attr=self._get_init_attr_name(),
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=str(input_list),
            output=str(op_output),
            op_register_func=''.join([self.op_info.fix_op_type, '_impl']),
            first_input_shape=first_input_shape,
            input_shapes=input_shapes,
            first_input_dtype=first_input_dtype,
            input_dtypes=input_dtypes)
        return ms_proto_str

    def _get_init_attr_name(self):
        attr_list = []
        for index, attr_info in enumerate(self.op_info.parsed_attr_info):
            if attr_info:
                attr_list.append(attr_info[0])
        if len(attr_list) > 0:
            init_attr = 'self, ' + ', '.join(attr_list)
        else:
            init_attr = 'self'
        return init_attr
