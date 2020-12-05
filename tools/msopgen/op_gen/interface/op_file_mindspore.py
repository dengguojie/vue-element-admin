#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves class for generating mindspore operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
try:
    import os
    import sys
    from .op_file import OPFile
    from . import op_tmpl
    from . import utils
except (ImportError,) as import_error:
    sys.exit("[ERROR][op_file_mindspore] Unable to import module: %s." % str(
        import_error))


class OpFileMindSpore(OPFile):
    """
    CLass for generate mindspore op files
    """

    def generate(self):
        """
        Function Description:
            generate mindspore project or only generator an mindspore operator
            according to mode
        """
        if self.mode == utils.GenModeType.GEN_OPERATOR:
            if self.fmk_type in utils.FMK_MS:
                if os.path.isdir(os.path.join(self.output_path,
                                              utils.PROJ_MS_NAME)):
                    utils.print_info_log("Start to add a new mindspore "
                                         "operator.")

                if not os.path.isdir(os.path.join(self.output_path,
                                                  utils.PROJ_MS_NAME)):
                    utils.print_error_log("A new mindspore operators cannot be"
                                          " added to other operator project!")
                    raise utils.MsOpGenException(
                        utils.MS_OP_GEN_INVALID_PARAM_ERROR)
        else:
            utils.print_info_log("Start to generator a new mindspore project.")
        # generate mindspore operator
        self._new_operator()

    def _new_operator(self):
        self.generate_impl()
        self._generate_op_proto()

    def _generate_mindspore_path(self):
        ms_dir = os.path.join(self.output_path, utils.PROJ_MS_NAME)
        utils.make_dirs(ms_dir)

    def generate_impl(self):
        """
        Function Description:
            generate mindspore operator implementation.
        Parameter:
        Return Value:
        """
        self._generate_mindspore_path()
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty, failed to generate "
                                 "impl files. Please check.")
            return
        # 1.make head string
        head_str = op_tmpl.PY_MS_HEAD
        # 2.make [op_type]_compute()
        op_input = ", ".join(list(self.op_info.parsed_input_info))
        op_input_x = list(self.op_info.parsed_input_info)[0]
        op_output = ", ".join(list(self.op_info.parsed_output_info))
        head_str += op_tmpl.PY_MS_COMPUTE.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            output=op_output)
        # 3.parse op_info
        var_list = []
        # parse inputs
        input_list = []
        for input_name in self.op_info.parsed_input_info:
            var_list.append(input_name)
            str_name = op_tmpl.PY_MS_INPUT_INFO.format(input_name=input_name)
            input_list.append(str_name)
        # parse outputs
        output_list = []
        for output_name in self.op_info.parsed_output_info:
            var_list.append(output_name)
            str_output_name = op_tmpl.PY_MS_OUTPUT_INFO.format(
                output_name=output_name)
            output_list.append(str_output_name)
        # parse dtype_format
        input_attr_list = self.op_info.parsed_input_info.get(var_list[0])
        ir_type_list = input_attr_list.get("ir_type_list")
        data_types_list = []
        if len(ir_type_list) == 1 and ir_type_list[0] == '':
            utils.print_error_log("The attr types in the input file are "
                                  "unsupported. Please check the input or "
                                  "output types.")
            raise utils.MsOpGenException(
                    utils.MS_OP_GEN_INVALID_PARAM_ERROR)
        for dtype_format in ir_type_list:
            type_list = []
            for type_count in range(len(var_list)):
                type_list.append(op_tmpl.PY_MS_DATA_TYPE.format(
                    data_type=dtype_format))
            data_type_join = ', '.join(type_list)
            data_types_list.append(op_tmpl.PY_MS_DTYPE_FORMAT.format(
                data_types_join=data_type_join))

        head_str += op_tmpl.PY_MS_OP_INFO.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            inputs='\n    '.join(input_list),
            outputs='\n    '.join(output_list),
            data_types='\n    '.join(data_types_list))

        # 4.make op_info_register
        head_str += op_tmpl.PY_MS_OP_INFO_REGISTER.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            input_x=op_input_x,
            output=op_output)
        head_str += op_tmpl.PY_MS_OP_INFO_REGISTER_CONFIG

        # create mindspore directory
        py_dir = os.path.join(self.output_path, utils.MS_IMPL_DIR)
        py_path = os.path.join(py_dir, self.op_info.fix_op_type
                               + utils.IMPL_NAME
                               + utils.IMPL_SUFFIX)
        utils.make_dirs(py_dir)
        utils.write_files(py_path, head_str)

    def _generate_op_proto(self):
        if not self.op_info.fix_op_type:
            utils.print_warn_log("The op type is empty, failed to generate "
                                 "op proto files. Please check.")
            return
        template_path = os.path.join(self.output_path, utils.MS_PROTO_PATH)
        utils.make_dirs(template_path)
        self._generate_ms_proto()

    def _generate_ms_proto(self):
        op_input = ", ".join(list(self.op_info.parsed_input_info))
        op_output = ", ".join(list(self.op_info.parsed_output_info))
        # 1.make mindspore proto string
        ms_proto_str = op_tmpl.PY_MS_PROTO_HEAD.format(
            name=self.op_info.fix_op_type,
            up_name=self.op_info.op_type,
            input_name=op_input,
            output=op_output)
        # create ms_proto_dir
        ms_proto_dir = os.path.join(self.output_path, "op_proto")
        ms_proto_path = os.path.join(ms_proto_dir, self.op_info.fix_op_type +
                                     utils.IMPL_SUFFIX)
        utils.make_dirs(ms_proto_dir)
        utils.write_files(ms_proto_path, ms_proto_str)

    def generate_info_cfg(self):
        """
        Function Description:
            generate operator info config file
        Parameter:
        Return Value:
        """
        pass
