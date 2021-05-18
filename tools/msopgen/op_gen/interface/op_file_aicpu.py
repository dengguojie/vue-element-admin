#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves class for generating aicpu operator files.
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
    sys.exit("[ERROR][op_file_aicpu]Unable to import module: %s." % str(
        import_error))


class OpFileAiCpu(OPFile):
    """
    CLass for generate aicpu op files
    """

    def generate_impl(self):
        """
        Function Description:
            generate operator implementation.
        Parameter:
        Return Value:
        """
        op_info = self.op_info
        if self.mode == utils.GenModeType.GEN_PROJECT:
            self._generate_cmake_lists()
        self._generate_impl_cc(op_info)
        self._generate_impl_h(op_info)

    def _generate_cmake_lists(self):
        impl_dir = os.path.join(self.output_path, 'cpukernel')
        utils.make_dirs(impl_dir)
        template_path = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            utils.OP_TEMPLATE_AICPU_PATH)
        utils.copy_template(template_path, impl_dir, True)

    def _generate_impl_cc(self, op_info):
        cc_str = op_tmpl.AICPU_IMPL_CPP_STRING.format(
            fix_op_type=op_info.fix_op_type,
            op_type=op_info.op_type,
            op_type_upper=op_info.fix_op_type.upper(),
            left_braces=utils.LEFT_BRACES,
            right_braces=utils.RIGHT_BRACES)
        impl_dir = os.path.join(self.output_path, 'cpukernel', 'impl')
        cc_path = os.path.join(impl_dir, op_info.fix_op_type + '_kernels.cc')
        # create dir and write impl file
        utils.make_dirs(impl_dir)
        utils.write_files(cc_path, cc_str)

    def _generate_impl_h(self, op_info):
        h_str = op_tmpl.AICPU_IMPL_H_STRING.format(
            op_type=op_info.op_type,
            op_type_upper=op_info.fix_op_type.upper(),
            left_braces=utils.LEFT_BRACES,
            right_braces=utils.RIGHT_BRACES)
        impl_dir = os.path.join(self.output_path, 'cpukernel', 'impl')
        h_path = os.path.join(impl_dir, op_info.fix_op_type + '_kernels.h')
        # create dir and write impl file
        utils.make_dirs(impl_dir)
        utils.write_files(h_path, h_str)

    def generate_info_cfg(self):
        """
        Function Description:
            generate operator info config file
        Parameter:
        Return Value:
        """
        op_info = self.op_info
        new_str = op_tmpl.AICPU_INI_STRING.format(op_type=op_info.op_type)
        # create dir and write ini file
        info_dir = os.path.join(self.output_path, 'cpukernel',
                                'op_info_cfg', 'aicpu_kernel')
        info_path = os.path.join(info_dir, self.op_info.fix_op_type + ".ini")
        utils.make_dirs(info_dir)
        utils.write_files(info_path, new_str)
