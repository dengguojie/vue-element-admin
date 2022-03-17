#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves class for generating aicpu operator files.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
import os
from op_gen.interface.op_file import OPFile
from op_gen.interface.op_tmpl import OPTmpl
from op_gen.interface import utils
from op_gen.interface.const_manager import ConstManager


class OpFileAiCpu(OPFile):
    """
    CLass for generate aicpu op files
    """

    def generate_impl(self: any) -> None:
        """
        Function Description:
        generate operator implementation.
        Parameter:
        Return Value:
        """
        op_info = self.op_info
        self._generate_cmake_lists()
        self._generate_impl_cc(op_info)
        self._generate_impl_h(op_info)

    def _generate_cmake_lists(self: any) -> None:
        impl_dir = os.path.join(self.output_path, 'cpukernel')
        if os.path.exists(impl_dir):
            return
        utils.make_dirs(impl_dir)
        template_path = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            ConstManager.OP_TEMPLATE_AICPU_PATH)
        utils.copy_template(template_path, impl_dir, True)

    def _generate_impl_cc(self: any, op_info: any) -> None:
        cc_str = OPTmpl.AICPU_IMPL_CPP_STRING.format(
            fix_op_type=op_info.fix_op_type,
            op_type=op_info.op_type,
            op_type_upper=op_info.fix_op_type.upper(),
            left_braces=ConstManager.LEFT_BRACES,
            right_braces=ConstManager.RIGHT_BRACES)
        impl_dir = os.path.join(self.output_path, 'cpukernel', 'impl')
        cc_path = os.path.join(impl_dir, op_info.fix_op_type + '_kernels.cc')
        # create dir and write impl file
        utils.make_dirs(impl_dir)
        utils.write_files(cc_path, cc_str)

    def _generate_impl_h(self: any, op_info: any) -> None:
        h_str = OPTmpl.AICPU_IMPL_H_STRING.format(
            op_type=op_info.op_type,
            op_type_upper=op_info.fix_op_type.upper(),
            left_braces=ConstManager.LEFT_BRACES,
            right_braces=ConstManager.RIGHT_BRACES)
        impl_dir = os.path.join(self.output_path, 'cpukernel', 'impl')
        h_path = os.path.join(impl_dir, op_info.fix_op_type + '_kernels.h')
        # create dir and write impl file
        utils.make_dirs(impl_dir)
        utils.write_files(h_path, h_str)

    def generate_info_cfg(self: any) -> None:
        """
        Function Description:
        generate operator info config file
        Parameter:
        Return Value:
        """
        op_info = self.op_info
        new_str = OPTmpl.AICPU_INI_STRING.format(op_type=op_info.op_type)
        # create dir and write ini file
        info_dir = os.path.join(self.output_path, 'cpukernel',
                                'op_info_cfg', 'aicpu_kernel')
        info_path = os.path.join(info_dir, self.op_info.fix_op_type + ".ini")
        utils.make_dirs(info_dir)
        utils.write_files(info_path, new_str)
