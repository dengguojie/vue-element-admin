#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Structures used by DRV
"""
# Standard Packages
import ctypes

MAX_CHIP_NAME = 32


class dsmi_chip_info_stru(ctypes.Structure):
    _fields_ = [('chip_type', ctypes.c_char * MAX_CHIP_NAME),
                ('chip_name', ctypes.c_char * MAX_CHIP_NAME),
                ('chip_ver', ctypes.c_char * MAX_CHIP_NAME)]

    def get_complete_platform(self) -> str:
        res = self.chip_type + self.chip_name
        return res.decode("UTF-8")

    def get_ver(self) -> str:
        return self.chip_ver.decode("UTF-8")


class dsmi_ecc_info_stru(ctypes.Structure):
    _fields_ = [('enable_flag', ctypes.c_int),
                ('single_bit_error_count', ctypes.c_uint),
                ('double_bit_error_count', ctypes.c_uint)]
