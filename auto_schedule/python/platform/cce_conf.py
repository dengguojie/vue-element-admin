#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

parser the config params
"""
from __future__ import absolute_import as _abs

from te import tvm
from te import platform as cce
# pylint: disable=import-error
from .cce_policy import OpImplPolicy
from tvm._ffi.function import _init_api
from .cce_params import TRANS_TIK_API_TO_INSTR_MAP
from .cce_params import ONLY_TIK_API_MAP
from .cce_policy import set_L1_info
from .insn_cmd import get_insn_cmd
from .cce_params import ASCEND_910
from .cce_params import ASCEND_910H
from .cce_params import ASCEND_910M
from .cce_params import ASCEND_910P


# product version
# This is used for DSL/AutoSchedule ONLY!
# For other components, use te.platform.get_soc_spec("SOC_VERSION")!
VERSION_CLOUD = "1980"
VERSION_MINI = "1910"
VERSION_MINI_1951 = "1951dc"
VERSION_MINI_1951M = "1951mdc"
VERSION_MINI_1951PG2 = "1951pg2"
VERSION_SHISI = "smallhisi"
AIC = "AiCore"
AIV = "VectorCore"
SOC_VERSION_MAP = {
    "Ascend310": {
        AIC: {"AICoreNum": 2, "ProductVersion": VERSION_MINI},
    },
    "Ascend910": {
        AIC: {"AICoreNum": 32, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910Pro": {
        AIC: {"AICoreNum": 32, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend910Lite": {
        AIC: {"AICoreNum": 30, "ProductVersion": VERSION_CLOUD},
    },
    "Ascend710": {
        AIC: {"AICoreNum": 8, "ProductVersion": VERSION_MINI_1951},
        AIV: {"AICoreNum": 7, "ProductVersion": VERSION_MINI_1951},
    },
    "Ascend610": {
        AIC: {"AICoreNum": 10, "ProductVersion": VERSION_MINI_1951M},
        AIV: {"AICoreNum": 8, "ProductVersion": VERSION_MINI_1951M},
    },
    "Ascend615": {
        AIC: {"AICoreNum": 10, "ProductVersion": VERSION_MINI_1951PG2},
        AIV: {"AICoreNum": 8, "ProductVersion": VERSION_MINI_1951PG2 + AIV},
    },
    "Hi3796CV300ES": {
        AIC: {"AICoreNum": 1, "ProductVersion": VERSION_SHISI},
    },
    "Hi3796CV300CS": {
        AIC: {"AICoreNum": 1, "ProductVersion": VERSION_SHISI},
    },
    "Hi3519AV200": {
        AIC: {"AICoreNum": 1, "ProductVersion": None},
    },
}


# pylint: disable=invalid-name
def getValue(key):
    """
    call global func to get product value

    Parameters
        ----------
        key : str
            key
    """
    if "Buffer" in key:
        func = tvm.get_global_func("cce.product_conf_buffer")
        value = func(key)
        if value == -1:
            raise RuntimeError("Unsupported buffer name: %s" % key.split("_Buffer")[0])
        return value

    if "Compiler" in key:
        func = tvm.get_global_func("cce.product_conf_compiler")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported compiler param: %s" % key.split("Compiler_")[1])
        return value

    if "Intrinsic" in key:
        func = tvm.get_global_func("cce.product_conf_intrinsic")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported intrinsic: %s" % key.split("Intrinsic_")[1])
        return value

    if "Sid" in key:
        func = tvm.get_global_func("cce.product_conf_sid")
        value = func(key)
        if value == "":
            raise RuntimeError("Unsupported sid param: %s" % key.split("Sid_")[1])
        return value

    if "Device" in key:
        func = tvm.get_global_func("cce.product_conf_device")
        value = func(key)
        if value == -1:
            raise RuntimeError("Unsupported device param: %s" % key.split("Device_")[1])
        return value

    return None


# pylint: disable=useless-object-inheritance, bad-classmethod-argument, invalid-name
class CceProductParams(object):
    """
    define Cce Product Params class
    """
    __instance = None

    cce_product = None

    def __init__(self):
        pass

        # singletom pattern

    def __new__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = object.__new__(self, *args, **kwargs)
        return self.__instance

    def getParams(self, key):
        """
        get Cce Product Params info
        """
        if self.cce_product is None:
            raise RuntimeError("not set product info")

        value = getValue(key)

        # if product supports os
        if key == "Compiler_aicpu_support_os":
            # string to bool
            value = (value == "true")

        return value

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SOC_VERSION")!
    def get_product_version(self):
        """
        get product version
        ----------

        Returns
        -------
        cloud: cloud product
        mini: mini product
        """
        return self.cce_product

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SOC_VERSION")!
    def is_mini_version(self):
        """
        check if mini version
        -------

        Returns
        -------
        True: mini version
        False: Other version
        """
        if self.cce_product == VERSION_MINI:
            return True
        return False

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SOC_VERSION")!
    def is_cloud_version(self):
        """
        check if cloud-1980 version
        ----------

        Returns
        -------
        True: cloud version
        False: Other version
        """
        if self.cce_product == VERSION_CLOUD:
            return True
        return False

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SOC_VERSION")!
    def is_lhisi_version(self):
        """
        check if 3796ES version
        -------

        Returns
        -------
        True: 3796ES version
        False: Other version
        """
        if self.cce_product == VERSION_SHISI:
            return True
        return False

    # This is used for DSL/AutoSchedule ONLY!
    # For other components, use te.platform.get_soc_spec("SOC_VERSION")!
    def is_1951_version(self):
        """
        check if 1951/1951m version
        ----------

        Returns
        -------
        True: 1951 version
        False: Other version
        """
        if self.cce_product in (VERSION_MINI_1951, VERSION_MINI_1951M, VERSION_MINI_1951PG2):
            return True
        return False


def set_status_check(bool_status):
    """
    call global func to set debug mode to
    add status special register check code
    to check if the compute overflow.

    Parameters
        ----------
        bool_status : boolean
            when true, the code will print the check code
    """
    if not isinstance(bool_status, bool):
        raise RuntimeError("The input value type must be boolean")

    func = tvm.get_global_func("cce.status_check")

    func(bool_status)



def _check_soc_version(soc_version, core_type):
    # check Soc_Vesion
    if not isinstance(soc_version, str):
        raise RuntimeError("Soc_Vesion type should be 'str', it is [%s]"
                           % type(soc_version))
    if soc_version not in SOC_VERSION_MAP:
        raise RuntimeError("Unsupported Soc_Vesion: %s" % soc_version)

    # check Core_Type
    if not isinstance(core_type, str):
        raise RuntimeError("Core_Type type should be 'str', it is [%s]"
                           % type(core_type))
    if core_type not in SOC_VERSION_MAP[soc_version]:
        raise RuntimeError("%s Unsupported Core_Type: %s"
                           % (soc_version, core_type))


def _check_and_get_aicore_num(soc_version, core_type, aicore_num):
    # check AICore_Num
    max_aicore_num = SOC_VERSION_MAP[soc_version][core_type]["AICoreNum"]
    if aicore_num in [None, "0", 0, ""]:
        aicore_num = ""
    elif isinstance(aicore_num, int):
        if not 0 < aicore_num <= max_aicore_num:
            raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)
        aicore_num = str(aicore_num)
    elif isinstance(aicore_num, str):
        try:
            check_num = int(aicore_num)
        except Exception:
            raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)
        if not 0 < check_num <= max_aicore_num:
            raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)
    else:
        raise RuntimeError("Unsupported AICore_Num: %s" % aicore_num)

    return aicore_num


def _check_and_get_l1_fusion(l1_fusion):
    # check l1_fusion
    if l1_fusion is None:
        l1_fusion = ""
    elif l1_fusion is True:
        l1_fusion = "true"
    elif l1_fusion is False:
        l1_fusion = "false"
    elif l1_fusion in ("True", "False", "TRUE", "FALSE", "true", "false", ""):
        l1_fusion = l1_fusion.lower()
    else:
        raise RuntimeError("Unsupported l1_fusion: %s" % l1_fusion)

    return l1_fusion


def te_set_version(soc_version, core_type="AiCore",
                   aicore_num=None, l1_fusion=None,
                   l2_mode="0", l2_fusion=None, kwargs=None):
    """set version info

    Parameters
    ----------
    soc_version : str
        "Ascend310"/"Ascend910"/"Ascend710"/"Ascend610" ...
    core_type : str
        "AiCore" or "VectorCore"
    aicore_num: int
        example: 32
    l1_fusion: bool
        example: True/False

    Returns
    -------
    errmsg : str
        error message, 'success' for OK.
    """
    l1_fusion = _check_and_get_l1_fusion(l1_fusion)
    if l1_fusion in ("true", True):
        set_L1_info("L1_fusion_enabled", True)
    elif l1_fusion in ("false", False):
        set_L1_info("L1_fusion_enabled", False)

    if l2_fusion in ("true", True):
        set_L1_info("L2_fusion_enabled", True)
    else:
        set_L1_info("L2_fusion_enabled", False)

    if core_type in (None, ""):
        core_type = "AiCore"
    _check_soc_version(soc_version, core_type)
    aicore_num = _check_and_get_aicore_num(soc_version,
                                           core_type,
                                           aicore_num)

    func = tvm.get_global_func("cce.product_init")
    value = func(soc_version, core_type, aicore_num, l1_fusion)
    if value != "success":
        raise RuntimeError("te_set_version() return error.")

    te_set_l2_mode(l2_mode)
    te_set_op_impl_mode(kwargs)
    te_set_tbe_debug_level(kwargs)

    cur_cce_product_params = CceProductParams()
    cur_cce_product_params.cce_product = \
        SOC_VERSION_MAP[soc_version][core_type]["ProductVersion"]
    return cur_cce_product_params


def te_set_op_impl_mode(kwargs):
    """
    set op impl mode
    """
    if not isinstance(kwargs, dict):
        return
    OpImplPolicy.op_impl_mode = kwargs.get('op_impl_mode', None)
    OpImplPolicy.op_impl_mode_list = kwargs.get('op_impl_mode_list', [])


def te_set_tbe_debug_level(kwargs):
    """
    set tbe_debug_level:
    0 disable
    1 enable tbe debug : pipe_all, python(tik)-cce mapping file)
    2 enable tbe debug : pipe_all, python(tik)-cce mapping file,
                         ccec compiler with "O0 - g"
    """
    if not isinstance(kwargs, dict):
        return
    op_debug_level_str = kwargs.get('op_debug_level', '0')
    if op_debug_level_str is (None or ""):
        return
    debug_level_strs = ('0', '1', '2')
    if op_debug_level_str not in debug_level_strs:
        raise RuntimeError("Unsupported op_debug_level: %s, it must be "
                           "one of ('0', '1', '2') and the data type"
                           " must be string " % op_debug_level_str)
    tbe_debug_level = int(op_debug_level_str)
    if tbe_debug_level == 0:
        return
    else:
        build_config_items = {"tbe_debug_level": tbe_debug_level,
                              "sync_mode": 0}
        cce.cce_build.build_config = cce.cce_build.build_config_update_list(
            cce.cce_build.build_config,
            build_config_items)
        return


def te_set_l2_mode(l2_mode):
    """set l2 flag

    Parameters
    ----------
    l2_flag : int

    Returns
    -------
    succ_flag : boolean
    """
    func = tvm.get_global_func("cce.set_L2_status")
    return func(int(l2_mode))



SOC_VERSION = "SOC_VERSION"
AICORE_TYPE = "AICORE_TYPE"
CORE_NUM = "CORE_NUM"
UB_SIZE = "UB_SIZE"
L2_SIZE = "L2_SIZE"
L1_SIZE = "L1_SIZE"
CUBE_SIZE = "CUBE_SIZE"
L0A_SIZE = "L0A_SIZE"
L0B_SIZE = "L0B_SIZE"
L0C_SIZE = "L0C_SIZE"
SMASK_SIZE = "SMASK_SIZE"
UNZIP = "UNZIP"


def get_soc_spec(key):
    """
    call global func to get soc spec

    Parameters
        ----------
        key : str
            key
    """
    support_key = (SOC_VERSION, AICORE_TYPE, CORE_NUM, UB_SIZE,
                   L2_SIZE, L1_SIZE, CUBE_SIZE, UNZIP,
                   L0A_SIZE, L0B_SIZE, L0C_SIZE, SMASK_SIZE)
    if key not in support_key:
        raise RuntimeError("Unsupported Key Value of get_soc_spec(): %s" % key)

    func = tvm.get_global_func("cce.get_soc_spec")
    value = func(key)
    if value == "":
        raise RuntimeError("Unsupported Key Value of get_soc_spec(): %s" % key)

    str2int_list = (CORE_NUM, UB_SIZE, L2_SIZE, L1_SIZE,
                    L0A_SIZE, L0B_SIZE, L0C_SIZE, SMASK_SIZE)
    if key in str2int_list:
        try:
            value = int(value)
        except Exception:
            raise RuntimeError("return value %s is not 'int' type" % value)
    elif key in (CUBE_SIZE, UNZIP):
        value_str_list = value.split(",")
        value_int_list = []
        for i in value_str_list:
            try:
                value_int_list.append(int(i))
            except Exception:
                raise RuntimeError("return value %s is not 'int' type" % value)
        value = value_int_list

    return value


@tvm.register_func("te.cce.get_product")
def get_product():
    """
    get product c++ code.

    Parameters
    ----------

    Returns
    -------
    value: device product.
        end of execution
    """

    return CceProductParams().cce_product


def api_check_support(intrinsic, dtype=""):
    """
    check if current chip support this api.

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    import te.lang.cce
    import te.lang.dynamic
    if not isinstance(intrinsic, str):
        raise RuntimeError("intrinsic type should be 'str', it is [%s]"
                           % type(intrinsic))
    if not isinstance(dtype, str):
        raise RuntimeError("dtype type should be 'str', it is [%s]"
                           % type(dtype))
    if intrinsic.startswith("tik."):
        return _deal_tik_api(intrinsic, dtype)

    if intrinsic.startswith("te.lang.cce."):
        return te.lang.cce.dsl_check_support(intrinsic, dtype)

    if intrinsic.startswith("te.lang.dynamic."):
        return te.lang.dynamic.dsl_check_support(intrinsic, dtype)

    return False


def intrinsic_check_support(intrinsic, dtype=""):
    """
    check if current chip support this intrinsic.

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    if not isinstance(intrinsic, str):
        raise RuntimeError("intrinsic type should be 'str', it is [%s]"
                           % type(intrinsic))
    if not intrinsic.startswith("Intrinsic_"):
        raise RuntimeError("intrinsic type should start with Intrinsic_")
    if not isinstance(dtype, str):
        raise RuntimeError("dtype type should be 'str', it is [%s]"
                           % type(dtype))
    func = tvm.get_global_func("cce.intrinsic_check_support")
    value = func(intrinsic, dtype)
    if value in (-1, ""):
        raise RuntimeError("Unsupported Key Value"
                           " of get_soc_spec(): %s" % intrinsic)
    if value == "True":
        return True
    return False


def _deal_tik_api(intrinsic, dtype):
    """
    deal tik api

    Parameters
    ----------
    intrinsic : str, the intrinsic need to check
    dtype: str, optional args, if not empty, will check the dtype.
    Returns
    -------
    value: bool, True if chip contains such api, else return False

    """
    tik_api_instr = intrinsic[4:]
    soc_total_version = _get_soc_name() + \
                        get_soc_spec("AICORE_TYPE")
    if tik_api_instr in ONLY_TIK_API_MAP:
        if soc_total_version in ONLY_TIK_API_MAP[tik_api_instr]:
            if dtype == "":
                return True
            return dtype in ONLY_TIK_API_MAP[tik_api_instr][soc_total_version]
        else:
            return False
    if tik_api_instr in TRANS_TIK_API_TO_INSTR_MAP:
        tik_api_instr = TRANS_TIK_API_TO_INSTR_MAP[tik_api_instr]
    if tik_api_instr.startswith("vec_"):
        tik_api_instr = "v" + tik_api_instr[4:]
    return intrinsic_check_support("Intrinsic_" + tik_api_instr, dtype)

def _get_soc_name():
    """
    get current soc's name for tik api
    :return: SOC_VERSION
    """
    temp_version = get_soc_spec("SOC_VERSION")
    if temp_version in (ASCEND_910H, ASCEND_910M, ASCEND_910P):
        return ASCEND_910
    return temp_version

_init_api("te.platform.cce_conf")
