#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Structures used by RTS
"""
# Standard Packages
import ctypes

# Third-Party Packages
from . import rts_info


# For ctypes interpretation and construction of rtDevBinary_t from pointer
class rtDevBinary_t(ctypes.Structure):
    """
    Device Binary structure
    """
    _fields_ = [('magic', ctypes.c_uint32),
                ('version', ctypes.c_uint32),
                ('data', ctypes.c_char_p),
                ('length', ctypes.c_uint64)]


# For online profiling
class rtProfDataInfo_t(ctypes.Structure):
    """
    Profiling result info structure
    """
    _fields_ = [('stubfunc', ctypes.c_void_p),
                ('blockDim', ctypes.c_uint32),
                ('args', ctypes.c_void_p),
                ('argsSize', ctypes.c_uint32),
                ('smDesc', ctypes.c_void_p),
                ('stream', ctypes.c_uint64),
                ('totalcycle', ctypes.c_uint64),
                ('ovcycle', ctypes.c_uint64),
                ('pmu_cnt', ctypes.c_uint64 * rts_info.ONLINE_PROF_MAX_PMU_NUM)]


# For profiler
class rtCommandHandleParams(ctypes.Structure):
    """
    Profiling switch structure
    """
    _fields_ = [('pathLen', ctypes.c_uint32),
                ('storageLimit', ctypes.c_uint32),
                ('profDataLen', ctypes.c_uint32),
                ('path', ctypes.c_char * (rts_info.RT_PROF_PATH_LEN_MAX + 1)),
                ('profData', ctypes.c_char * (rts_info.RT_PROF_PARAM_LEN_MAX + 1))]


class rtProfCommandHandle_t(ctypes.Structure):
    """
    Profiling switch structure
    """
    _fields_ = [('profSwitch', ctypes.c_uint64),
                ('profSwitchHi', ctypes.c_uint64),
                ('devNums', ctypes.c_uint32),
                ('devIdList', ctypes.c_uint32 * rts_info.RT_PROF_MAX_DEV_NUM),
                ('modelId', ctypes.c_uint32),
                ('type', ctypes.c_uint32),
                ('commandHandleParams', rtCommandHandleParams)]


# For device info
class rtDeviceInfo_t(ctypes.Structure):
    """
    Device info structure
    """
    _fields_ = [('env_type', ctypes.c_uint8),
                ('ctrl_cpu_ip', ctypes.c_uint32),
                ("ctrl_cpu_core_num", ctypes.c_uint32),
                ("ctrl_cpu_endian_little", ctypes.c_uint32),
                ("ts_cpu_core_num", ctypes.c_uint32),
                ("ai_cpu_core_num", ctypes.c_uint32),
                ("ai_core_num", ctypes.c_uint32),
                ("ai_core_freq", ctypes.c_uint32),
                ("ai_cpu_core_id", ctypes.c_uint32),
                ("ai_core_id", ctypes.c_uint32),
                ("aicpu_occupy_bitmap", ctypes.c_uint32),
                ("hardware_version", ctypes.c_uint32),
                ("ts_num", ctypes.c_uint32)]


# For platform info
class rtPlatformInfo_t(ctypes.Structure):
    """
    Platform info structure
    """
    _fields_ = [('platformConfig', ctypes.c_uint32)]


# For AiCore Spec
class rtAiCoreSpec_t(ctypes.Structure):
    """
    AiCore Spec structure
    """
    _fields_ = [('cubeFreq', ctypes.c_uint32),
                ('cubeMSize', ctypes.c_uint32),
                ('cubeKSize', ctypes.c_uint32),
                ('cubeNSize', ctypes.c_uint32),
                ('cubeFracMKNFp16', ctypes.c_uint64),
                ('cubeFracMKNInt8', ctypes.c_uint64),
                ('vecFracVmulMKNFp16', ctypes.c_uint64),
                ('vecFracVmulMKNInt8', ctypes.c_uint64),
                ]
