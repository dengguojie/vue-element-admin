#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Structures used by DRV
"""
# Standard Packages
import ctypes

PROF_CHANNEL_NAME_LEN = 32
PROF_CHANNEL_NUM_MAX = 160


class prof_start_para_t(ctypes.Structure):
    """
    Profiling start para
    """
    _fields_ = [('channel_type', ctypes.c_uint),
                ('sample_period', ctypes.c_uint),
                ('real_time', ctypes.c_uint),
                ('user_data', ctypes.c_void_p),
                ('user_data_size', ctypes.c_uint)]


class prof_poll_info_t(ctypes.Structure):
    """
    Profiling pool info
    """
    _fields_ = [('device_id', ctypes.c_uint),
                ('channel_id', ctypes.c_uint)]


class aic_user_data_t(ctypes.Structure):
    """
    User data
    """
    _fields_ = [('type', ctypes.c_uint32),
                ('almost_full_threshold', ctypes.c_uint32),
                ('period', ctypes.c_uint32),
                ('core_mask', ctypes.c_uint32),
                ('event_num', ctypes.c_uint32),
                ('event', ctypes.c_uint32 * 8)]


class l2_user_data_t(ctypes.Structure):
    """
    User data
    """
    _fields_ = [('event_num', ctypes.c_uint32),
                ('event', ctypes.c_uint32 * 8)]


class channel_info_t(ctypes.Structure):
    """
    Channel info
    """
    # noinspection PyTypeChecker
    _fields_ = [('channel_name', ctypes.c_char * PROF_CHANNEL_NAME_LEN),
                ('channel_type', ctypes.c_uint),
                ('channel_id', ctypes.c_uint)]


class channel_list_t(ctypes.Structure):
    """
    Channel list
    """
    # noinspection PyTypeChecker
    _fields_ = [
        ('chip_type', ctypes.c_uint),
        ('channel_num', ctypes.c_uint),
        ('channel', channel_info_t * PROF_CHANNEL_NUM_MAX)
    ]


class poll_info(ctypes.Structure):
    """
    Pool info
    """
    _fields_ = [
        ('device_id', ctypes.c_uint),
        ('channel_id', ctypes.c_uint)
    ]


class aic_profiling_result(ctypes.Structure):
    """
    Profiling result structure
    """
    _fields_ = [
        ('type', ctypes.c_uint16),
        ('magic6bd3', ctypes.c_uint16),
        ('reserved0', ctypes.c_uint16),
        ('taskid', ctypes.c_uint16),
        ("reserved1", ctypes.c_uint16),
        ("reserved2", ctypes.c_uint16),
        ("total_cycle", ctypes.c_uint64),
        ("ov_cycle", ctypes.c_uint64),
        ("pmu_event0", ctypes.c_uint64),
        ("pmu_event1", ctypes.c_uint64),
        ("pmu_event2", ctypes.c_uint64),
        ("pmu_event3", ctypes.c_uint64),
        ("pmu_event4", ctypes.c_uint64),
        ("pmu_event5", ctypes.c_uint64),
        ("pmu_event6", ctypes.c_uint64),
        ("pmu_event7", ctypes.c_uint64),
        ("streamid", ctypes.c_uint32),
        ("reserved34", ctypes.c_uint64),
        ("reserved56", ctypes.c_uint64),
        ("reserved78", ctypes.c_uint64),
        ("reserved9x", ctypes.c_uint64),
    ]

    def getdict(self) -> dict:
        """
        Convert structure to dict
        :return:
        """
        return dict((f, getattr(self, f)) for f, _ in self._fields_)


class l2_profiling_result(ctypes.Structure):
    """
    Profiling result structure
    """
    _fields_ = [
        ('task_type', ctypes.c_uint16),   # 0
        ('stream_id', ctypes.c_uint16),   # 2
        ('task_id', ctypes.c_uint16),     # 4
        ('reserved', ctypes.c_uint16),    # 6
        ("pmu_event0", ctypes.c_uint64),  # 8
        ("pmu_event1", ctypes.c_uint64),  # 16
        ("pmu_event2", ctypes.c_uint64),  # 24
        ("pmu_event3", ctypes.c_uint64),  # 32
        ("pmu_event4", ctypes.c_uint64),  # 40
        ("pmu_event5", ctypes.c_uint64),  # 48
        ("pmu_event6", ctypes.c_uint64),  # 56
        ("pmu_event7", ctypes.c_uint64),  # 64
        # 72
    ]

    def getdict(self) -> dict:
        """
        Convert structure to dict
        :return:
        """
        return dict((f, getattr(self, f)) for f, _ in self._fields_)
