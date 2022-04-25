#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
DRV Interface
"""
# Standard Packages
import ctypes
import logging
from typing import Tuple
from typing import Union
from typing import Optional

# Third-Party Packages
from ...utilities import get_loaded_so_path
from .dsmi_info import DSMI_FREQ_DEVICE_TYPE
from .dsmi_info import DSMI_ERROR_CODE
from .dsmi_info import DSMI_HEALTH_STATE
from .dsmi_info import DSMI_UTIL_DEVICE_TYPE
from .dsmi_structures import dsmi_chip_info_stru


class DSMIInterface:
    """
    DRV Function Wrappers
    """
    prof_online: dict = {}

    def __init__(self):
        self.dsmidll = ctypes.CDLL("libdrvdsmi_host.so")

    def print_so_path(self):
        """
        Print a debug message for libruntime.so path
        """
        # noinspection PyBroadException
        try:
            logging.debug("Using libdrvdsmi_host.so from %s" % get_loaded_so_path(self.dsmidll))
        except:
            logging.warning("Get libdrvdsmi_host.so path failed, apt install lsof or yum install lsof may solve this.")

    def get_device_count(self) -> int:
        device_count = (ctypes.c_int * 1)()
        self.dsmidll.dsmi_get_device_count.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_count(device_count)
        self._parse_error(error_code, "dsmi_get_device_count")
        return device_count[0]

    def list_logical_device_id(self) -> Tuple[int, ...]:
        device_count = self.get_device_count()
        device_ids = (ctypes.c_int * device_count)()
        self.dsmidll.dsmi_list_device.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_list_device(device_ids, ctypes.c_int(device_count))
        self._parse_error(error_code, "dsmi_list_device")
        return tuple(device_ids)

    def get_physical_id_from_logical_id(self, logical_id: int) -> int:
        device_logicid = ctypes.c_int(logical_id)
        device_phyid = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_phyid_from_logicid.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_phyid_from_logicid(device_logicid, device_phyid)
        self._parse_error(error_code, "dsmi_get_phyid_from_logicid")
        return device_phyid[0]

    def get_device_health_state(self, device_id: int) -> Union[DSMI_HEALTH_STATE, int]:
        device_id = ctypes.c_int(device_id)
        device_phealth = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_health.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_health(device_id, device_phealth)
        self._parse_error(error_code, "dsmi_get_device_health")
        try:
            return DSMI_HEALTH_STATE(device_phealth[0])
        except ValueError:
            return device_phealth[0]

    def get_device_error(self, device_id: int) -> Tuple[int, Tuple[int, ...]]:
        device_id = ctypes.c_int(device_id)
        device_errorcount = (ctypes.c_uint * 1)()
        device_perrorcode = (ctypes.c_uint * 128)()
        self.dsmidll.dsmi_get_device_errorcode.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_errorcode(device_id, device_errorcount, device_perrorcode)
        self._parse_error(error_code, "dsmi_get_device_errorcode")
        return device_errorcount[0], tuple(device_perrorcode)

    def get_device_error_description(self, device_id: int, error_code: int) -> bytes:
        device_id = ctypes.c_int(device_id)
        device_errorcode = ctypes.c_uint(error_code)
        device_perrorinfo = (ctypes.c_char * 256)()
        self.dsmidll.dsmi_query_errorstring.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_query_errorstring(device_id, device_errorcode, device_perrorinfo,
                                                         ctypes.c_int(256))
        self._parse_error(error_code, "dsmi_query_errorstring")
        return bytes(device_perrorinfo)

    def get_chip_info(self, device_id: int) -> dsmi_chip_info_stru:
        device_id = ctypes.c_int(device_id)
        result_struct = dsmi_chip_info_stru()
        self.dsmidll.dsmi_get_chip_info.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_chip_info(device_id, ctypes.c_void_p(ctypes.addressof(result_struct)))
        self._parse_error(error_code, "dsmi_get_chip_info")
        return result_struct

    def get_device_frequency(self, device_id: int, device_type: Union[int, DSMI_FREQ_DEVICE_TYPE]) -> Optional[int]:
        if isinstance(device_type, int):
            device_type = DSMI_FREQ_DEVICE_TYPE(device_type)
        device_id = ctypes.c_int(device_id)
        pfrequency = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_frequency.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_frequency(device_id, device_type.value, pfrequency)
        if error_code == DSMI_ERROR_CODE.DSMI_ERROR_NOT_SUPPORT.value:
            return None
        self._parse_error(error_code, "dsmi_get_device_frequency")
        return pfrequency[0]

    def get_device_temperature(self, device_id: int) -> Optional[int]:
        device_id = ctypes.c_int(device_id)
        ptemperature = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_temperature.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_temperature(device_id, ptemperature)
        if error_code == DSMI_ERROR_CODE.DSMI_ERROR_NOT_SUPPORT.value:
            return None
        self._parse_error(error_code, "dsmi_get_device_temperature")
        return ptemperature[0]

    def get_device_util(self, device_id: int, device_type: Union[int, DSMI_UTIL_DEVICE_TYPE]) -> Optional[int]:
        if isinstance(device_type, int):
            device_type = DSMI_UTIL_DEVICE_TYPE(device_type)
        device_id = ctypes.c_int(device_id)
        putil = (ctypes.c_uint * 1)()
        self.dsmidll.dsmi_get_device_utilization_rate.restype = ctypes.c_int
        error_code = self.dsmidll.dsmi_get_device_utilization_rate(device_id, device_type.value, putil)
        if error_code == DSMI_ERROR_CODE.DSMI_ERROR_NOT_SUPPORT.value:
            return None
        self._parse_error(error_code, "dsmi_get_device_utilization_rate")
        return putil[0]

    @staticmethod
    def _parse_error(error_code: int, function_name: str, allow_positive=False):
        if error_code != 0:
            if allow_positive and error_code > 0:
                logging.debug("DRV API Call %s() Success with return code %d" % (function_name, error_code))
            else:
                try:
                    raise RuntimeError(f"DSMI API Call {function_name} failed: {DSMI_ERROR_CODE(error_code).name}")
                except ValueError:
                    pass
                raise RuntimeError(f"DSMI API Call {function_name} failed with unknown code: {error_code}")
        else:
            logging.debug("DSMI API Call %s() Success" % function_name)
