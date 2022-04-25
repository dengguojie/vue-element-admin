#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
DRV Interface
"""
# Standard Packages
import time
import ctypes
import concurrent.futures
import logging
import contextlib
from typing import Tuple
from typing import NoReturn

# Third-Party Packages
import tbetoolkits
from . import drv_structures
from .drv_info import PROF_CHANNEL_TYPE
from .drv_info import PROF_CHANNEL_ID
from .drv_info import PROF_TYPE
from .drv_info import tagDrvStatus
from .drv_info import AICEventID
from .drv_info import L2EventID
from ...utilities import PMU_MODE


class DRVInterface:
    """
    DRV Function Wrappers
    """

    def __init__(self):
        self.drvdll = ctypes.CDLL("libascend_hal.so")

    @contextlib.contextmanager
    def PMU(self, runtime_interface: "tbetoolkits.RTSInterface", mode, pmu_result, launch):
        if not launch:
            yield
        else:
            runtime_interface.start_profiler(runtime_interface.device_id)
            pmu_status = False
            l2_pmu_status = False
            l2_events = (L2EventID.V100_910_DHA_AICORE_REQ.value,
                         L2EventID.V100_910_DHA_AICORE_L2_HIT.value,
                         L2EventID.V100_910_DHA_AICORE_DIR_HIT.value,
                         L2EventID.V100_910_DHA_AICORE_VICTIM.value,
                         L2EventID.V100_910_AICORE_READ_REQ.value,
                         L2EventID.V100_910_AICORE_WRITE_REQ.value,
                         L2EventID.V100_910_AICORE_ALLOC_REQ.value,
                         L2EventID.V100_910_AICORE_READ_HIT_L2_FORWARD.value)
            l2_channel = PROF_CHANNEL_ID.CHANNEL_TSFW_L2
            channel = PROF_CHANNEL_ID.CHANNEL_AICORE
            if mode == PMU_MODE.ADVANCED:
                events = (AICEventID.BANKGROUP_STALL_CYCLES.value,
                          AICEventID.BANK_STALL_CYCLES.value,
                          AICEventID.VEC_RESC_CONFLICT_CYCLES.value,
                          AICEventID.MTE1_IQ_FULL_CYCLES.value,
                          AICEventID.MTE2_IQ_FULL_CYCLES.value,
                          AICEventID.MTE3_IQ_FULL_CYCLES.value,
                          AICEventID.CUBE_IQ_FULL_CYCLES.value,
                          AICEventID.VEC_IQ_FULL_CYCLES.value)
            elif mode == PMU_MODE.DEFAULT:
                events = (AICEventID.VEC_BUSY_CYCLES.value,
                          AICEventID.CUBE_BUSY_CYCLES.value,
                          AICEventID.SCALAR_BUSY_CYCLES.value,
                          AICEventID.MTE1_BUSY_CYCLES.value,
                          AICEventID.MTE2_BUSY_CYCLES.value,
                          AICEventID.MTE3_BUSY_CYCLES.value,
                          AICEventID.ICACHE_REQ.value,
                          AICEventID.ICACHE_MISS.value)
            else:
                raise NotImplementedError(f"Unknown PMU mode {mode}")
            # noinspection PyBroadException
            try:
                self.prof_start(runtime_interface.device_id, channel, events)
            except:
                logging.exception("PMU Start Failure")
                self.disable_pmu(runtime_interface, channel)
            else:
                logging.debug("PMU Start Success")
                pmu_status = True
            # noinspection PyBroadException
            try:
                self.prof_start(runtime_interface.device_id, l2_channel, l2_events)
            except:
                logging.exception("L2 PMU Start Failure")
            else:
                logging.debug("L2 PMU Start Success")
                l2_pmu_status = True
            yield
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                waiting_channel = []
                if l2_pmu_status:
                    waiting_channel.append(l2_channel)
                if pmu_status:
                    waiting_channel.append(channel)
                stopped_channel = []
                while waiting_channel:
                    poll_wrapper = executor.submit(self.prof_get_result)
                    result = poll_wrapper.result()
                    if result:
                        for _received_channel in result.copy():
                            if _received_channel == l2_channel:
                                channel_result = result[_received_channel]
                                pmu_result[9] = channel_result["pmu_event0"]
                                pmu_result[10] = channel_result["pmu_event1"]
                                pmu_result[11] = channel_result["pmu_event2"]
                                pmu_result[12] = channel_result["pmu_event3"]
                                pmu_result[13] = channel_result["pmu_event4"]
                                pmu_result[14] = channel_result["pmu_event5"]
                                pmu_result[15] = channel_result["pmu_event6"]
                                pmu_result[16] = channel_result["pmu_event7"]
                                waiting_channel.remove(_received_channel)
                            elif _received_channel == channel:
                                channel_result = result[_received_channel]
                                pmu_result[0] = channel_result["total_cycle"]
                                pmu_result[1] = channel_result["pmu_event0"]
                                pmu_result[2] = channel_result["pmu_event1"]
                                pmu_result[3] = channel_result["pmu_event2"]
                                pmu_result[4] = channel_result["pmu_event3"]
                                pmu_result[5] = channel_result["pmu_event4"]
                                pmu_result[6] = channel_result["pmu_event5"]
                                pmu_result[7] = channel_result["pmu_event6"]
                                pmu_result[8] = channel_result["pmu_event7"]
                                waiting_channel.remove(_received_channel)
                                runtime_interface.stop_profiler(runtime_interface.device_id)
                            else:
                                raise NotImplementedError("PMU Currently support aicore and l2 channel only!")
                            if _received_channel not in stopped_channel:
                                self.prof_stop(runtime_interface.device_id, _received_channel)
                    else:
                        executor.submit(self.prof_stop, runtime_interface.device_id, waiting_channel[-1])
                        stopped_channel.append(waiting_channel[-1])

    def prof_start(self,
                   device_id: int,
                   channel_id: PROF_CHANNEL_ID,
                   register_settings: Tuple[int, int, int, int, int, int, int, int],
                   prof_channel_type: PROF_CHANNEL_TYPE = PROF_CHANNEL_TYPE.PROF_TS_TYPE,
                   prof_type: PROF_TYPE = PROF_TYPE.PROF_TASK_BASED,
                   coremask: int = 0xFFFFFFFF) -> NoReturn:
        """
        Trigger ts or peripheral devices to start preparing for sampling profile information
        :param device_id: device ID
        :param channel_id:  Channel ID(CHANNEL_TSCPU--(CHANNEL_TSCPU_MAX - 1))
        :param prof_channel_type: prof_tscpu_start or prof_peripheral_start
        :param prof_type: task base or sample base
        :param register_settings: PMU Register settings
        :param coremask
        :return:
        """
        start_time = time.time()
        device_id = ctypes.c_uint(device_id)
        c_channel_id = ctypes.c_uint(channel_id.value)
        channel_type = prof_channel_type.value
        prof_type = prof_type.value
        if channel_type != 0:
            raise NotImplementedError("Currently support TS_TYPE Profiling only!!!")
        if prof_type != 0:
            raise NotImplementedError("Currently support Task Based Profiling only!!!")
        if len(register_settings) != 8:
            raise IndexError("Register settings must have 8 values")
        if channel_id == PROF_CHANNEL_ID.CHANNEL_AICORE:
            user_data = drv_structures.aic_user_data_t(
                type=ctypes.c_uint32(prof_type),
                almost_full_threshold=ctypes.c_uint32(0xFF),
                period=ctypes.c_uint32(0x0),
                core_mask=ctypes.c_uint32(coremask),
                event_num=ctypes.c_uint32(8),
                event=(ctypes.c_uint32 * 8)(*register_settings)
            )
        elif channel_id == PROF_CHANNEL_ID.CHANNEL_TSFW_L2:
            user_data = drv_structures.l2_user_data_t(
                event_num=ctypes.c_uint32(8),
                event=(ctypes.c_uint32 * 8)(*register_settings)
            )
        else:
            raise NotImplementedError("Currently support L2 and AICORE CHANNEL Profiling only!!!")
        prof_start_para = drv_structures.prof_start_para_t(
            channel_type=ctypes.c_uint(channel_type),
            sample_period=ctypes.c_uint(0x0),
            real_time=ctypes.c_uint(1),
            user_data=ctypes.c_void_p(ctypes.addressof(user_data)),
            user_data_size=ctypes.c_uint(ctypes.sizeof(user_data)))
        self.drvdll.prof_drv_start.restype = ctypes.c_int
        return_value = self.drvdll.prof_drv_start(device_id, c_channel_id,
                                                  ctypes.c_void_p(ctypes.addressof(prof_start_para)))
        self._parse_error(return_value, "prof_drv_start",
                          extra_info=f"costs {round(time.time() - start_time, 3)} seconds")

    def prof_stop(self,
                  device_id: int,
                  channel_id: PROF_CHANNEL_ID) -> NoReturn:
        """
        Trigger ts or peripheral devices to start preparing for sampling profile information
        :param device_id: device ID
        :param channel_id:  Channel ID(CHANNEL_TSCPU--(CHANNEL_TSCPU_MAX - 1))
        :return:
        """
        start_time = time.time()
        device_id = ctypes.c_uint(device_id)
        channel_id = ctypes.c_uint(channel_id.value)
        self.drvdll.prof_stop.restype = ctypes.c_int
        return_value = self.drvdll.prof_stop(device_id, channel_id)
        self._parse_error(return_value, "prof_stop", extra_info=f"costs {round(time.time() - start_time, 3)} seconds")

    # noinspection PyCallingNonCallable
    def prof_poll(self,
                  channel_num: int,
                  timeout: int = 1) -> tuple:
        """
        Querying valid channel information
        :param channel_num: num of channel
        :param timeout: timeout in seconds
        :return: positive number for channels Number
        """
        start_time = time.time()
        # noinspection PyTypeChecker
        out_buf = (drv_structures.prof_poll_info_t * channel_num)()
        self.drvdll.prof_channel_poll.restype = ctypes.c_int
        return_value = self.drvdll.prof_channel_poll(out_buf,
                                                     ctypes.c_int32(channel_num),
                                                     ctypes.c_int32(timeout))
        self._parse_error(return_value, "prof_poll", True, f"costs {round(time.time() - start_time, 3)} seconds")
        return return_value, out_buf

    def prof_read(self,
                  device_id: int,
                  channel_id: PROF_CHANNEL_ID,
                  buffer_size: int = 512) -> dict:
        """
        Read and collect profile information
        :param device_id: Device ID
        :param channel_id: channel ID(1--(CHANNEL_NUM - 1))
        :param buffer_size: length of the profile to be read
        :return: positive number for readable buffer length
        :return: buffer
        """
        start_time = time.time()
        # noinspection PyCallingNonCallable,PyTypeChecker
        if channel_id == PROF_CHANNEL_ID.CHANNEL_AICORE:
            buffer = drv_structures.aic_profiling_result()
        elif channel_id == PROF_CHANNEL_ID.CHANNEL_TSFW_L2:
            buffer = drv_structures.l2_profiling_result()
        else:
            raise NotImplementedError("Currently support L2 and AICORE CHANNEL Profiling only!!!")
        self.drvdll.prof_channel_read.restype = ctypes.c_int
        return_value = self.drvdll.prof_channel_read(ctypes.c_uint(device_id),
                                                     ctypes.c_int(channel_id.value),
                                                     ctypes.c_void_p(ctypes.addressof(buffer)),
                                                     ctypes.c_uint(buffer_size))
        self._parse_error(return_value, "prof_read", True, f"costs {round(time.time() - start_time, 3)} seconds")
        return buffer.getdict()

    def prof_get_result(self) -> dict:
        ready_num, ready_structure = self.prof_poll(5)
        results = {}
        for i in range(ready_num):
            dev = ready_structure[i].device_id
            channel = PROF_CHANNEL_ID(ready_structure[i].channel_id)
            results[channel] = self.prof_read(dev, channel)
        logging.debug(f"PMU Get Result received {results}")
        return results

    def prof_get_channels(self, device_id: int) -> dict:
        """
        Trigger to get enable channels
        :param device_id: device ID
        :return:
        """
        self.drvdll.prof_drv_get_channels.restype = ctypes.c_int
        channel_list = drv_structures.channel_list_t()
        return_value = self.drvdll.prof_drv_get_channels(ctypes.c_uint(device_id),
                                                         ctypes.c_void_p(ctypes.addressof(channel_list)))
        self._parse_error(return_value, "prof_drv_get_channels")
        return {"chip_type": channel_list.chip_type,
                "channel_num": channel_list.channel_num,
                "channel": tuple({"channel_name": channel.channel_name,
                                  "channel_type": channel.channel_type,
                                  "channel_id": channel.channel_id} for channel in channel_list.channel)}

    def prof_flush(self,
                   device_id: int,
                   channel_id: PROF_CHANNEL_ID):
        data_len = (ctypes.c_uint32 * 1)()
        self.drvdll.halProfDataFlush.restype = ctypes.c_int
        return_value = self.drvdll.halProfDataFlush(ctypes.c_uint(device_id),
                                                    ctypes.c_int(channel_id.value),
                                                    data_len)
        self._parse_error(return_value, "halProfDataFlush", True)
        return int(data_len[0])

    def get_device_status(self, device_id: int) -> tagDrvStatus:
        """
        Get device status
        :param device_id:
        :return:
        """
        result = ctypes.c_uint()
        self.drvdll.drvDeviceStatus.restype = ctypes.c_int
        return_value = self.drvdll.drvDeviceStatus(ctypes.c_uint32(device_id),
                                                   ctypes.c_void_p(ctypes.addressof(result)))
        self._parse_error(return_value, "drvDeviceStatus")
        return tagDrvStatus(result.value)

    def disable_pmu(self,
                    runtime_interface: "tbetoolkits.core_modules.runtime.RTSInterface",
                    channel_id: PROF_CHANNEL_ID):
        """Disable PMU"""
        # noinspection PyBroadException
        try:
            self.prof_stop(runtime_interface.device_id,
                           channel_id=channel_id)
        except:
            logging.exception(f"PMU Stop Failure for {channel_id.name}")
        # noinspection PyBroadException
        try:
            runtime_interface.stop_profiler(runtime_interface.device_id)
        except:
            logging.exception(f"RTS PMU Profiler Stop Failure for {channel_id.name}")

    @staticmethod
    def _parse_error(error_code: int, function_name: str, allow_positive=False, extra_info=""):
        if error_code != 0:
            if allow_positive and error_code > 0:
                logging.debug(f"DRV API Call {function_name}() Success with return code {error_code}, {extra_info}")
            else:
                raise RuntimeError(f"DRV API Call {function_name} Failed with return code {error_code}, {extra_info}")
        else:
            logging.debug(f"DRV API Call {function_name}() Success, {extra_info}")
