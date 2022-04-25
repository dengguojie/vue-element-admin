#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
RTS Interface
"""
# Standard Packages
import math
import time
import ctypes
import logging
from typing import Any
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NoReturn

# Third-Party Packages
import numpy
from . import rts_info
from . import rts_structures
from ...utilities import read_file
from ...utilities import get_loaded_so_path


# noinspection PyPep8Naming,PyUnusedLocal
@ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
def MsprofReporterCallbackPlaceholder(_moduleId, _type, _data, _len=None):
    """
    This is just a placeholder for MsprofReporterCallback
    :return:
    """
    return 0


class RTSInterface:
    """
    RTS Function Wrappers
    """

    def __init__(self, camodel=False, rts_custom_path=""):
        if camodel:
            self.rtsdll = ctypes.CDLL(f"{rts_custom_path}libruntime_camodel.so")
        else:
            self.rtsdll = ctypes.CDLL(f"{rts_custom_path}libruntime.so")
        self.device_id = None
        self.context = None
        self.camodel = camodel
        # Data storage
        self.kernel_binary_storage = {}
        self.kernel_name_storage = {}
        self.context_storage = []
        self.memory_manager = {}
        self.online_profiling_status: bool = False

    def print_so_path(self):
        """
        Print a debug message for libruntime.so path
        """
        # noinspection PyBroadException
        try:
            logging.debug("Using libruntime.so from %s" % get_loaded_so_path(self.rtsdll))
        except:
            logging.debug("Using libruntime.so from UNKNOWN, install lsof may solve this.")

    def set_device(self, device_id: int, retry=0) -> None:
        """
        Set device_id for current thread

        ***Warning***: Although thread_id is an instance variable, it is actually being assigned
                       to current thread. Please do not maintain multiple Interface with different
                       thread_id in one thread.
        :param retry:
        :param device_id: thread_id you want to switch to
        :return: None
        """
        start_time = time.time()
        self.rtsdll.rtSetDevice.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtSetDevice(device_id)
        # noinspection PyBroadException
        self.parse_error(rt_error, "rtSetDevice", f"on device {device_id} "
                                                  f"costs {round(time.time() - start_time, 3)} seconds")
        self.device_id = device_id

    def set_device_without_tsd(self, device_id: int, retry=0) -> None:
        """
        Set device_id for current thread without tsd

        ***Warning***: Although thread_id is an instance variable, it is actually being assigned
                       to current thread. Please do not maintain multiple Interface with different
                       thread_id in one thread.
        :param retry:
        :param device_id: thread_id you want to switch to
        :return: None
        """
        start_time = time.time()
        self.rtsdll.rtSetDeviceWithoutTsd.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtSetDeviceWithoutTsd(device_id)
        # noinspection PyBroadException
        try:
            self.parse_error(rt_error, "rtSetDeviceWithoutTsd", f"costs {round(time.time() - start_time, 3)} seconds")
        except:
            if retry < 3:
                logging.error(f"rtSetDeviceWithoutTsd Failed, wait 1 second and retry, count: {retry}")
                time.sleep(1)
                return self.set_device_without_tsd(device_id, retry + 1)
            else:
                raise RuntimeError("rtSetDeviceWithoutTsd for device %d failed for 4 times, check your davinci device."
                                   % device_id) from None
        self.device_id = device_id

    def get_device_info(self, device_id: int, module_type: str, info_type: rts_info.RTS_INFO_TYPE):
        """
        Get device info by device_id and corresponding module_type and info_type
        :param device_id:
        :param module_type:
        :param info_type:
        :return:
        """
        start_time = time.time()
        c_info = (ctypes.c_int64 * 8)()
        module_type = rts_info.rt_module_type[module_type]
        self.rtsdll.rtGetDeviceInfo.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtGetDeviceInfo(ctypes.c_uint32(device_id),
                                               ctypes.c_int32(module_type),
                                               ctypes.c_int32(info_type.value),
                                               c_info)
        self.parse_error(rt_error, "rtGetDeviceInfo", f"costs {round(time.time() - start_time, 3)} seconds")
        result = int(c_info[0])
        if info_type == rts_info.RTS_INFO_TYPE.INFO_TYPE_ENV:
            info_dict = {
                0: "FPGA",
                1: "EMU",
                2: "ESL",
                3: "ASIC"
            }
            if result in info_dict:
                return info_dict[result]
        return result

    def get_aicore_count(self) -> int:
        """
        Get Number of AI core
        :return:
        """
        start_time = time.time()
        c_uint32_t = (ctypes.c_uint32 * 8)()
        self.rtsdll.rtGetAiCoreCount.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtGetAiCoreCount(c_uint32_t)
        self.parse_error(rt_error, "rtGetAiCoreCount", f"costs {round(time.time() - start_time, 3)} seconds")
        return int(c_uint32_t[0])

    def get_device_ids(self, device_count: int) -> tuple:
        """
        Get IDs of all devices
        :param device_count: Number of device
        :return:
        """
        start_time = time.time()
        c_uint32_t = (ctypes.c_uint32 * device_count)(*[-1 for _ in range(device_count)])
        self.rtsdll.rtGetDeviceIDs.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtGetDeviceIDs(c_uint32_t, device_count)
        self.parse_error(rt_error, "rtGetDeviceIDs", f"costs {round(time.time() - start_time, 3)} seconds")
        return tuple(c_uint32_t)

    def get_device_count(self) -> int:
        """
        Get Number of Device
        :return:
        """
        start_time = time.time()
        self.print_so_path()
        c_uint32_t = (ctypes.c_uint32 * 8)()
        rt_error = self.rtsdll.rtGetDeviceCount(c_uint32_t)
        self.parse_error(rt_error, "rtGetAiCoreCount", f"costs {round(time.time() - start_time, 3)} seconds")
        return int(c_uint32_t[0])

    def get_soc_version(self, max_len: int = 256) -> str:
        """
        Get chipType
        :return:
        """
        start_time = time.time()
        c_char_t = ctypes.create_string_buffer(b'\xff' * max_len, max_len)
        self.rtsdll.rtGetSocVersion.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(max_len))
        self.parse_error(rt_error, "rtGetSocVersion", f"costs {round(time.time() - start_time, 3)} seconds")
        return c_char_t.value.decode("utf-8")

    def get_platform_config(self) -> Tuple[rts_info.RTS_ARCHITECTURE_TYPE,
                                           rts_info.RTS_CHIP_TYPE,
                                           rts_info.RTS_PLATFORM_TYPE]:
        """
        Get Device Platform
        :return:
        """
        raise NotImplementedError("This API has been removed from runtime library")
        # c_platform_config_struct = rts_structures.rtPlatformInfo_t()
        # c_platform_config_p = ctypes.c_void_p(ctypes.addressof(c_platform_config_struct))
        # self.rtsdll.rtGetPlatformConfig.restype = ctypes.c_uint64
        # rt_error = self.rtsdll.rtGetPlatformConfig(c_platform_config_p)
        # self.parse_error(rt_error, "rtGetPlatformConfig", f"costs {round(time.time() - start_time, 3)} seconds")
        # _result = int(c_platform_config_struct.platformConfig)
        # arch = rts_info.RTS_ARCHITECTURE_TYPE(_result >> 16 & 0xFFFF)
        # chip = rts_info.RTS_CHIP_TYPE(_result >> 8 & 0xFF)
        # plat = rts_info.RTS_PLATFORM_TYPE(_result & 0xFF)
        # return arch, chip, plat

    def create_context(self, context_mode: str) -> ctypes.c_void_p:
        """
        Create a new context and bind it with current thread

        :param context_mode: Check runtime.rts_info for available context mode
        :return: context pointer
        """
        start_time = time.time()
        c_context = ctypes.c_void_p()
        c_context_p = ctypes.c_void_p(ctypes.addressof(c_context))
        self.rtsdll.rtCtxCreate.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtCtxCreate(c_context_p,
                                           ctypes.c_uint32(rts_info.rt_context_mode[context_mode]),
                                           ctypes.c_int32(self.device_id))
        self.parse_error(rt_error, "rtCtxCreate", f"costs {round(time.time() - start_time, 3)} seconds")
        self.context = c_context
        self.context_storage.append(c_context)
        return c_context

    def destroy_context(self, c_context: ctypes.c_void_p = None) -> None:
        """
        Destroy used context
        :param c_context: context pointer
        :return:
        """
        start_time = time.time()
        self.rtsdll.rtCtxDestroy.restype = ctypes.c_uint64
        if c_context is None:
            if self.context not in self.context_storage:
                raise ValueError("Input context does not exist in current interface's context storage")
            rt_error = self.rtsdll.rtCtxDestroy(self.context)
        else:
            if c_context not in self.context_storage:
                raise ValueError("Input context does not exist in current interface's context storage")
            rt_error = self.rtsdll.rtCtxDestroy(c_context)
        self.parse_error(rt_error, "rtCtxDestroy", f"costs {round(time.time() - start_time, 3)} seconds")
        self.context = None

    def set_context(self, c_context):
        """
        Bind a context to current thread
        :param c_context: context pointer
        :return:
        """
        start_time = time.time()
        if c_context not in self.context_storage:
            raise ValueError("Input context does not exist in current interface's context storage")
        self.rtsdll.rtCtxSetCurrent.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtCtxSetCurrent(c_context)
        self.parse_error(rt_error, "rtCtxSetCurrent", f"costs {round(time.time() - start_time, 3)} seconds")
        self.context = c_context

    def create_stream(self, priority=0) -> ctypes.c_void_p:
        """
        Create a new stream on current thread
        :param priority: Default priority at 0
        :return: c_stream: a void* representing the stream
        """
        start_time = time.time()
        c_stream = ctypes.c_void_p()
        c_stream_p = ctypes.c_void_p(ctypes.addressof(c_stream))
        self.rtsdll.rtStreamCreate.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtStreamCreate(c_stream_p, priority)
        self.parse_error(rt_error, "rtStreamCreate", f"costs {round(time.time() - start_time, 3)} seconds")
        return c_stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        """
        Destroy stream
        :param stream: void* of the stream you want to destroy
        :return: None
        """
        start_time = time.time()
        self.rtsdll.rtStreamDestroy.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtStreamDestroy(stream)
        self.parse_error(rt_error, "rtStreamDestroy", f"costs {round(time.time() - start_time, 3)} seconds")

    def register_device_binary_kernel(self, kernel_path: str, core_type: str) -> ctypes.c_void_p:
        """
        Register device kernel on current thread
        :param core_type:
        :param kernel_path: path to the device kernel binary
        :return: rts_binary_handle: a void* representing the kernel
        """
        start_time = time.time()
        if core_type == "AiCore":
            magic = rts_info.RT_DEV_BINARY_MAGIC_ELF
        elif core_type == "VectorCore":
            magic = rts_info.RT_DEV_BINARY_MAGIC_ELF_AIVEC
        elif core_type == "CubeCore":
            magic = rts_info.RT_DEV_BINARY_MAGIC_ELF_AICUBE
        elif core_type == "AiCPU":
            magic = rts_info.RT_DEV_BINARY_MAGIC_ELF_AICPU
        else:
            raise RuntimeError("Unknown core_type: %s" % core_type)
        # Read kernel
        kernel = read_file(kernel_path)
        kernel_size = len(kernel)
        logging.debug("Read %d bytes from kernel" % kernel_size)
        # Check kernel size
        if kernel_size <= 0:
            raise IOError("RTS Interface received kernel of invalid size %d with path %s" % (kernel_size, kernel_path))
        c_kernel_p = ctypes.c_char_p(kernel)
        # Construct device binary structure
        rts_device_binary = rts_structures.rtDevBinary_t(data=c_kernel_p,
                                                         length=ctypes.c_uint64(len(kernel)),
                                                         version=ctypes.c_uint32(0),
                                                         magic=ctypes.c_uint32(magic))
        # Prepare result structure
        rts_binary_handle = ctypes.c_void_p()
        self.rtsdll.rtDevBinaryRegister.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtDevBinaryRegister(
            ctypes.c_void_p(ctypes.addressof(rts_device_binary)),
            ctypes.c_void_p(ctypes.addressof(rts_binary_handle)))
        self.parse_error(rt_error, "rtDevBinaryRegister", f"costs {round(time.time() - start_time, 3)} seconds")
        self.kernel_binary_storage[rts_binary_handle.value] = kernel
        return rts_binary_handle

    def unregister_device_binary_kernel(self, rts_binary_handle: ctypes.c_void_p):
        """
        Unregister device kernel
        :param rts_binary_handle: pointer to device binary
        :return:
        """
        start_time = time.time()
        self.rtsdll.rtDevBinaryUnRegister.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtDevBinaryUnRegister(rts_binary_handle)
        self.parse_error(rt_error, "rtDevBinaryUnRegister", f"costs {round(time.time() - start_time, 3)} seconds")
        del self.kernel_binary_storage[rts_binary_handle.value]
        if rts_binary_handle.value in self.kernel_name_storage:
            self.kernel_name_storage[rts_binary_handle.value].clear()
            del self.kernel_name_storage[rts_binary_handle.value]

    def register_function(self, rts_binary_handle: ctypes.c_void_p, kernel_name: str,
                          func_mode: int) -> ctypes.c_void_p:
        """
        Register function in device kernel on current hread
        :param rts_binary_handle: pointer to device binary
        :param kernel_name: function name
        :param func_mode: function mode
        :return:
        """
        start_time = time.time()
        if rts_binary_handle.value not in self.kernel_name_storage:
            self.kernel_name_storage[rts_binary_handle.value] = []
        kernel_name_bytes = kernel_name.encode("UTF-8")
        c_kernel_name_p = ctypes.c_char_p(kernel_name_bytes)
        c_func_mode = ctypes.c_uint32(func_mode)
        self.rtsdll.rtFunctionRegister.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtFunctionRegister(rts_binary_handle,
                                                  c_kernel_name_p,
                                                  c_kernel_name_p,
                                                  c_kernel_name_p,
                                                  c_func_mode)
        self.parse_error(rt_error, "rtFunctionRegister", f"costs {round(time.time() - start_time, 3)} seconds")
        self.kernel_name_storage[rts_binary_handle.value].append(kernel_name_bytes)
        return ctypes.cast(c_kernel_name_p, ctypes.c_void_p)

    def copy_bin_file_to_hbm(self, bin_path: str) -> ctypes.c_void_p:
        """
        Copy data of binary file into device hbm
        :param bin_path: path to the binary
        :return:
        """
        data = read_file(bin_path, 32212254720)
        return self.copy_bin_to_hbm(data)

    def copy_bin_to_hbm(self, _bin: bytes) -> ctypes.c_void_p:
        """
        Copy raw binaries into device hbm
        :param _bin:
        :return:
        """
        if not isinstance(_bin, bytes):
            raise TypeError("Copy binary to hbm supports bytes only, received %s" % str(type(_bin)))
        c_memory_p = self.malloc(int(math.ceil(len(_bin) / 32) * 32 + 32),
                                 rts_info.RTS_MEMORY_TYPE.RT_MEMORY_HBM,
                                 "RT_MEMORY_POLICY_HUGE_PAGE_ONLY" if len(_bin) > 2048 else "RT_MEMORY_POLICY_NONE")
        self.memcpy(c_memory_p, int(math.ceil(len(_bin) / 32) * 32 + 32),
                    _bin, len(_bin),
                    "RT_MEMCPY_HOST_TO_DEVICE")
        return c_memory_p

    def copy_hbm_to_hbm(self, c_memory_p: Union[ctypes.c_void_p, int], length: int) -> ctypes.c_void_p:
        """
        Duplicate data in hbm to hbm
        :param c_memory_p:
        :param length: in bytes
        :return:
        """
        if not isinstance(c_memory_p, ctypes.c_void_p):
            c_memory_p = ctypes.c_void_p(c_memory_p)
        try:
            c_memory_target_p = self.malloc(int(length),
                                            rts_info.RTS_MEMORY_TYPE.RT_MEMORY_HBM,
                                            "RT_MEMORY_POLICY_HUGE_PAGE_ONLY" if int(length) > (1024 * 1024)
                                            else "RT_MEMORY_POLICY_NONE")
        except:
            logging.error("rtMalloc on HBM failed, HBM memory info: %s"
                          % (str(self.get_memory_info_ex(rts_info.RTS_MEMORY_INFO_TYPE.RT_MEMORYINFO_HBM))))
            raise
        self.memcpy(c_memory_target_p, int(length),
                    c_memory_p, length,
                    "RT_MEMCPY_DEVICE_TO_DEVICE")
        return c_memory_target_p

    def get_data_from_hbm(self,
                          c_memory_p: Union[ctypes.c_void_p, int],
                          data_size: int):
        """
        :param c_memory_p: a void* which points to the hbm address you want to access
        :param data_size: data size in bytes
        :return: bytes
        """
        if not isinstance(c_memory_p, ctypes.c_void_p):
            c_memory_p = ctypes.c_void_p(c_memory_p)
        # noinspection PyTypeChecker
        c_buffer_type: Any = ctypes.c_char * data_size
        c_buffer: Any = c_buffer_type()
        self.memcpy(c_buffer,
                    data_size, c_memory_p, data_size,
                    "RT_MEMCPY_DEVICE_TO_HOST")
        return c_buffer

    def memcpy(self,
               c_memory_p: ctypes.c_void_p,
               memory_size: int,
               data: Union[bytes, ctypes.c_void_p],
               data_size: int,
               memcpy_kind: str = "RT_MEMCPY_HOST_TO_HOST") -> None:
        """
        RTS memcpy interface
        :param c_memory_p:
        :param memory_size:
        :param data:
        :param data_size:
        :param memcpy_kind:
        :return:
        """
        start_time = time.time()
        if memory_size <= 0:
            logging.warning("rtMemcpy() called with negative or zero memory size, aligned to 1!")
            memory_size = 1
        if data_size <= 0:
            logging.warning("rtMemcpy() called with negative or zero data size, aligned to 1!")
            data_size = 1
        if isinstance(data, bytes) and len(data) < data_size:
            logging.warning("rtMemcpy() called with insufficient data, filled with zero!")
            data = numpy.zeros((data_size,), dtype="uint8").tobytes()
        if isinstance(data, bytes):
            c_data_p = ctypes.c_char_p(data)
        elif isinstance(data, ctypes.c_void_p):
            c_data_p = data
        else:
            raise TypeError("Runtime function memcpy supports bytes or c_voidp only!")
        c_data_size = ctypes.c_uint64(data_size)
        c_memory_size = ctypes.c_uint64(memory_size)
        self.rtsdll.rtMemcpy.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtMemcpy(c_memory_p, c_memory_size,
                                        c_data_p, c_data_size,
                                        rts_info.rt_memcpy_kind[memcpy_kind])
        self.parse_error(rt_error, "rtMemcpy", f"costs {round(time.time() - start_time, 3)} seconds")

    def memset(self,
               c_memory_p: ctypes.c_void_p, memory_size: int,
               data: int, count: int):
        """
        Set memory value with uint32_t value
        :param c_memory_p: a void* to the memory
        :param memory_size: size of the memory
        :param data: uint32_t value used to fill the memory
        :param count: number of values you want to fill
        :return: None
        """
        start_time = time.time()
        if 0xFFFFFFFF < data < 0:
            raise RuntimeError("Invalid memset value, out of uint32_t range: %d" % data)
        if memory_size <= 0:
            logging.warning("rtMemset() called with negative or zero size, aligned to 1!")
            memory_size = 1
        if count <= 0:
            logging.warning("rtMemset() called with negative or data count, aligned to 1!")
            count = 1
        c_data_size = ctypes.c_uint64(count)
        c_data = ctypes.c_uint32(data)
        c_memory_size = ctypes.c_uint64(memory_size)
        self.rtsdll.rtMemset.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtMemset(c_memory_p, c_memory_size,
                                        c_data, c_data_size)
        self.parse_error(rt_error, "rtMemset", f"costs {round(time.time() - start_time, 3)} seconds")

    def malloc(self,
               memory_size: int,
               memory_type: rts_info.RTS_MEMORY_TYPE = rts_info.RTS_MEMORY_TYPE.RT_MEMORY_DEFAULT,
               memory_policy: str = "RT_MEMORY_POLICY_NONE") -> ctypes.c_void_p:
        """
        RTS malloc interface
        :param memory_size:
        :param memory_type:
        :param memory_policy:
        :return:
        """
        start_time = time.time()
        if memory_size <= 0:
            logging.warning("rtMalloc() called with negative or zero size, aligned to 1!")
            memory_size = 1
        c_memory_p = ctypes.c_void_p()
        c_memory_size = ctypes.c_uint64(memory_size)
        self.rtsdll.rtMalloc.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtMalloc(ctypes.c_void_p(ctypes.addressof(c_memory_p)),
                                        c_memory_size,
                                        memory_type.value
                                        | rts_info.rt_memory_policy[memory_policy])
        self.parse_error(rt_error, "rtMalloc", f"trying to allocate {memory_size} bytes, "
                                               f"costs {round(time.time() - start_time, 3)} seconds")
        self.memory_manager[c_memory_p.value] = (memory_type.value | rts_info.rt_memory_policy[memory_policy],
                                                 c_memory_size)
        return c_memory_p

    def host_malloc(self, memory_size: int) -> ctypes.c_void_p:
        """
        RTS host malloc interface
        :param memory_size:
        :return:
        """
        start_time = time.time()
        c_memory_p = ctypes.c_void_p()
        c_memory_size = ctypes.c_uint64(memory_size)
        self.rtsdll.rtMallocHost.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtMallocHost(ctypes.c_void_p(ctypes.addressof(c_memory_p)),
                                            c_memory_size)
        self.parse_error(rt_error, "rtMallocHost", f"trying to allocate {memory_size} bytes, "
                                                   f"costs {round(time.time() - start_time, 3)} seconds")
        return c_memory_p

    def free(self, c_memory_p: ctypes.c_void_p):
        """
        RTS memfree interface
        :param c_memory_p:
        :return:
        """
        start_time = time.time()
        self.rtsdll.rtFree.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtFree(c_memory_p)
        self.parse_error(rt_error, "rtFree", f"costs {round(time.time() - start_time, 3)} seconds")
        del self.memory_manager[c_memory_p.value]

    def host_free(self, c_memory_p: ctypes.c_void_p):
        """
        RTS host memfree interface
        :param c_memory_p:
        :return:
        """
        start_time = time.time()
        self.rtsdll.rtFreeHost.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtFreeHost(c_memory_p)
        self.parse_error(rt_error, "rtFreeHost", f"costs {round(time.time() - start_time, 3)} seconds")

    def launch_kernel(self,
                      stubfunc: ctypes.c_void_p,
                      blockdim: int,
                      args: tuple, s_args: int,
                      sm_desc: Optional[Union[int, ctypes.c_uint64]],
                      stream: Optional[ctypes.c_void_p]) -> None:
        """
        Launch registered kernel function on device
        :param stubfunc:
        :param blockdim:
        :param args:
        :param s_args:
        :param sm_desc:
        :param stream:
        :return:
        """
        start_time = time.time()
        c_blockdim = ctypes.c_uint32(blockdim)
        c_args = ctypes.c_uint64 * s_args
        c_args_p = c_args(*[arg if not isinstance(arg, ctypes.c_void_p) else arg.value for arg in args])
        c_s_args = ctypes.c_uint32(s_args * 8)
        c_sm_desc = ctypes.c_void_p(sm_desc)
        self.rtsdll.rtKernelLaunch.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtKernelLaunch(stubfunc,
                                              c_blockdim,
                                              ctypes.c_void_p(ctypes.addressof(c_args_p)),
                                              c_s_args,
                                              c_sm_desc,
                                              stream)
        self.parse_error(rt_error, "rtKernelLaunch", f"costs {round(time.time() - start_time, 3)} seconds")

    def aicpu_launch_kernel(self):
        pass

    def synchronize_with_stream(self, stream: Optional[ctypes.c_void_p]) -> None:
        """
        Synchronize with device, get device task status
        :param stream:
        :return:
        """
        start_time = time.time()
        self.rtsdll.rtStreamSynchronize.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtStreamSynchronize(stream)
        self.parse_error(rt_error, "rtStreamSynchronize", f"costs {round(time.time() - start_time, 3)} seconds")

    def reset(self, device_id=None):
        """
        Reset device
        :param device_id:
        :return:
        """
        start_time = time.time()
        if device_id is None:
            device_id = self.device_id
        if device_id is None:
            logging.warning("Trying to reset device before set device, ignored.")
            return
        if self.online_profiling_status:
            # noinspection PyBroadException
            try:
                self.stop_online_profiling(None)
            except:
                pass
        for binary_kernel_pointer in tuple(self.kernel_binary_storage.keys()):
            self.unregister_device_binary_kernel(ctypes.c_void_p(binary_kernel_pointer))
        for c_memory_p_value in tuple(self.memory_manager.keys()):
            c_memory_p = ctypes.c_void_p(c_memory_p_value)
            self.free(c_memory_p)
        self.rtsdll.rtDeviceReset.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtDeviceReset(ctypes.c_int32(device_id))
        self.parse_error(rt_error, "rtDeviceReset", f"costs {round(time.time() - start_time, 3)} seconds")
        self.device_id = None

    def start_online_profiling(self, stream: Optional[ctypes.c_uint64], profiling_count: int):
        """
        Start online task profiling
        :param stream:
        :param profiling_count:
        :return:
        """
        start_time = time.time()
        if self.online_profiling_status:
            logging.warning("RTS Online Profiling already launched!")
            return
        self.rtsdll.rtStartOnlineProf.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtStartOnlineProf(stream, ctypes.c_uint32(profiling_count))
        self.parse_error(rt_error, "rtStartOnlineProf", f"costs {round(time.time() - start_time, 3)} seconds")
        self.online_profiling_status = True

    def stop_online_profiling(self, stream: Optional[ctypes.c_uint64]):
        """
        Stop online task profiling
        :param stream:
        :return:
        """
        start_time = time.time()
        if self.online_profiling_status:
            self.rtsdll.rtStopOnlineProf.restype = ctypes.c_uint64
            rt_error = self.rtsdll.rtStopOnlineProf(stream)
            self.parse_error(rt_error, "rtStopOnlineProf", f"costs {round(time.time() - start_time, 3)} seconds")
            self.online_profiling_status = False
        else:
            logging.warning("RTS Online Profiling already stopped!")

    def get_online_profiling_data(self, stream: Optional[ctypes.c_uint64], profiling_count: int):
        """
        Get online task profiling result
        :param stream:
        :param profiling_count:
        :return:
        """
        start_time = time.time()
        # noinspection PyTypeChecker
        c_structs_type: Any = rts_structures.rtProfDataInfo_t * profiling_count
        c_structs = c_structs_type()
        c_structs_p = ctypes.cast(c_structs, ctypes.POINTER(rts_structures.rtProfDataInfo_t))
        c_profdata_id = ctypes.c_uint32(profiling_count)
        self.rtsdll.rtGetOnlineProfData.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtGetOnlineProfData(stream, c_structs_p, c_profdata_id)
        self.parse_error(rt_error, "rtGetOnlineProfData", f"costs {round(time.time() - start_time, 3)} seconds")
        return c_structs_p

    def start_profiler(self, device_id: int):
        """
        Start profiling
        :param device_id:
        :return:
        """
        start_time = time.time()
        # noinspection PyBroadException
        try:
            self.__set_MsprofReporterCallbackPlaceholder()
        except:
            logging.warning("Set MsprofReporterCallback failed")
        c_device_ids_type: Any = ctypes.c_uint32 * rts_info.RT_PROF_MAX_DEV_NUM
        c_device_ids = c_device_ids_type(device_id)
        c_prof_config = ctypes.c_uint64(0b11111)
        invalid = False
        # noinspection PyBroadException
        try:
            self.rtsdll.rtProfilerStart.restype = ctypes.c_uint64
            rt_error = self.rtsdll.rtProfilerStart(c_prof_config,
                                                   ctypes.c_int32(1),
                                                   ctypes.c_void_p(ctypes.addressof(c_device_ids)))
            self.parse_error(rt_error, "rtProfilerStart", f"costs {round(time.time() - start_time, 3)} seconds")
        except:
            invalid = True
        if invalid:
            self.rtsdll.rtProfSetProSwitch.restype = ctypes.c_uint64
            start_type = rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_START.value
            rt_prof_command_handle = \
                rts_structures.rtProfCommandHandle_t(profSwitch=c_prof_config,
                                                     profSwitchHi=ctypes.c_uint64(0),
                                                     devNums=ctypes.c_uint32(1),
                                                     devIdList=c_device_ids,
                                                     modelId=ctypes.c_uint32(3),
                                                     type=ctypes.c_uint32(
                                                         start_type)
                                                     )
            rt_error = self.rtsdll.rtProfSetProSwitch(ctypes.pointer(rt_prof_command_handle),
                                                      ctypes.sizeof(rts_structures.rtProfCommandHandle_t))
            self.parse_error(rt_error, "rtProfSetProSwitch", f"costs {round(time.time() - start_time, 3)} seconds")

    def stop_profiler(self, device_id: int):
        """
        Start profiling
        :param device_id:
        :return:
        """
        start_time = time.time()
        c_device_ids_type: Any = ctypes.c_uint32 * rts_info.RT_PROF_MAX_DEV_NUM
        c_device_ids = c_device_ids_type(device_id)
        c_prof_config = ctypes.c_uint64(0b11111)
        # noinspection PyBroadException
        try:
            self.rtsdll.rtProfilerStop.restype = ctypes.c_uint64
            rt_error = self.rtsdll.rtProfilerStop(c_prof_config,
                                                  ctypes.c_int32(1),
                                                  ctypes.c_void_p(ctypes.addressof(c_device_ids)))
            self.parse_error(rt_error, "rtProfilerStop", f"costs {round(time.time() - start_time, 3)} seconds")
        except:
            self.rtsdll.rtProfSetProSwitch.restype = ctypes.c_uint64
            stop_type = rts_info.MsprofCommandHandleType.PROF_COMMANDHANDLE_TYPE_STOP.value
            rt_prof_command_handle = \
                rts_structures.rtProfCommandHandle_t(profSwitch=c_prof_config,
                                                     profSwitchHi=ctypes.c_uint64(0),
                                                     devNums=ctypes.c_uint32(1),
                                                     devIdList=c_device_ids,
                                                     modelId=ctypes.c_uint32(3),
                                                     type=ctypes.c_uint32(
                                                         stop_type)
                                                     )
            rt_error = self.rtsdll.rtProfSetProSwitch(ctypes.pointer(rt_prof_command_handle),
                                                      ctypes.sizeof(rt_prof_command_handle))
            self.parse_error(rt_error, "rtProfSetProSwitch", f"costs {round(time.time() - start_time, 3)} seconds")

    def __set_MsprofReporterCallbackPlaceholder(self):
        start_time = time.time()
        self.rtsdll.rtSetMsprofReporterCallback.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtSetMsprofReporterCallback(MsprofReporterCallbackPlaceholder)
        self.parse_error(rt_error, "rtSetMsprofReporterCallback", f"costs {round(time.time() - start_time, 3)} seconds")

    def get_memory_info_ex(self, memory_info_type: rts_info.RTS_MEMORY_INFO_TYPE) -> Tuple[int, int]:
        """
        Get device memory info
        :param memory_info_type:
        :return:
        """
        start_time = time.time()
        # noinspection PyTypeChecker
        _type: Any = ctypes.c_size_t * 1
        _free = _type()
        _total = _type()
        self.rtsdll.rtMemGetInfoEx.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtMemGetInfoEx(memory_info_type.value,
                                              _free,
                                              _total)
        self.parse_error(rt_error, "rtMemGetInfoEx", f"costs {round(time.time() - start_time, 3)} seconds")
        return int(_free[0]), int(_total[0])

    def get_aicore_spec(self) -> rts_structures.rtAiCoreSpec_t:
        """
        Get AiCore Specification
        :return: Struct rtAiCorSpec_t
        cubeFreq
        cubeMSize
        cubeKSize
        cubeNSize
        """
        start_time = time.time()
        c_struct_type: Any = rts_structures.rtAiCoreSpec_t
        c_struct = (c_struct_type * 1)()
        self.rtsdll.rtGetAiCoreSpec.restype = ctypes.c_uint64
        rt_error = self.rtsdll.rtGetAiCoreSpec(c_struct)
        self.parse_error(rt_error, "rtGetAiCoreSpec", f"costs {round(time.time() - start_time, 3)} seconds")
        return c_struct[0]

    @staticmethod
    def _parse_error_code(error_type: int, error_code: int) -> str:
        if error_code >= len(rts_info.rt_raw_error_code_dict[error_type]):
            return hex(0x07000000 + error_type + error_code)
        return rts_info.rt_raw_error_code_dict[error_type][error_code]

    @staticmethod
    def _parse_acl_error_code(error_code: int) -> str:
        if error_code in rts_info.rt_acl_error_code_dict:
            return rts_info.rt_acl_error_code_dict[error_code]
        return ""

    def parse_error(self, rt_error: ctypes.c_uint64, rt_api_name: str, extra_info: str) -> NoReturn:
        """
        Parse error code returned by rts interface
        :param rt_error:
        :param rt_api_name:
        :param extra_info:
        :return:
        """
        # Convert rt_error to int if received ctypes object
        if isinstance(rt_error, ctypes.c_uint64):
            rt_error = rt_error.value
        elif isinstance(rt_error, int):
            pass
        else:
            raise TypeError("Invalid rt_error type %s for %s" % (str(type(rt_error)), str(rt_error)))
        # RT Success
        if rt_error == 0x0:
            logging.debug(f"Runtime API Call {rt_api_name}() Success, {extra_info}")
            return

        rt_error_magic = rt_error & 0xFF000000
        if rt_error_magic != 0x07000000 and not self.camodel:
            acl_result = self._parse_acl_error_code(rt_error)
            if acl_result:
                raise RuntimeError(f"Runtime API Call {rt_api_name}() Failed: {acl_result}, {extra_info}")
            raise RuntimeError(f"Received invalid runtime error code for Runtime API Call {rt_api_name}(): "
                               f"{hex(rt_error)}, {extra_info}")
        rt_error_type = rt_error & 0x00FF0000
        if rt_error_type not in rts_info.rt_error_type_dict and not self.camodel:
            raise RuntimeError(f"Received invalid runtime error type for Runtime API Call {rt_api_name}(): "
                               f"{hex(rt_error)}, {extra_info}")
        rt_error_code = rt_error & 0x0000FFFF
        raise RuntimeError(f"Runtime API Call {rt_api_name}() Failed: "
                           f"{self._parse_error_code(rt_error_type, rt_error_code)}, {extra_info}")
