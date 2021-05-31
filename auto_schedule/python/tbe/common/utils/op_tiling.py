# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
op tiling interface
"""

import os
import ctypes
import json
import struct
import hashlib
import threading
from pathlib import Path

from tbe.common.utils.errormgr import get_error_message


_MAX_RUN_INFO_SIZE = 1024*64
_ASCEND_OPP_PATH_ENV = "ASCEND_OPP_PATH"
_ASCEND_OPP_PATH_DEFAULT = "/usr/local/Ascend/opp"
_BUILTIN_TILING_PATH = "op_impl/built-in/ai_core/tbe/op_tiling/liboptiling.so"
_CUSTOM_TILING_PATH_DEFAULT = "op_impl/custom/ai_core/tbe/op_tiling/liboptiling.so"

# Tiling is likely running in thread pool or single-threaded process,
# using thread local buffer reduces memory allocation
_TILING_DATA = threading.local()

# Initializing thread local data when importing this py module,
# which is helpful in case of single-threaded profiling test
_TILING_DATA.buf = ctypes.create_string_buffer(_MAX_RUN_INFO_SIZE)
_TILING_DATA.buf_size = ctypes.c_size_t(_MAX_RUN_INFO_SIZE)

def do_op_tiling(optype, compile_info, inputs, outputs, compile_info_hash=None, timer=None):
    """
    do op tilinng
    """
    def _load_lib():
        opp_path = Path(os.environ.get(_ASCEND_OPP_PATH_ENV, _ASCEND_OPP_PATH_DEFAULT))
        builtin_optiling_lib_path = opp_path.joinpath(_BUILTIN_TILING_PATH)
        custom_optiling_lib_path = opp_path.joinpath(_CUSTOM_TILING_PATH_DEFAULT)
        libregister = ctypes.CDLL("libregister.so")
        ctypes.CDLL(builtin_optiling_lib_path)
        try:
            ctypes.CDLL(custom_optiling_lib_path)
        except OSError:
            # Custom op tiling lib may not exists
            pass
        return libregister

    libregister = _load_lib()
    optype_c = optype.encode('utf_8')
    compile_info_c = json.dumps(compile_info).encode('utf_8')
    inputs_c = json.dumps(inputs).encode('utf_8')
    outputs_c = json.dumps(outputs).encode('utf_8')
    if not compile_info_hash:
        hashstr = hashlib.sha1()
        hashstr.update(compile_info_c)
        compile_info_hash = hashstr.hexdigest()
    compile_info_hash_c = compile_info_hash.encode('utf_8')

    if not hasattr(_TILING_DATA, "buf") or not hasattr(_TILING_DATA, "buf_size"):
        _TILING_DATA.buf = ctypes.create_string_buffer(_MAX_RUN_INFO_SIZE)
        _TILING_DATA.buf_size = ctypes.c_size_t(_MAX_RUN_INFO_SIZE)

    tiling_func = libregister.TbeOpTilingPyInterfaceEx2
    if isinstance(timer, list):
        array_c = ctypes.c_uint64 * 3
        elapse_c = array_c(0, 0, 0)
        res = tiling_func(optype_c, compile_info_c, inputs_c, outputs_c,
                          _TILING_DATA.buf, _TILING_DATA.buf_size, compile_info_hash_c,
                          elapse_c)
        for i in range(0, 3):
            timer.append(elapse_c[i])
    else:
        res = tiling_func(optype_c, compile_info_c, inputs_c, outputs_c,
                          _TILING_DATA.buf, _TILING_DATA.buf_size, compile_info_hash_c,
                          ctypes.c_void_p())
    if not res:
        dict_args = dict()
        dict_args["errCode"] = "E90003"
        dict_args["detailed_cause"] = "Tiling func failed."
        raise RuntimeError(dict_args, get_error_message(dict_args))

    run_info = json.loads(_TILING_DATA.buf.value)
    run_info['tiling_data'] = bytes.fromhex(run_info['tiling_data'])
    return run_info


def decode(tiling_data, fmt):
    """decode tiling data"""
    offset = 0

    def _get_value(tiling_data, fmt, offset=0):
        """
        fmt example: [-1, "int"]   # int arrary of unknown size
                     [10, "int"]   # arrary of 10 ints
                     "int"         # single int
        """
        fmt_def = {"char": "c",
                   "int": "i", "uint": "I",
                   "int8": "b", "uint8": "B",
                   "int16": "h", "uint16": "H",
                   "int64": "l", "uint64": "L",
                   "double": "d"}
        count = 1
        unpack_size = 0
        if isinstance(fmt, (list, tuple)):
            count = fmt[0]
            if count < 0:
                fmt_size = struct.calcsize("i")
                res = struct.unpack_from("i", tiling_data, offset)
                fmt = fmt_def[fmt[1]]
                count = res[0]
                unpack_size += fmt_size

        if count == 0:
            return [unpack_size, []]

        fmt_str = "{}{}".format(count, fmt_def[fmt])
        fmt_size = struct.calcsize(fmt_str)
        res = struct.unpack_from(fmt_str, tiling_data, offset + unpack_size)
        unpack_size += fmt_size
        if isinstance(fmt, (list, tuple)):
            return [unpack_size, res]
        return [unpack_size, res[0]]

    res = {}
    for key, value in fmt.items():
        unpack_size, res[key] = _get_value(tiling_data, value, offset)
        offset += unpack_size

    return res
