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

import ctypes
import json
import struct


_MAX_RUN_INFO_SIZE = 16384


def do_op_tiling(optype, compile_info, inputs, outputs):
    """
    do op tilinng
    """
    libregister = ctypes.CDLL("libregister.so")
    ctypes.CDLL("liboptiling.so")

    optype_c = optype.encode('utf_8')
    compile_info_c = json.dumps(compile_info).encode('utf_8')
    inputs_c = json.dumps(inputs).encode('utf_8')
    outputs_c = json.dumps(outputs).encode('utf_8')

    run_info_c = ctypes.create_string_buffer(_MAX_RUN_INFO_SIZE)
    run_info_size_c = ctypes.c_size_t(_MAX_RUN_INFO_SIZE)

    tiling_func = libregister.TbeOpTilingPyInterface

    res = tiling_func(optype_c, compile_info_c, inputs_c, outputs_c,
                      run_info_c, run_info_size_c)
    if not res:
        raise RuntimeError("Tiling func failed")

    run_info = json.loads(run_info_c.value)
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
