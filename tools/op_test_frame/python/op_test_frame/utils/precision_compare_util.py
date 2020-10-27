# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np
from op_test_frame.common import precision_info, op_status
from op_test_frame.common.precision_info import PrecisionStandard, PrecisionCompareResult


def _get_np_dtype(d_type):
    if d_type.strip() == "float16":
        return np.float16
    elif d_type.strip() == "float32":
        return np.float32
    elif d_type.strip() == "float64" or d_type.strip() == "double":
        return np.float64
    elif d_type.strip() == "int8":
        return np.int8
    elif d_type.strip() == "int16":
        return np.int16
    elif d_type.strip() == "int32":
        return np.int32
    elif d_type.strip() == "uint8":
        return np.uint8
    elif d_type.strip() == "uint16":
        return np.uint16
    elif d_type.strip() == "uint32":
        return np.uint32
    return np.float16


def compare_precision(actual_data_file: str, expect_data_file: str,
                      precision_standard: PrecisionStandard) -> PrecisionCompareResult:
    if not isinstance(actual_data_file, str):
        actual_data = actual_data_file.reshape([-1, ])
    else:
        actual_data_dt_str = actual_data_file[-14:]
        actual_data_dt_str = actual_data_dt_str[actual_data_dt_str.index("_") + 1:-4]
        np_dtype = _get_np_dtype(actual_data_dt_str)
        actual_data = np.fromfile(actual_data_file, np_dtype)

    if not isinstance(expect_data_file, str):
        expect_data = expect_data_file.reshape([-1, ])
        np_dtype = expect_data.dtype
    else:
        expect_data_dt_str = expect_data_file[-14:]
        expect_data_dt_str = expect_data_dt_str[expect_data_dt_str.index("_") + 1:-4]
        np_dtype = _get_np_dtype(expect_data_dt_str)
        expect_data = np.fromfile(expect_data_file, np_dtype)

    actual_size = len(actual_data)
    expect_size = len(expect_data)
    compare_size = min(actual_size, expect_size)
    atol_cnt = 0
    max_atol_cnt = 0
    if not precision_standard:
        precision_standard = precision_info.get_default_standard(np_dtype)
    for i in range(compare_size):
        min_abs_actual_expect = min(abs(actual_data[i]), abs(expect_data[i]))
        if abs(actual_data[i] - expect_data[i]) > min_abs_actual_expect * precision_standard.atol:
            atol_cnt += 1
            if precision_standard.max_atol and \
                    abs(actual_data[i] - expect_data[i]) > min_abs_actual_expect * precision_standard.max_atol:
                max_atol_cnt += 1

    status = op_status.SUCCESS
    err_msg = ""
    if atol_cnt > precision_standard.rtol * compare_size:
        status = op_status.FAILED
        err_msg = "Error count: %s larger than (rtol: %s * data_size: %s). " % (
            str(atol_cnt), str(precision_standard.rtol), str(compare_size))
    if max_atol_cnt > 0:
        status = op_status.FAILED
        err_msg += "Max atol larger than max_atol: %s . " % str(precision_standard.max_atol)

    return PrecisionCompareResult(status, err_msg)
