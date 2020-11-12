import numpy as np
from op_test_frame.common import precision_info
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
    actual_data_dt_str = actual_data_file[-14:]
    actual_data_dt_str = actual_data_dt_str[actual_data_dt_str.index("_") + 1:-4]
    np_dtype = _get_np_dtype(actual_data_dt_str)
    actual_data = np.fromfile(actual_data_file, np_dtype)
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
        if abs(actual_data[i] - expect_data[i]) > min(abs(actual_data[i]),
                                                      abs(expect_data[i])) * precision_standard.atol:
            atol_cnt += 1
            if precision_standard.max_atol and abs(actual_data[i] - expect_data[i]) > min(abs(actual_data[i]),
                                                                                          abs(expect_data[
                                                                                                  i])) * precision_standard.max_atol:
                max_atol_cnt += 1

    status = "SUCCESS"
    err_msg = ""
    if atol_cnt > precision_standard.rtol * compare_size:
        status = "FAILED"
        err_msg = "Error count: %s larger than (rtol: %s * data_size: %s). " % (
            str(atol_cnt), str(precision_standard.rtol), str(compare_size))
    if max_atol_cnt > 0:
        status = "FAILED"
        err_msg += "Max atol larger than max_atol: %s . " % str(precision_standard.max_atol)

    return PrecisionCompareResult(status, err_msg)
