#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

import tbe
from tbe.common.platform.platform_info import set_current_compile_soc_info
from impl.dynamic.scan_pq_codes import scan_pq_codes
import tbe.common.context.op_info as operator_info

def test_scan_pq_codes_01():
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("ScanPQCodes", "ScanPQCodes")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        scan_pq_codes({"shape": (10240, 16), "dtype": "uint8", "format": "ND", "ori_shape": (10240, 16), "ori_format": "ND",
             "range": [(10240, 10240), (16, 16)]},
            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "int64", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1, 16, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 16, 256), "ori_format": "ND",
             "range": [(1, 1), (16, 16), (256, 256)]},

            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (10240,), "dtype": "float16", "format": "ND", "ori_shape": (10240, ), "ori_format": "ND",
             "range": [(10240, 10240)]},
            {"shape": (160,), "dtype": "float16", "format": "ND", "ori_shape": (160, ), "ori_format": "ND",
             "range": [(160, 160)]},
            {"shape": (10240,), "dtype": "int32", "format": "ND", "ori_shape": (10240, ), "ori_format": "ND",
             "range": [(10240, 10240)]},
            {"shape": (10240,), "dtype": "int32", "format": "ND", "ori_shape": (10240, ), "ori_format": "ND",
             "range": [(10240, 10240)]},
            10240, 64, 0, 1, 0)

def test_scan_pq_codes_02():
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("ScanPQCodes", "ScanPQCodes")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        scan_pq_codes({"shape": (20480, 16), "dtype": "uint8", "format": "ND", "ori_shape": (20480, 16),
            "ori_format": "ND",
             "range": [(20480, 20480), (16, 16)]},
            {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "int64", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2, 16, 256), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 256), "ori_format": "ND",
             "range": [(2, 2), (16, 16), (256, 256)]},

            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
            "range": [(1, 1)]},
            {"shape": (262144,), "dtype": "float16", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            {"shape": (4096,), "dtype": "float16", "format": "ND", "ori_shape": (4096, ), "ori_format": "ND",
            "range": [(4096, 4096)]},
            {"shape": (262144,), "dtype": "int32", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            {"shape": (262144,), "dtype": "int32", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            262144, 64, 0, 2, 1)

def test_scan_pq_codes_03():
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("ScanPQCodes", "ScanPQCodes")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        scan_pq_codes({"shape": (20480, 16), "dtype": "uint8", "format": "ND", "ori_shape": (20480, 16),
             "ori_format": "ND",
             "range": [(20480, 20480), (16, 16)]},
            {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "int64", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2, 16, 256), "dtype": "float16", "format": "ND", "ori_shape": (2, 16, 256), "ori_format": "ND",
             "range": [(2, 2), (16, 16), (256, 256)]},

            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
            "range": [(1, 1)]},
            {"shape": (262144,), "dtype": "float16", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            {"shape": (4096,), "dtype": "float16", "format": "ND", "ori_shape": (4096, ), "ori_format": "ND",
            "range": [(4096, 4096)]},
            {"shape": (262144,), "dtype": "int32", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            {"shape": (262144,), "dtype": "int32", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            262144, 64, 0, 2, 0)

def test_scan_pq_codes_04():
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("ScanPQCodes", "ScanPQCodes")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        scan_pq_codes({"shape": (20480, 32), "dtype": "uint8", "format": "ND", "ori_shape": (20480, 32),
            "ori_format": "ND",
             "range": [(20480, 20480), (32, 32)]},
            {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "float16", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "int32", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2,), "dtype": "int64", "format": "ND", "ori_shape": (2, ), "ori_format": "ND",
             "range": [(2, 2)]},
            {"shape": (2, 32, 256), "dtype": "float16", "format": "ND", "ori_shape": (2, 32, 256), "ori_format": "ND",
             "range": [(2, 2), (32, 32), (256, 256)]},

            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
            "range": [(1, 1)]},
            {"shape": (262144,), "dtype": "float16", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            {"shape": (4096,), "dtype": "float16", "format": "ND", "ori_shape": (4096, ), "ori_format": "ND",
            "range": [(4096, 4096)]},
            {"shape": (262144,), "dtype": "int32", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            {"shape": (262144,), "dtype": "int32", "format": "ND", "ori_shape": (262144, ), "ori_format": "ND",
            "range": [(262144, 262144)]},
            262144, 64, 0, 2, 0)

if __name__ == "__main__":
    set_current_compile_soc_info("Ascend710")
    test_scan_pq_codes_01()
    test_scan_pq_codes_02()
    test_scan_pq_codes_03()
    test_scan_pq_codes_04()
    exit(0)