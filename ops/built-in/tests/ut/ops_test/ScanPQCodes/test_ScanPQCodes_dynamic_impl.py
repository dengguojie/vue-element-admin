#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import tbe
from tbe.common.platform import set_current_compile_soc_info
from impl.dynamic.scan_pq_codes import scan_pq_codes

ut_case = OpUT("ScanPQCodes", "impl.dynamic.scan_pq_codes", "scan_pq_codes")


def test_1951_uint8_small_shape_001(test_arg):
    set_current_compile_soc_info('Ascend710')
    with tbe.common.context.op_context.OpContext("dynamic"):
        scan_pq_codes(
            {"shape": (10240, 16), "dtype": "uint8", "format": "ND", "ori_shape": (1000, 16), "ori_format": "ND",
             "range": [(1000, 1000), (16, 16)]},
            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1, 16, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 16, 256), "ori_format": "ND",
             "range": [(1, 1), (16, 16), (256, 256)]},

            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND",
             "range": [(1, 1)]},
            10240, 64, 0, 1, 0,
            "test_1951_uint8_small_shape_001")
    set_current_compile_soc_info(test_arg)

def test_1951_uint8_small_shape_002(test_arg):
    set_current_compile_soc_info('Ascend710')
    with tbe.common.context.op_context.OpContext("dynamic"):
        scan_pq_codes(
            {"shape": (204800, 16), "dtype": "uint8", "format": "ND", "ori_shape": (2000, 16), "ori_format": "ND",
             "range": [(1000, 1000), (16, 16)]},
            {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5,), "dtype": "float16", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5, 16, 256), "dtype": "float16", "format": "ND", "ori_shape": (5, 16, 256), "ori_format": "ND",
             "range": [(5, 5), (16, 16), (256, 256)]},

            {"shape": (8,), "dtype": "int32", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5,), "dtype": "float16", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5,), "dtype": "float16", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5, ), "ori_format": "ND",
             "range": [(1, 1)]},
            204800, 64, 0, 1, 0,
            "test_1951_uint8_small_shape_002")
    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(test_func=test_1951_uint8_small_shape_001)
ut_case.add_cust_test_func(test_func=test_1951_uint8_small_shape_002)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run('Ascend710')
