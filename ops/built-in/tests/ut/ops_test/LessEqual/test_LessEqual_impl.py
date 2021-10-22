#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = BroadcastOpUT("LessEqual", None, None)


# ============ auto gen ["Ascend910A"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (100, 100), (100, 100))
# ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1, 512, 1), (1,))
# ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16", "float32"], (11, 33), (11, 33))


def calc_expect_func(x1, x2, y):
    res = np.less_equal(x1['value'], x2['value'])
    res = res.astype(y['dtype'])
    return res

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (11,33), "dtype": "float32", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (11,33), "dtype": "int8", "format": "ND", "ori_shape": (11,33),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (100,100), "dtype": "float32", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (100,100), "dtype": "int8", "format": "ND", "ori_shape": (100,100),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32,128), "dtype": "float32", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32,128), "dtype": "int8", "format": "ND", "ori_shape": (32,128),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,16,512), "dtype": "float32", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,16,512), "dtype": "int8", "format": "ND", "ori_shape": (1,16,512),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1024,16), "dtype": "float32", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1024,16), "dtype": "int8", "format": "ND", "ori_shape": (1024,16),"ori_format": "ND", "param_type": "output"}],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })


if __name__ == '__main__':
    ut_case.run("Ascend910A")