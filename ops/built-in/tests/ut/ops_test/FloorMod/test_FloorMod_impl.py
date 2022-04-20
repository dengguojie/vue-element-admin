#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info
from tbe.common.platform.platform_info import set_current_compile_soc_info
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.floor_mod import check_supported as check_supported_static
from impl.dynamic.floor_mod import check_supported as check_supported_dynamic
import numpy as np
import os
import tbe
import te
ut_case = BroadcastOpUT("FloorMod", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1), (1, 1))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 32), (16, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 32), (16, 2, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 4, 32), (16, 2, 4, 32))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (512, 1024), (512, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (32, 128, 1024), (32, 128, 1024))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 512, 1), (1,))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (9973, 1), (9973, 1))
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 12), (10, 11), expect=RuntimeError)
# ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 13), (10, 11, 12), expect=RuntimeError)

ut_case.add_case(
    'Ascend910',
    {
        'params': [{"shape": (10, 12), "dtype": "int32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"},
                   {"shape": (10, 12), "dtype": "int32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"},
                   {"shape": (10, 12), "dtype": "int32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_precision'},
        'case_name': 'floor_mod_case1',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True,
    },
)

ut_case.add_case(
    'Ascend910',
    {
        'params': [{"shape": (10, 12), "dtype": "float32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"},
                   {"shape": (10, 12), "dtype": "float32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"},
                   {"shape": (10, 12), "dtype": "float32", "format": "ND", "ori_shape": (10, 12),"ori_format": "ND"}
        ],
        'addition_params': {'impl_mode': 'high_precision'},
        'case_name': 'floor_mod_case2',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True,
    },
)
# ============ auto gen ["Ascend910"] test cases end =================


def test_check_supported_001(test_args): 
    def side_effects(*args):
        context = tbe.common.context.op_context.OpContext()
        context.add_addition("op_impl_mode_dict", {"FloorMod": "high_precision"})
        return context

    with patch('tbe.common.context.op_context.get_context', MagicMock(side_effect=side_effects)):
        check_supported_static({"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")

        check_supported_dynamic({"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")
        # ut_case.run("Ascend910", check_supported_static)
        # ut_case.run("Ascend910", check_supported_dynamic)

def test_check_supported_002(test_args): 
    def side_effects(*args):
        context = tbe.common.context.op_context.OpContext()
        context.add_addition("op_impl_mode_dict", {"FloorMod": "high_preformance"})
        return context

    with patch('tbe.common.context.op_context.get_context', MagicMock(side_effect=side_effects)):
        check_supported_static({"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")

        check_supported_dynamic({"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        {"shape": (16,2), "dtype": "float32", "format": "ND", "ori_shape": (16,2),"ori_format": "ND"},
                        "floor_mod",
                        "high_precision")

        # ut_case.run("Ascend910", check_supported_static)
        # ut_case.run("Ascend910", check_supported_dynamic)

ut_case.add_cust_test_func("Ascend910A", test_func=test_check_supported_001)
ut_case.add_cust_test_func("Ascend910A", test_func=test_check_supported_002)

def calc_expect_func(x, y, output):
    input_x = x['value'].astype(np.float32)
    input_y = y['value'].astype(np.float32)
    result_np = np.divide(input_x, input_y)
    result_np = np.floor(result_np)
    result_np = np.multiply(result_np, input_y)
    result_np = np.subtract(input_x, result_np)

    result_np = result_np.astype(output['dtype'])
    return result_np

precision_case1 = {"params": [{"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,32), "dtype": "float16", "format": "ND", "ori_shape": (16,32),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case2 = {"params": [{"shape": (1,2), "dtype": "float32", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,2), "dtype": "float32", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"input"},
                              {"shape": (1,2), "dtype": "float32", "format": "ND", "ori_shape": (1,2),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}
precision_case3 = {"params": [{"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"input"},
                              {"shape": (16,2,32), "dtype": "float32", "format": "ND", "ori_shape": (16,2,32),"ori_format": "ND","param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910A",precision_case1)
ut_case.add_precision_case("Ascend910A",precision_case2)
ut_case.add_precision_case("Ascend910A",precision_case3)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, ".mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
