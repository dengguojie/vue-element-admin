#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT

ut_case = ElementwiseOpUT("Abs", None, None)

# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 4, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (512, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (2, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (4096, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (32, 128, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (100, 100))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 512, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 16, 512, 512))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (9973, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1024, 1024, 256))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (11, 33))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 12))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (10, 13))


# def test_abs(test_arg):
#     from impl import abs
#     abs({"dtype": "float16", "format": "ND", "ori_format": "ND", "shape": (2, 3, 4), "ori_shape": (2, 3, 4)},
#         {"dtype": "float16", "format": "ND", "ori_format": "ND", "shape": (2, 3, 4), "ori_shape": (2, 3, 4)})
#     pass


# ut_case.add_cust_test_func(test_func=test_abs)

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run("Ascend910")
