#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT

ut_case = ElementwiseOpUT("Threshold", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend910A"], ["float16", "float32"], (512, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (2, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (4096, 1024))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (32, 128, 1024))XW
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (100, 100))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 512, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1, 16, 512, 512))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (9973, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (1024, 1024, 256))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (11, 33))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 12))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32"], (10, 13))

# ============ auto gen ["Ascend910"] test cases end =================
# import pytest
if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)