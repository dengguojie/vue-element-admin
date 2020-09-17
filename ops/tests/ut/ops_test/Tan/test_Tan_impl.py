#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT

ut_case = ElementwiseOpUT("Tan", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1,))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (1, 1))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 32))
# ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910"], ["float16", "float32", "int32"], (16, 2, 4, 32))

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run("Ascend910")
