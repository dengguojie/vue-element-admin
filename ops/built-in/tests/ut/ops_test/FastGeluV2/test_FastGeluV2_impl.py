#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
from op_test_frame.common import precision_info

ut_case = ElementwiseOpUT("FastGeluV2", None, None)

# ============ auto gen ["Ascend710"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (1,))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (512, 1024))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (2, 1024))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (4096, 1024))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (32, 128, 1024))
ut_case.add_elewise_case_simple(["Ascend710"], ["float16", "float32"], (100, 100))
ut_case.add_elewise_case_simple(["all"], ["float16", "float32"], (1, 512, 1))

if __name__ == '__main__':
    ut_case.run("Ascend310")