#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ElementwiseOpUT
import numpy as np
from op_test_frame.common import precision_info
import os

ut_case = ElementwiseOpUT("Invert", None, None)


# ============ auto gen ["Ascend910A"] test cases start ===============
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (1,))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (1, 1))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (16, 32))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (16, 2, 32))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (16, 2, 4, 32))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (512, 1024))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (2, 1024))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (4096, 1024))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (32, 128, 1024))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (100, 100))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (1, 512, 1))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (1, 16, 512, 512))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (9973, 1))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (1024, 1024, 256))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (11, 33))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (10, 12))
ut_case.add_elewise_case_simple(["Ascend910A"], ["int16", "uint16"], (10, 13))

# ============ auto gen ["Ascend910A"] test cases end =================
