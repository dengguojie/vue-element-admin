#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("BitwiseAnd", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["int16", "uint16"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run("Ascend910")
