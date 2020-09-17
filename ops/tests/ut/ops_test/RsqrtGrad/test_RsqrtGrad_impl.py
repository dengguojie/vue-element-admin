#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("RsqrtGrad", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32", "int32", "int8"], (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run("Ascend910")
