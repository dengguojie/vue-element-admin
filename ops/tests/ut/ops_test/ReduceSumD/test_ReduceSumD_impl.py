#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import ReduceOpUT

ut_case = ReduceOpUT("ReduceSumD", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1,), (0,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1,), 0, False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 1), (1,), True)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 1), (1,), False)
ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (101, 10241), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (101, 10241), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1023*255, ), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1023*255, ), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (0, 1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (51, 101, 1023), (0, 1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (99991, 10), (0, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (99991, 10), (0, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991, 10), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float16", "float32"], (1, 99991, 10), (1, ), False)

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run("Ascend910")
