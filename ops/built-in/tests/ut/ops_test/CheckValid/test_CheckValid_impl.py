#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("CheckValid", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (1,4), (1,2,3,4))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (2,4), (2,2,3,4))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (8,4), (8,2,3,4))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (16,4), (16,2,3,4))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (16,4), (16,4,6,7))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (16,4), (16,16,16,16))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (16,4), (16,16,128,1))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (16,4), (16,16,32,32))
ut_case.add_broadcast_case_simple(["Ascend910A"], ["float16"], (16,4), (16,512,512,513))
ut_case.add_broadcast_case_simple(["Ascend910A","Ascend310","Ascend710"], ["float16"], (16,4), (16,1,1,1))



# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run(["Ascend910A","Ascend310","Ascend710"])
    # ut_case.run()
    exit(0)
