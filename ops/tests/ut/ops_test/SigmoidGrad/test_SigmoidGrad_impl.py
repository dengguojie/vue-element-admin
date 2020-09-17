# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

SigmoidGrad ut case
"""
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("SigmoidGrad", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910"], ["float16", "float32"],
                                  (10, 13), (10, 11, 12), expect=RuntimeError)

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)
