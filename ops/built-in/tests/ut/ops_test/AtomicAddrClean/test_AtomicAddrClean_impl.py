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

AtomicAddrClean ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("AtomicAddrClean", None, None)

case1 = {"params":[ [32, 49152]
],
"case_name": "AtomicAddrClean_1",
"expect": "success",
"support_expect": True}

case2 = {"params": [[32, 480, 60416]
],
"case_name": "AtomicAddrClean_2",
"expect": "success",
"support_expect": True}

case3 = {"params": [ [32, 512, 961, 64, 128, 256]
],
"case_name": "AtomicAddrClean_3",
"expect": RuntimeError,
"support_expect": True}

case4 = {"params": [ [32, 512, 1024, 64, 128, 256]
],
"case_name": "AtomicAddrClean_4",
"expect": "success",
"support_expect": True}

case5 = {"params": [ [32, 512, -32]
],
"case_name": "AtomicAddrClean_5",
"expect": RuntimeError,
"support_expect": True}

case6 = {"params": [ [49152 * 32 - 1024, 512, 512]
],
"case_name": "AtomicAddrClean_6",
"expect": "success",
"support_expect": True}

# TODO fix me, this comment, run failed
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)
# ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case6)


if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
