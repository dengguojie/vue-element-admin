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

MaxPool3DWithArgmax ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPool3DWithArgmax", "impl.max_pool3d_with_argmax", "max_pool3d_with_argmax")

case1 = {"params": [{"shape": (1, 10, 1, 6, 6, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]]}
case2 = {"params": [{"shape": (3, 87, 6, 100, 64, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 1, 1, 2], [1, 1, 10, 1, 1]]}
case3 = {"params": [{"shape": (10, 225, 1, 103, 176, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 1, 1, 1], [1, 1, 5, 3, 2]]}
case4 = {"params": [{"shape": (4, 25, 1, 70, 30, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 2, 2, 2], [1, 1, 5, 4, 2]]}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run()
    exit(0)

