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
from unittest.mock import MagicMock
from unittest.mock import patch

ut_case = OpUT("MaxPool3DWithArgmax", "impl.max_pool3d_with_argmax", "max_pool3d_with_argmax")

case1 = {"params": [{"shape": (1, 10, 1, 6, 6, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]]}
case2 = {"params": [{"shape": (3, 87, 6, 100, 64, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 1, 1, 2], [1, 1, 10, 1, 1]]}
case3 = {"params": [{"shape": (10, 225, 1, 103, 176, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 1, 1, 1], [1, 1, 5, 3, 2]]}
case4 = {"params": [{"shape": (4, 25, 1, 70, 30, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 2, 2, 2], [1, 1, 5, 4, 2]]}
case5 = {"params": [{"shape": (1, 3, 2, 4, 4, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 2, 2, 2], [1, 1, 1, 2, 2]]}  # WholeKernel
case6 = {"params": [{"shape": (1, 2, 3, 189, 21, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 2, 3, 3], [1, 1, 1, 2, 2]]}  # CUT_H
case7 = {"params": [{"shape": (1, 3, 2, 7, 419, 16), "dtype": "float16"},
                    {}, {}, [1, 1, 2, 5, 5], [1, 1, 1, 2, 2]]}  # CUT_H_AND_W

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case7)

vals = {("tik.load3dv1", ): False}
def side_effects(*args):
    return vals[args]
with patch("impl.util.platform_adapter.tbe_platform.api_check_support", MagicMock(side_effect=side_effects)):
    ut_case.run("Ascend910A")

if __name__ == '__main__':
    ut_case.run()
    exit(0)

