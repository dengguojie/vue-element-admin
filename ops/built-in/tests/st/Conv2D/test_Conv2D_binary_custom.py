# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from __future__ import absolute_import
from op_test_frame.ut import OpUT
from tbe.common.context import op_context
from tbe.common.context import get_context
from tbe.common.context import op_info
from tbe.dsl.base import operation
from impl.dynamic.conv2d import conv2d

conv2d_binary_testcase = [
    ["all",
     {'ori_shape': (1, 32, -1, -1), 'shape': (1, 2, -1, -1, 16), 'ori_format': 'NCHW',
      'dtype': 'float16', "range": [(1, 1), (32, 32), (10, 25), (10, 25)]},
     {"ori_shape": [64, 32, 3, 3], "dtype": "float16", "ori_format": "NCHW",
      "range": [(64, 64), (32, 32), (3, 3), (3, 3)]},
     None, None,
     {'ori_shape': (1, 32, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16'},
     [-1, -1, -1, -1], (0, 0, 0, 0), (1, 1, 1, 1), 1, "NCHW", 0, "success", "binary_case"]
]

binary_compile_infos = {
    "feature": [
        {"load2d_flag": False},
        {"load2d_flag": True}
    ]
}


def compile_binary_kernels(test_arg):
    for binary_info in binary_compile_infos.get("feature"):
        for case in conv2d_binary_testcase:
            with op_context.OpContext():
                with operation.dynamic():
                    context = get_context()
                    op_info_conv2d = op_info.OpInfo("Conv2D", "Conv2D")
                    op_info_conv2d.extra_params = {"multiple_templates": binary_info}
                    context.add_op_info(op_info_conv2d)
                    conv2d(*case[1:-2], "conv2d_binary")


if __name__ == "__main__":
    print("test compile_binary_kernels")
    compile_binary_kernels("")
    print("test compile_binary_kernels end")
    exit(0)
