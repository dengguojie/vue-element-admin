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
"""
ut test: for resize_nearest_neighbor_v2_grad
"""

from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeNearestNeighborV2Grad", "impl.dynamic.resize_nearest_neighbor_v2_grad",
               "resize_nearest_neighbor_v2_grad")

case1 = {"params": [{"shape": (32, 1, 2, 2, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (34, 2, 1, 1, 16),
                     "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (32, 1, 4, 4, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_grad_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (34, 2, 1, 1, 16),
                     "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_grad_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)


if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
