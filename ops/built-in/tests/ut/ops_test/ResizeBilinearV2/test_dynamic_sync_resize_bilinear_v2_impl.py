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
"""
ut for resize
"""
import json
from te.platform.cce_conf import te_set_version
from op_test_frame.ut import OpUT


ut_case = OpUT("ResizeBilinearV2", "impl.dynamic.sync_resize_bilinear_v2", "sync_resize_bilinear_v2")

case1 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, [16, 16], [8, 8], 0, 0,
                    False, False],
         "case_name": "dynamic_sync_resize_bilinear_v2_d_fp16_to_fp16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, [16, 16], [8, 8], 0, 0,
                    False, True],
         "case_name": "dynamic_sync_resize_bilinear_v2_d_fp16_to_fp16_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, [16, 16], [8, 8], 0, 0,
                    True, False],
         "case_name": "dynamic_sync_resize_bilinear_v2_d_fp16_to_fp16_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)


if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend920A"])
