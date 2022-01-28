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
ConcatV2D op_select_format ut test
"""
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ConcatV2D", "impl.dynamic.concat_v2_d", "op_select_format")

case1 = {"params": [[{"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                     {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (128, 128, 128, 128), "ori_format": "NHWC"}],
                    {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                    1],
         "case_name": "dynamic_concat_v2_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [[{"ori_shape": (-2,), "dtype": "float16", "format": "NHWC", "shape": (128, 128, 128, 128),
                      "ori_format": "NHWC"},
                     {"ori_shape": (-2,), "dtype": "float16", "format": "NHWC", "shape": (128, 128, 128, 128),
                      "ori_format": "NHWC"}],
                    {"ori_shape": (-2,), "dtype": "float16", "format": "NHWC", "shape": (128, 128, 128, 128),
                     "ori_format": "NHWC"},
                    1],
         "case_name": "dynamic_concat_v2_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [[{"ori_shape": (-1, -1, -1, -1), "dtype": "float16", "format": "NHWC",
                      "shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                     {"ori_shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                      "shape": (128, 128, 128, 128), "ori_format": "NHWC"}],
                    {"ori_shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                     "shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                    -1],
         "case_name": "dynamic_concat_v2_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [[{"ori_shape": (-2,), "dtype": "float16", "format": "NHWC",
                      "shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                     {"ori_shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                      "shape": (128, 128, 128, 128), "ori_format": "NHWC"}],
                    {"ori_shape": (-2,), "dtype": "float16", "format": "NHWC",
                     "shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                    -1],
         "case_name": "dynamic_concat_v2_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [[{"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                     {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                      "ori_shape": (128, 128, 128, 128), "ori_format": "NHWC"}],
                    {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC",
                     "ori_shape": (128, 128, 128, 128), "ori_format": "NHWC"},
                    -1],
         "case_name": "dynamic_concat_v2_d_op_select_format_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend310"], case3)
ut_case.add_case(["Ascend310"], case4)
ut_case.add_case(["Ascend310"], case5)

if __name__ == "__main__":
    ut_case.run("Ascend310")
