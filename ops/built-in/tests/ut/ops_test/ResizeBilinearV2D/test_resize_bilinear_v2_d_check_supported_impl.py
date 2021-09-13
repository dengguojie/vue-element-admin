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
from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeBilinearV2D", "impl.resize_bilinear_v2_d", "check_supported")


case1 = {
    "params": [{
        "shape": (34, 2, 1, 16),
        "dtype": "float32",
        "format": "NCHW",
        "ori_shape": (34, 2, 1, 1, 16),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 1, 16),
        "dtype": "float32",
        "format": "NCHW",
        "ori_shape": (34, 2, 1, 1, 16),
        "ori_format": "NCHW"
    }, (1, 1), False, False],
    "case_name": "resize_nearest_neighbor_v2_d_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": [{
        "shape": (34, 2, 200, 16),
        "dtype": "float16",
        "format": "CHW",
        "ori_shape": (34, 2, 300, 200, 16),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 200, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 2, 300, 200, 16),
        "ori_format": "NCHW"
    }, (300, 200), False, True],
    "case_name": "resize_nearest_neighbor_v2_d_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case3 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 2, 31, 51, 16),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 2, 31, 51, 16),
        "ori_format": "NCHW"
    }, (31, 51), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case4 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, (31, 51), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case5 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (4, 16, 480, 640),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, (800, 1067), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case6 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (4, 16, 480, 640),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, (800, 1067), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_6",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}
case7 = {
    "params": [{
        "shape": (34, 2, 200, 16),
        "dtype": "float16",
        "format": "ABCD",
        "ori_shape": (34, 2, 300, 200, 16),
        "ori_format": "ABCD"
    }, {
        "shape": (34, 2, 200, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 2, 300, 200, 16),
        "ori_format": "NCHW"
    }, (300, 200), False, True],
    "case_name": "resize_nearest_neighbor_v2_d_7",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
}
case8 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (4, 16, 480, 640),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, (3000, 1067), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_8",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
}
case9 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (4, 16, 480, 640),
        "ori_format": "NCHW"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, (-10, 1067), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_9",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
}
case10 = {
    "params": [{
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (4, 16, 480, 640),
        "ori_format": "NHWC"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, (-10, 1067), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_10",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
}
case11 = {
    "params": [{
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (4, 16, 480, 640),
        "ori_format": "NHW"
    }, {
        "shape": (34, 2, 51, 16),
        "dtype": "float16",
        "format": "NCHW",
        "ori_shape": (34, 31, 51, 16),
        "ori_format": "NCHW"
    }, (-10, 1067), True, False],
    "case_name": "resize_nearest_neighbor_v2_d_11",
    "expect": "failed",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
