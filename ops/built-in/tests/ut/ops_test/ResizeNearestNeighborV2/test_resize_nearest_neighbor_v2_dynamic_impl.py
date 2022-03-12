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
ut test: for resize_nearest_neighbor_v2
"""

import math
import numpy as np

from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("ResizeNearestNeighborV2", "impl.dynamic.resize_nearest_neighbor_v2", "resize_nearest_neighbor_v2")

case1 = {"params": [{"shape": (32, 1, 512, 512, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (32, 1, 512, 512, 16), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (32, 1, 512, 512, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_1",
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
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1, 1, 3, 2, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (2,), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1, 1, 1, 2, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1, 1, 3, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (2,), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1, 1, 2, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1, 1, 16, 4, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (2,), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1, 1, 15, 2, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (16, 16, 16, 16, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (2,), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (16, 16, 16, 31, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


case8 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (16, 16, 16, 16, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (-1,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (2,), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (16, 16, 32, 8, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    False, False],
         "case_name": "dynamic_resize_nearest_neighbor_v2_d_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
ut_case.add_case("all", case6)
ut_case.add_case("all", case7)
ut_case.add_case("all", case8)

# 'pylint: disable=unused-argument
def test_op_check_supported_1(test_arg):
    from impl.dynamic.resize_nearest_neighbor_v2 import check_supported
    images = {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NC1HWC0",
              "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NC1HWC0",
              "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    size = {'ori_shape': (2,), 'shape': (2,), 'ori_format': 'NHWC', 'format': 'NHWC',
            'dtype': 'int32', "const_value": (10, 10)}
    y = {"shape": (-1, -1, 10, 10, 16), "dtype": "float32", "format": "NC1HWC0",
         "ori_shape": (34, 2, 10, 10, 16), "ori_format": "NC1HWC0",
         "range": [(1, None), (1, None), (10, 10), (10, 10), (1, None)]}

    if not check_supported(images, size, y, align_corners=False, half_pixel_centers=False,
                       kernel_name="resize_nearest_neighbor_v2"):
        raise Exception("Failed to call check_supported in resize_nearest_neighbor_v2.")

ut_case.add_cust_test_func(test_func=test_op_check_supported_1)


def calc_expect_func(images, size, y, align_corners, half_pixel_center):
    srcN, srcC1, srcH, srcW, srcC0 = images.get("run_shape")
    _, _, dstH, dstW, _ = y.get("run_shape")

    hScale, wScale = srcH / dstH, srcW / dstW
    if align_corners and dstH > 1:
        hScale = (srcH - 1.0) / (dstH - 1.0)
    if align_corners and dstW > 1:
        wScale = (srcW - 1.0) / (dstW - 1.0)

    dtype = images.get("dtype")
    srcVal = images.get("value")
    dstImages = np.zeros(shape=(srcN, srcC1, dstH, dstW, srcC0),
                         dtype=np.float16 if dtype in ("float16",) else np.float32)
    for n in range(srcN):
        for c1 in range(srcC1):
            for h in range(dstH):
                for w in range(dstW):
                    srcX, srcY = h * hScale, w * wScale
                    if half_pixel_center:
                        srcX, srcY = (0.5 + h) * hScale, (0.5 + w) * wScale

                    if align_corners:
                        dstImages[n, c1, h, w] = srcVal[n, c1, round(srcX), round(srcY)]
                    else:
                        dstImages[n, c1, h, w] = srcVal[n, c1, math.floor(srcX), math.floor(srcY)]

    return dstImages


def build_precision_case(case, images, size, alialign_corners, half_pixel_center):
    image_dynamic_shape = (-1, -1, -1, -1, 16)
    image_dynamic_range = [(1, None), (1, None), (1, None), (1, None)]
    image_dtype = images.get("dtype")

    srcN, srcC1, srcH, srcW, srcC0 = images.get("run_shape")
    dstH, dstW = size.get("value")
    size_dtype = size.get("dtype")

    input_ori_shape = (srcN, srcH, srcW, srcC1 * srcC0)
    input_run_shape = images.get("run_shape")
    output_ori_shape = (srcN, dstH, dstW, srcC1 * srcC0)
    output_run_shape = (srcN, srcC1, dstH, dstW, srcC0)

    return {
        "params": [{"shape": image_dynamic_shape, "ori_shape": input_ori_shape, "run_shape": input_run_shape,
                    "range": image_dynamic_range,
                    "format": "NC1HWC0", "ori_format": "NHWC",
                    "dtype": image_dtype, "param_type": "input"},
                   {"shape": (2,), "ori_shape": (2,), "run_shape": (2,),
                    "format": "ND", "ori_format": "ND",
                    "dtype": size_dtype, "param_type": "input",
                    "value": np.array([dstH, dstW]).astype(np.int32 if size_dtype in ("int32",) else "int64")},
                   {"shape": image_dynamic_shape, "ori_shape": output_ori_shape, "run_shape": output_run_shape,
                    "range": image_dynamic_range,
                    "format": "NC1HWC0", "ori_format": "NHWC",
                    "dtype": image_dtype, "param_type": "output"},
                   alialign_corners, half_pixel_center
                   ],
        "case_name": case,
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
    }

'''
# 111000
ut_case.add_precision_case(
    "all", build_precision_case("ResizeNN_101",
                                images={"dtype": "float16", "run_shape": (1, 1, 32, 32, 16)},
                                size={"value": [96, 96], "dtype": "int32"},
                                alialign_corners=False, half_pixel_center=False))

# 100000
ut_case.add_precision_case(
    "all", build_precision_case("ResizeNN_102",
                                images={"dtype": "float16", "run_shape": (1, 1, 2, 2, 16)},
                                size={"value": [3, 3], "dtype": "int32"},
                                alialign_corners=False, half_pixel_center=False))

# 101000
ut_case.add_precision_case(
    "all", build_precision_case("ResizeNN_103",
                                images={"dtype": "float16", "run_shape": (1, 1, 32, 32, 16)},
                                size={"value": [48, 96], "dtype": "int32"},
                                alialign_corners=False, half_pixel_center=False))

# 103000
ut_case.add_precision_case(
    "all", build_precision_case("ResizeNN_1031",
                                images={"dtype": "float16", "run_shape": (1, 1, 32, 32, 16)},
                                size={"value": [48, 32], "dtype": "int32"},
                                alialign_corners=False, half_pixel_center=False))

# 113000
ut_case.add_precision_case(
    "all", build_precision_case("ResizeNN_104",
                                images={"dtype": "float16", "run_shape": (1, 1, 32, 32, 16)},
                                size={"value": [64, 96], "dtype": "int32"},
                                alialign_corners=False, half_pixel_center=False))

# 111001
ut_case.add_precision_case(
    "all", build_precision_case("ResizeNN_105",
                                images={"dtype": "float16", "run_shape": (1, 1, 1, 5, 16)},
                                size={"value": [2, 10], "dtype": "int32"},
                                alialign_corners=False, half_pixel_center=False))

# 113001
ut_case.add_precision_case(
    "all", build_precision_case("ResizeNN_1051",
                                images={"dtype": "float16", "run_shape": (1, 1, 1, 5, 16)},
                                size={"value": [2, 5], "dtype": "int32"},
                                alialign_corners=False, half_pixel_center=False))
'''

if __name__ == '__main__':
    ut_case.run("Ascend910A", simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
