# Copyright 2020 Huawei Technologies Co., Ltd
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
ut for IMGWarpResize
"""
import os
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
ut_case = OpUT("IMGWarpResize", "impl.img_warp_resize", "img_warp_resize")


# pylint: disable=unused-argument,invalid-name,too-many-locals
def get_special_shape(ori_shape, ori_format, dst_format, align_num=16):
    """
    get_special_shape
    """

    def _ceil_div(dim):
        return (dim + align_num - 1) // align_num

    dst_shape = []
    if dst_format in ("FRACTAL_NZ",):
        dst_shape = ori_shape[:-2] + [
            _ceil_div(ori_shape[-1]),
            _ceil_div(ori_shape[-2]), align_num, align_num
        ]
    dst_shape_len = len(dst_shape)
    return dst_shape if dst_shape_len != 0 else ori_shape


def tensor_dict(tensor_ori_shape, tensor_ori_format, tensor_type, tensor_format=None, is_output=False):
    """
    return a dict
    """
    if tensor_format is None:
        tensor_format = tensor_ori_format
    tensor_shape = get_special_shape(tensor_ori_shape, tensor_ori_format, tensor_format)

    gen_dict = dict()
    gen_dict["ori_shape"] = tensor_ori_shape
    gen_dict["ori_format"] = tensor_ori_format
    gen_dict["dtype"] = tensor_type
    gen_dict["shape"] = tensor_shape
    gen_dict["format"] = tensor_format
    gen_dict["range"] = [(1, 100000)] * len(tensor_shape)
    param_type = "output" if is_output else "input"
    gen_dict["param_type"] = param_type
    return gen_dict


def numpy_resization_compute(input_image_np, input_warp_np):
    """
    numpy_resization_compute
    """
    image_dtype = input_image_np.dtype
    np_batch = input_image_np.shape[0]
    np_c = input_image_np.shape[2]
    np_h = input_image_np.shape[3]
    np_w = input_image_np.shape[4]
    np_area = np_h * np_w
    input_image_np = input_image_np.reshape([np_batch, 4, np_c, np_area]).astype(np.float32)
    input_warp_np = input_warp_np.reshape([np_batch, 2, np_area]).astype(np.float32)
    output_img = np.random.uniform(0, 0, size=[np_batch, np_c, np_area]).astype(np.float32)
    for batch_idx in range(np_batch):
        lerp_x_data = input_warp_np[batch_idx, 0, :]
        lerp_y_data = input_warp_np[batch_idx, 1, :]
        lerp_x_data_int = lerp_x_data.astype(np.int32)
        lerp_y_data_int = lerp_y_data.astype(np.int32)
        lerp_x_data = lerp_x_data - lerp_x_data_int.astype(np.float32)
        lerp_y_data = lerp_y_data - lerp_y_data_int.astype(np.float32)
        for batch_c in range(np_c):
            top_left = input_image_np[batch_idx, 0, batch_c, :]
            top_right = input_image_np[batch_idx, 1, batch_c, :]
            bottom_left = input_image_np[batch_idx, 2, batch_c, :]
            bottom_right = input_image_np[batch_idx, 3, batch_c, :]
            top = (top_right - top_left) * lerp_x_data + top_left
            bottom = (bottom_right - bottom_left) * lerp_x_data + bottom_left
            out = (bottom - top) * lerp_y_data + top
            output_img[batch_idx, batch_c, :] = out
    output_img = output_img.reshape([np_batch, np_c, np_h, np_w]).astype(image_dtype)
    return output_img


def calc_expect_func(img, warp_index, output=None):
    """
    calc_expect_func
    """
    img_value = img.get("value")
    warp_index_value = warp_index.get("value")
    result = numpy_resization_compute(img_value, warp_index_value)
    return (result,)


case_1 = {
    "params": [
        tensor_dict([2, 4, 3, 1, 1920], "ND", "float32"),
        tensor_dict([2, 2, 1, 1920], "ND", "float32"),
        tensor_dict([2, 3, 1, 1920], "ND", "float32", is_output=True),
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}

ut_case.add_precision_case("Ascend910A", case_1)

case_2 = {
    "params": [
        tensor_dict([2, 4, 3, 1, 129], "ND", "float16"),
        tensor_dict([2, 2, 1, 129], "ND", "float32"),
        tensor_dict([2, 3, 1, 129], "ND", "float16", is_output=True),
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
}

ut_case.add_precision_case("Ascend910A", case_2)

if __name__ == '__main__':
    user_home_path = os.path.expanduser("~")
    simulator_lib_path = os.path.join(user_home_path, "Ascend/toolkit/tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
