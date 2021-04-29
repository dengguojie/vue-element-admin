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
st for Remap
"""
import numpy as np


# pylint: disable=unused-argument,invalid-name,too-many-locals
def remap_offset(image, offset):
    """
    remap_offset
    """
    input_shape = image.shape
    offset_shape = offset.shape
    batch_dim = input_shape[0]
    c_dim = input_shape[3]
    output_shape = [batch_dim, 4, offset_shape[1], offset_shape[2], c_dim]
    output_img = np.random.uniform(0, 0, size=output_shape).astype(np.float32)
    offset_floor = np.floor(offset).astype(np.int32)
    offset_ceil = np.ceil(offset).astype(np.int32)
    for x in range(offset_shape[1]):
        for y in range(offset_shape[2]):
            y_ceil = offset_ceil[0, x, y, 1]
            y_floor = offset_floor[0, x, y, 1]
            x_ceil = offset_ceil[0, x, y, 0]
            x_floor = offset_floor[0, x, y, 0]
            output_img[0, 0, x, y, :] = image[0, y_floor, x_floor, :]
            output_img[0, 1, x, y, :] = image[0, y_floor, x_ceil, :]
            output_img[0, 2, x, y, :] = image[0, y_ceil, x_floor, :]
            output_img[0, 3, x, y, :] = image[0, y_ceil, x_ceil, :]

    return output_img


def remap_offset_new(image, offset):
    """
    remap_offset_new
    """
    input_shape = image.shape
    offset_shape = offset.shape
    batch_dim = input_shape[0]
    c_dim = input_shape[3]
    output_shape = [batch_dim, 4, offset_shape[2], offset_shape[3], c_dim]
    output_img = np.random.uniform(0, 0, size=output_shape).astype(image.dtype)
    offset = offset.astype(np.int32)
    image_1d = image.reshape(-1)
    for x in range(offset_shape[2]):
        for y in range(offset_shape[3]):
            output_img[0, 0, x, y, :] = image_1d[offset[0, 0, x, y]:offset[0, 0, x, y] + 3]
            output_img[0, 1, x, y, :] = image_1d[offset[0, 1, x, y]:offset[0, 1, x, y] + 3]
            output_img[0, 2, x, y, :] = image_1d[offset[0, 2, x, y]:offset[0, 2, x, y] + 3]
            output_img[0, 3, x, y, :] = image_1d[offset[0, 3, x, y]:offset[0, 3, x, y] + 3]
    return output_img


def numpy_resization_compute(input_image_np, input_warp_np):
    """
    numpy_resization_compute
    """
    src_dtype = input_image_np.dtype
    np_batch = input_image_np.shape[0]
    np_c = input_image_np.shape[2]
    np_h = input_image_np.shape[3]
    np_w = input_image_np.shape[4]
    np_area = np_h * np_w
    input_image_resize = input_image_np.reshape([np_batch, 4, np_c, np_area]).astype(np.float32)
    input_warp_resize = input_warp_np.reshape([np_batch, 2, np_area]).astype(np.float32)
    output_img = np.random.uniform(0, 0, size=[np_batch, np_c, np_area]).astype(np.float32)
    for batch_idx in range(np_batch):
        lerp_x_data = input_warp_resize[batch_idx, 0, :]
        lerp_y_data = input_warp_resize[batch_idx, 1, :]
        lerp_x_data_int = lerp_x_data.astype(np.int32)
        lerp_y_data_int = lerp_y_data.astype(np.int32)
        lerp_x_data = lerp_x_data - lerp_x_data_int.astype(np.float32)
        lerp_y_data = lerp_y_data - lerp_y_data_int.astype(np.float32)
        for batch_c in range(np_c):
            top_left = input_image_resize[batch_idx, 0, batch_c, :]
            top_right = input_image_resize[batch_idx, 1, batch_c, :]
            bottom_left = input_image_resize[batch_idx, 2, batch_c, :]
            bottom_right = input_image_resize[batch_idx, 3, batch_c, :]
            top = (top_right - top_left) * lerp_x_data + top_left
            bottom = (bottom_right - bottom_left) * lerp_x_data + bottom_left
            out = (bottom - top) * lerp_y_data + top
            output_img[batch_idx, batch_c, :] = out
        output_img_resize = output_img.reshape([np_batch, np_c, np_h, np_w])
        output_img_resize = output_img_resize.astype(src_dtype)
        return output_img_resize


def remap(image, offset):
    """
    remap
    """
    image = image.astype(np.float32)
    input_shape = image.shape
    offset_shape = offset.shape
    batch_dim = input_shape[0]
    c_dim = input_shape[3]

    # offset trans
    offset_trans = np.transpose(offset, [0, 3, 1, 2])
    offset_x = offset_trans[0:1, 0:1, :, :]
    offset_y = offset_trans[0:1, 1:2, :, :]
    calcu_dtype = np.int32
    offset_x_floor = np.floor(offset_x).astype(calcu_dtype)
    offset_x_ceil = np.ceil(offset_x).astype(calcu_dtype)
    offset_y_floor = np.floor(offset_y).astype(calcu_dtype)
    offset_y_ceil = np.ceil(offset_y).astype(calcu_dtype)
    offset_x_floor_mul = offset_x_floor * np.array([c_dim], calcu_dtype)
    offset_x_ceil_mul = offset_x_ceil * np.array([c_dim], calcu_dtype)
    offset_y_floor_mul = offset_y_floor * np.array([input_shape[2] * c_dim], calcu_dtype)
    offset_y_ceil_mul = offset_y_ceil * np.array([input_shape[2] * c_dim], calcu_dtype)
    offset_x1y1 = offset_x_floor_mul + offset_y_floor_mul
    offset_x2y1 = offset_x_ceil_mul + offset_y_floor_mul
    offset_x1y2 = offset_x_floor_mul + offset_y_ceil_mul
    offset_x2y2 = offset_x_ceil_mul + offset_y_ceil_mul
    offset_concat = np.concatenate([offset_x1y1, offset_x2y1, offset_x1y2, offset_x2y2], 1)

    remap_offset_data = remap_offset_new(image, offset_concat)
    # cmp aicpu
    # remap_offset_data_old = remap_offset(image, offset)
    # error_v = remap_offset_data_old - remap_offset_data
    # print(error_v.reshape(-1)[0:20])
    # print("old_result - new_result max value = ", np.max(error_v))
    # print("old_result - new_result min value = ", np.min(error_v))

    aicore_input_0 = np.transpose(remap_offset_data, [0, 1, 4, 2, 3])
    aicore_input_1 = np.transpose(offset, [0, 3, 1, 2])
    result = numpy_resization_compute(aicore_input_0, aicore_input_1)
    result = result.reshape([batch_dim, c_dim, offset_shape[1], offset_shape[2]])
    result = np.transpose(result, [0, 2, 3, 1])

    return result


def remap_old_aicpu(image, offset):
    """
    remap
    """
    src_dtype = image.dtype
    image = image.astype(np.float32)
    input_shape = image.shape
    offset_shape = offset.shape
    batch_dim = input_shape[0]
    c_dim = input_shape[3]
    remap_offset_data = remap_offset(image, offset)

    aicore_input_0 = np.transpose(remap_offset_data, [0, 1, 4, 2, 3])
    aicore_input_1 = np.transpose(offset, [0, 3, 1, 2])
    result = numpy_resization_compute(aicore_input_0, aicore_input_1).reshape(
        [batch_dim, c_dim, offset_shape[1], offset_shape[2]])
    result = np.transpose(result, [0, 2, 3, 1])
    result = result.astype(src_dtype)
    return result


def calc_expect_func(input_x, input_y, output):
    """
    calc_expect_func
    """
    res = remap(input_x["value"], input_y["value"])
    return [res]
