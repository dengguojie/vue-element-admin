"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

keep_ration_resize_bilinear
"""
from te.utils.op_utils import *
from impl.resize_bilinear_v2_d import resize_bilinear_v2_d


@check_op_params(REQUIRED_INPUT, REQUIRED_OUTPUT, REQUIRED_ATTR_INT, REQUIRED_ATTR_INT,
                 OPTION_ATTR_BOOL, OPTION_ATTR_BOOL, KERNEL_NAME)
def keep_ration_resize_bilinear(images, y, min_dimension, max_dimension,
                                align_corners=False, half_pixel_centers=False,
                                kernel_name="keep_ration_resize_bilinear"):
    """keep_ration_resize_bilinear, Keep the HW ratio and do resize

    Parameters
    ----------
    images: dict
        dict info of images value, must include the keys(shape and dtype).
        and shape will be 5HD
    y: dict
        dict info of output value
    min_dimension: int
        the min_dimension
    max_dimension: int
        the max_dimension
    align_corners: bool
        If true, the centers of the 4 corner pixels
        of the input and output tensors are aligned
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is "keep_ration_resize_bilinear"

    returns
    -------
    None
    """
    image_shape = images.get("shape")
    image_h = image_shape[2]
    image_w = image_shape[3]
    min_shape_dim = min(image_h, image_w)
    max_shape_dim = max(image_h, image_w)

    # process min_dimension
    scale = min_shape_dim / min_dimension
    min_new_shape = [round(image_h / scale), round(image_w / scale)]
    min_new_shape_max_dim = max(min_new_shape)

    # process max_dimension
    scale = max_shape_dim / max_dimension
    max_new_shape = [round(image_h / scale), round(image_w / scale)]

    if min_new_shape_max_dim > max_dimension:
        size_for_bilinear = max_new_shape
    else:
        size_for_bilinear = min_new_shape

    resize_bilinear_v2_d(images, y, size_for_bilinear,
                         align_corners, half_pixel_centers, kernel_name)

