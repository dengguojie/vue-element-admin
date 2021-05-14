# -*- coding:utf-8 -*-
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
from . import get_version

tik, TBE_VERSION = get_version.get_tbe_version()


class ClipWindow:
    x = 0
    y = 0
    w = 0
    h = 0


def clip_to_window(tik_instance, y_min, x_min, y_max, x_max, length,
                   windowsize):
    """
    function: adjust left top and right bottom coordinates to assigned window
    @param [in/out] y_min y coordinate of left top
    @param [in/out] x_min x coordinate of left top
    @param [in/out] y_max y coordinate of right bottom
    @param [in/out] x_max  coordinate of right bottom
    @param [in] length, lenght of vector
    @param [in] windowsize,[x, y, w, h] equals [ 0, 0, 600, 1024]
    @constraints：length must align to 128
    """
    win_y_min = tik_instance.Tensor("float16", (length,), name="win_y_min", scope=tik.scope_ubuf)
    win_x_min = tik_instance.Tensor("float16", (length,), name="win_x_min", scope=tik.scope_ubuf)
    win_y_max = tik_instance.Tensor("float16", (length,), name="win_y_max", scope=tik.scope_ubuf)
    win_x_max = tik_instance.Tensor("float16", (length,), name="win_x_max", scope=tik.scope_ubuf)

    clip_y_min = tik_instance.Scalar("float16")
    clip_x_min = tik_instance.Scalar("float16")
    clip_y_max = tik_instance.Scalar("float16")
    clip_x_max = tik_instance.Scalar("float16")

    clip_y_min.set_as(windowsize.x)
    clip_x_min.set_as(windowsize.y)
    clip_y_max.set_as(windowsize.w)
    clip_x_max.set_as(windowsize.h)

    # duplicate clipwindow shape to vector
    tik_instance.vector_dup(128, win_y_min, clip_y_min, length // 128, 1, 8, 0)
    tik_instance.vector_dup(128, win_x_min, clip_x_min, length // 128, 1, 8, 0)
    tik_instance.vector_dup(128, win_y_max, clip_y_max, length // 128, 1, 8, 0)
    tik_instance.vector_dup(128, win_x_max, clip_x_max, length // 128, 1, 8, 0)

    # y_min_clipped equals to tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
    tik_instance.vmin(128, y_min, y_min, win_y_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(128, y_min, y_min, win_y_min, length // 128, 1, 1, 1, 8, 8, 8, 0)

    # y_max_clipped equals to tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
    tik_instance.vmin(128, y_max, y_max, win_y_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(128, y_max, y_max, win_y_min, length // 128, 1, 1, 1, 8, 8, 8, 0)

    # x_min_clipped equals to tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
    tik_instance.vmin(128, x_min, x_min, win_x_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(128, x_min, x_min, win_x_min, length // 128, 1, 1, 1, 8, 8, 8, 0)

    # x_max_clipped equals to tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
    tik_instance.vmin(128, x_max, x_max, win_x_max, length // 128, 1, 1, 1, 8, 8, 8, 0)
    tik_instance.vmax(128, x_max, x_max, win_x_min, length // 128, 1, 1, 1, 8, 8, 8, 0)
