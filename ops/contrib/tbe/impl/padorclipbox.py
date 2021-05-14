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


def pad_or_clip_box(tik_instance, proposal_ub, box_list, y_idx, batch_num):
    """
    @param [in]: proposal_ub[16x, 8]
    @param [in]: y_idx, if size > y_idx, clip_x
                        else if size < y_idx, padzeor
    @param [in]: batchnum, support mutibatch
    @param [out]: box_list, [1, y, 4]
    """
    proposallen = proposal_ub.shape[0]
    y_min = tik_instance.Tensor("float16", (proposallen,), name="y_min",
                                scope=tik.scope_ubuf)
    x_min = tik_instance.Tensor("float16", (proposallen,), name="x_min",
                                scope=tik.scope_ubuf)
    y_max = tik_instance.Tensor("float16", (proposallen,), name="y_max",
                                scope=tik.scope_ubuf)
    x_max = tik_instance.Tensor("float16", (proposallen,), name="x_max",
                                scope=tik.scope_ubuf)

    tik_instance.vextract(y_min[0], proposal_ub[0, 0], proposallen // 16, 0)
    tik_instance.vextract(x_min[0], proposal_ub[0, 0], proposallen // 16, 1)
    tik_instance.vextract(y_max[0], proposal_ub[0, 0], proposallen // 16, 2)
    tik_instance.vextract(x_max[0], proposal_ub[0, 0], proposallen // 16, 3)
    real_dim = tik_instance.Scalar("int32")
    temp_n = batch_num - 1

    with tik_instance.if_scope(proposallen >= y_idx):  # clip or unchanged
        real_dim.set_as(y_idx)
    with tik_instance.else_scope():  # pad zero
        real_dim.set_as(proposallen)

    with tik_instance.for_range(0, real_dim) as i:
        box_list[temp_n, i, 0].set_as(y_min[i])
        box_list[temp_n, i, 1].set_as(x_min[i])
        box_list[temp_n, i, 2].set_as(y_max[i])
        box_list[temp_n, i, 3].set_as(x_max[i])
