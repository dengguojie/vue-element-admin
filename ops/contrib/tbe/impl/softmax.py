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


def _ceil_div_offline(value, factor):
    return (value + factor - 1) // factor


def _softmax_data_init(tik_instance, scores_gm, scores_ub, scores_ub_one_dim, neg_ten):
    with tik_instance.for_range(0, scores_gm.shape[0]) as loopi:
        tik_instance.data_move(scores_ub[loopi, 0], scores_gm[loopi, 0], 0, 1, scores_ub_one_dim // 16, 0, 0)
        if scores_ub_one_dim > (scores_gm.shape[1]):
            with tik_instance.for_range(scores_gm.shape[1], scores_ub_one_dim) as iter_score:
                scores_ub[loopi, iter_score] = neg_ten


def softmax(tik_instance, scores_gm, softmax_out):
    """
    function: do softmax calculation, customized shape
    @param [in] scores_gm : score of all class
    @param [in] softmax_out: softmax calculation result
    """
    scores_ub_zero_dim = _ceil_div_offline(scores_gm.shape[0], 16) * 16
    scores_ub_one_dim = _ceil_div_offline(scores_gm.shape[1], 16) * 16
    scores_ub = tik_instance.Tensor("float16", (scores_ub_zero_dim, scores_ub_one_dim), name="scores_ub",
                                    scope=tik.scope_ubuf)
    neg_ten = tik_instance.Scalar("float16", "neg_ten")
    neg_ten.set_as(-10.0)

    _softmax_data_init(tik_instance, scores_gm, scores_ub, scores_ub_one_dim, neg_ten)

    result_ub_ori = tik_instance.Tensor("float16", (scores_ub_zero_dim, scores_ub_one_dim),
                                        scope=tik.scope_ubuf, name="result_ub_ori")
    sum_ub = tik_instance.Tensor("float16", (scores_ub_zero_dim, 1), scope=tik.scope_ubuf, name="sum_ub")

    # sub max of group, avoid fp16 overflow
    # vcgmax,max values of each block & index of max vaules
    max_and_index = tik_instance.Tensor("float16", (2 * scores_ub_zero_dim, 1), name="max", scope=tik.scope_ubuf)
    tik_instance.vcmax(scores_ub_one_dim, max_and_index, scores_ub, scores_ub_zero_dim // 2, 0, 1,
                       scores_ub_one_dim // 16)
    tik_instance.vcmax(scores_ub_one_dim, max_and_index[scores_ub_zero_dim, 0], scores_ub[scores_ub_zero_dim // 2, 0],
                       scores_ub_zero_dim // 2, 0, 1, scores_ub_one_dim // 16)
    max_scalar = tik_instance.Scalar("float16", name="max_scalar")

    # sub max
    with tik_instance.for_range(0, scores_ub_zero_dim) as loopi:
        max_scalar.set_as(max_and_index[loopi * 2])
        max_tensor = tik_instance.Tensor("float16", (scores_ub_one_dim, 1), name="max_tensor", scope=tik.scope_ubuf)
        tik_instance.vector_dup(scores_ub_one_dim, max_tensor, max_scalar, 1, 1, 1)
        tik_instance.vsub(scores_ub_one_dim, scores_ub[loopi, 0], scores_ub[loopi, 0], max_tensor,
                          1, 1, 1, 1, scores_ub_one_dim // 16, scores_ub_one_dim // 16, scores_ub_one_dim // 16)

    # exp
    tik_instance.vexp(scores_ub_one_dim, scores_ub, scores_ub, scores_ub_zero_dim // 2, 1, 1, scores_ub_one_dim // 16,
                      scores_ub_one_dim // 16)
    tik_instance.vexp(scores_ub_one_dim, scores_ub[scores_ub_zero_dim // 2, 0], scores_ub[scores_ub_zero_dim // 2, 0],
                      scores_ub_zero_dim // 2, 1, 1, scores_ub_one_dim // 16, scores_ub_one_dim // 16)

    # sum scores_ub,setting offset_unit to 3 didn't worked
    tik_instance.vcadd(scores_ub_one_dim, sum_ub, scores_ub, scores_ub_zero_dim // 2, 0, 1, scores_ub_one_dim // 16, 0)
    tik_instance.vcadd(scores_ub_one_dim, sum_ub[scores_ub_zero_dim // 2, 0], scores_ub[scores_ub_zero_dim // 2, 0],
                       scores_ub_zero_dim // 2, 0, 1, scores_ub_one_dim // 16, 0)
    # vrec
    repeat_time = scores_ub_zero_dim // 128
    left_mask = scores_ub_zero_dim % 128
    if repeat_time:
        tik_instance.vrec(128, sum_ub, sum_ub, repeat_time, 1, 1, 8, 8)
    if left_mask:
        tik_instance.vrec(left_mask, sum_ub[repeat_time * 128, 0], sum_ub[repeat_time * 128, 0], 1, 1, 1, 8, 8)

    # vmuls
    with tik_instance.for_range(0, scores_gm.shape[0]) as loopi:
        temp_scalar = tik_instance.Scalar("float16", "temp_scalar")
        temp_scalar.set_as(sum_ub[loopi])
        tik_instance.vmuls(scores_ub_one_dim, result_ub_ori[loopi, 0], scores_ub[loopi, 0], temp_scalar, 1, 1, 1,
                           scores_ub_one_dim // 16, scores_ub_one_dim // 16)

    # transpose
    with tik_instance.for_range(0, scores_ub_one_dim // 16) as loopi:
        dst_list = [softmax_out[loopi * 16 + i, 0] for i in range(16)]
        src_list = [result_ub_ori[i, loopi * 16] for i in range(16)]
        tik_instance.vnchwconv(True, True, dst_list, src_list, scores_ub_zero_dim // 16, 1, scores_ub_one_dim)
    # slice column one
    with tik_instance.for_range(0, scores_gm.shape[1]) as loopi:
        tik_instance.data_move(softmax_out[loopi, 0], softmax_out[loopi + 1, 0], 0, 1, scores_ub_zero_dim // 16, 1, 1)
