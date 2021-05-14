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


def softmax(tik_instance, scores_left, scores_right, result_ub):
    """
    function: do softmax calculation, customized shape
    @param [in] scores_left : score of first class
    @param [in] scores_right : score of second class
    @param [in] result_ub: softmax calculation result
    """
    length = scores_left.shape[0]
    scores_left_fp32 = tik_instance.Tensor("float32", (length,), scope=tik.scope_ubuf, name="scores_left_fp32")
    scores_right_fp32 = tik_instance.Tensor("float32", (length,), scope=tik.scope_ubuf, name="scores_right_fp32")
    sum_scores = tik_instance.Tensor("float32", (length,), scope=tik.scope_ubuf, name="sum_scores")
    rec_scores = tik_instance.Tensor("float32", (length,), scope=tik.scope_ubuf, name="rec_scores")
    left_ub = tik_instance.Tensor("float16", (length,), scope=tik.scope_ubuf, name="left_ub")
    right_ub = tik_instance.Tensor("float16", (length,), scope=tik.scope_ubuf, name="right_ub")
    tik_instance.vexp(128, left_ub, scores_left, length // 128, 1, 1, 8, 8)
    tik_instance.vexp(128, right_ub, scores_right, length // 128, 1, 1, 8, 8)
    tik_instance.vconv(64, "none", scores_left_fp32, left_ub, length // 64, 1, 1, 8, 4)
    tik_instance.vconv(64, "none", scores_right_fp32, right_ub, length // 64, 1, 1, 8, 4)
    tik_instance.vadd(64, sum_scores, scores_left_fp32, scores_right_fp32, length // 64, 1, 1, 1, 8, 8, 8)
    tik_instance.vrec(64, rec_scores, sum_scores, length // 64, 1, 1, 8, 8)
    tik_instance.vmul(64, sum_scores, rec_scores, scores_right_fp32, length // 64, 1, 1, 1, 8, 8, 8)
    tik_instance.vconv(64, "none", result_ub, sum_scores, length // 64, 1, 1, 4, 8)
