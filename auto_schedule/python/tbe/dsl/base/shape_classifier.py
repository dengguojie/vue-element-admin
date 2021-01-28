# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
classifier of shape
"""
from tbe.dsl.base.classifier import classify_elewise
from tbe.dsl.base.classifier import classify_reduction


ELEWISE = "elewise"
BROADCAST = "broadcast"
REDUCE = "reduce"


CLASSIFY_SAME_PATTERN_MAP = {
    "ElemWise": ELEWISE,
    "Broadcast": BROADCAST,
    "CommReduce": REDUCE
}


def classify(ins: list, mode: str):
    """
    classify
    :param ins:
    :param mode:
    :return:
    """
    mode = CLASSIFY_SAME_PATTERN_MAP.get(mode, mode)
    if mode == ELEWISE:
        return classify_elewise(ins, support_broadcast=False)
    if mode == BROADCAST:
        return classify_elewise(ins, support_broadcast=True)
    if mode == REDUCE:
        return classify_reduction(ins)

    return [ins]
