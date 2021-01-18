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
classifier of shape in reduce
"""

from .known_reduce_classifier import KnownReduceClassifier
from .unknown_reduce_classifier import UnknownReduceClassifier

AXIS = "axis"


def classify(ins: list):
    """
    classify
    :param ins:
    :return:
    """
    for single_input in ins:
        if single_input.get("rel_pos_to_reduce") == AXIS:
            if single_input.get("value"):
                return KnownReduceClassifier(ins).classify()
            else:
                return UnknownReduceClassifier(ins).classify()

    return [ins]
