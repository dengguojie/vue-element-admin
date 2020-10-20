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
from enum import Enum
from enum import auto
from .classifier import ElewiseClassifier
from .classifier import ReduceClassifier


class Mode(Enum):
    """
    Mode
    """
    NONE = auto()
    ELEWISE = auto()
    ELEWISE_WITH_BROADCAST = auto()
    REDUCE = auto()


def classify(ins: list, mode: Mode = Mode.NONE):
    """
    :param ins:
    :param mode:
    :return:
    """
    if mode == Mode.NONE:
        return [ins]
    if mode == Mode.ELEWISE:
        return ElewiseClassifier(ins, support_broadcast=False).classify()
    if mode == Mode.ELEWISE_WITH_BROADCAST:
        return ElewiseClassifier(ins, support_broadcast=True).classify()
    if mode == Mode.REDUCE:
        return ReduceClassifier(ins).classify()

    return [ins]
