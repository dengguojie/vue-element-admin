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
from typing import Optional
from typing import Dict
from typing import Any


class Mode(Enum):
    """
    Mode
    """
    NONE = auto()
    ELEWISE = auto()
    ELEWISE_WITH_BROADCAST = auto()
    REDUCE = auto()


MODE_TO_STRING_MAP = {
    Mode.NONE: "",
    Mode.ELEWISE: "elewise",
    Mode.ELEWISE_WITH_BROADCAST: "broadcast",
    Mode.REDUCE: "reduce"
}


def classify(ins: list, mode: Mode = Mode.NONE, extra_params: Optional[Dict[str, Any]] = None):
    """
    classify
    :param ins:
    :param mode:
    :param extra_params: must include keepdims when mode is reduce
    :return:
    """
    import tbe
    return tbe.dsl.classify(ins, MODE_TO_STRING_MAP.get(mode), extra_params)
