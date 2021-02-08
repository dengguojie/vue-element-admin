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
classifier of shape in elewise
"""
from typing import Any
from typing import Dict
from typing import Optional

from te import platform as cce
from .broadcast_elewise_classifier import BroadcastElewiseClassifier
from .pure_elewise_classifier import PureElewiseClassifier


def classify(ins: list, support_broadcast: bool = False, extra_params: Optional[Dict[str, Any]] = None):
    """
    classify
    :param ins:
    :param support_broadcast:
    :return:
    """
    if cce.fusion_manager.fusion_manager.get_build_cfg() == "disable":
        return [ins]

    return BroadcastElewiseClassifier(ins, extra_params).classify() if support_broadcast else \
        PureElewiseClassifier(ins).classify()
