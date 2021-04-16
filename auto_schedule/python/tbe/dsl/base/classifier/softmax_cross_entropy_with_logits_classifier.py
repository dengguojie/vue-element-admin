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
classifier of shape in oftmax_cross_entropy_with_logits op
"""
from .with_reduce_softmax_cross_entropy_with_logits_classifier import WithReduceSoftmaxCrossEnTropyWithLogitsClassifier


def classify(ins: list, support_reduce: bool = False):
    """
    classify
    :param ins:
    :param support_reduce:
    :return:
    """
    return WithReduceSoftmaxCrossEnTropyWithLogitsClassifier(ins).classify()
