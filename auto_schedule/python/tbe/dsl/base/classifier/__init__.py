#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
shape classifier
"""
from .elewise_classifier import classify as classify_elewise
from .reduce_classifier import classify as classify_reduction
from .norm_classifier import classify as classify_norm
from .softmax_cross_entropy_with_logits_classifier import classify as classify_softmax_cross_entropy_with_logits
from .gather_classifier import classify_gather
from .gather_classifier import classify_gather_nd
from .transpose_classifier import classify as classify_transpose
from .concat_classifier import classify as classify_concat
from .transdata_classifier import classify as classify_transdata
