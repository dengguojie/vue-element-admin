# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
binary_register.py
"""
import tbe.common.register as tbe_register
from impl.util.util_binary import binary_match


def register_binary_match(op_name):
    tbe_register.register_param_generalization(op_name)(binary_match(op_name))


register_binary_match("ResizeBilinearV2")
