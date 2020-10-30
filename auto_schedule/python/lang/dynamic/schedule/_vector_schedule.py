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
Basic structure of schedule factory for all pure vector_core operator
Inherit VectorSchedule() and implement all abstract method to build your own schedule
"""
from abc import abstractmethod
from te import tvm
from te import platform as cceconf


class VectorSchedule:
    def __init__(self):
        pass

    def do_schedule(self):
        pass
