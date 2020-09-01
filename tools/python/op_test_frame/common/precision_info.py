# Copyright 2020 Huawei Technologies Co., Ltd
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

from op_test_frame.common import op_status


class PrecisionStandard:
    def __init__(self, rtol, atol, max_atol=None, precision_type="percent"):
        """
        init methos
        :param rtol: The relative tolerance parameter
        :param atol: The absolute tolerance parameter
        :param max_atol: The max absolute tolerance parameter
        """
        self.precision_type = precision_type
        self.rtol = rtol
        self.atol = atol
        self.max_atol = max_atol

    def to_json_obj(self):
        return {
            "precision_type": self.precision_type,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_atol": self.max_atol
        }

    @staticmethod
    def parse_json_obj(json_obj):
        if json_obj:
            return PrecisionStandard(json_obj['rtol'], json_obj['atol'], json_obj['max_atol'],
                                     json_obj['precision_type'])
        else:
            return None


class PrecisionCompareResult:
    def __init__(self, status, err_msg=None):
        self.status = status
        self.err_msg = err_msg

    def is_success(self):
        return self.status == op_status.SUCCESS

    def to_json_obj(self):
        return {
            "status": self.status,
            "err_msg": self.err_msg
        }

    @staticmethod
    def parse_json_obj(json_obj):
        return PrecisionCompareResult(json_obj['status'], json_obj['err_msg'])


def get_default_standard(dtype):
    if dtype == "float16":
        return PrecisionStandard(0.001, 0.001, 0.1)
    elif dtype == "float32":
        return PrecisionStandard(0.0001, 0.0001, 0.01)
    elif dtype == "int8" or dtype == "uint8":
        return PrecisionStandard(0.001, 1, 1, precision_type="absolute")
    else:
        return PrecisionStandard(0.001, 0.001, 0.1)
