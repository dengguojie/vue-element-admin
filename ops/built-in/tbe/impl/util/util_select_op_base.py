# Copyright 2019 Huawei Technologies Co., Ltd
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
util_select_op_base
"""
import json


def get_dynamic_param_in_json(param_desc_list):
    param_dynamic = {}
    for item in param_desc_list:
        param_dict = {}
        param_dict["name"] = item.element.name
        param_dict["dtype"] = item.element.datatype
        param_dict["format"] = item.element.format
        if item.element.unknownshape_format is not None:
            param_dict["unknownshape_format"] = \
                item.element.unknownshape_format
        param_dynamic[item.classify] = param_dict
    param_dynamic_in_json = json.dumps(param_dynamic, indent=4)
    return param_dynamic_in_json


# pylint: disable=locally-disabled,redefined-builtin
def gen_param(classify, name, datatype, format, unknownshape_format=None):
    return ParamItem(classify=classify,
                     element=Element(name=name,
                                     datatype=datatype,
                                     format=format,
                                     unknownshape_format=unknownshape_format))


class Element:
    def __init__(self, name, datatype, format, unknownshape_format):
        self.name = name
        self.datatype = datatype
        self.format = format
        self.unknownshape_format = unknownshape_format


class ParamItem:
    def __init__(self, classify, element):
        self.classify = classify
        self.element = element
