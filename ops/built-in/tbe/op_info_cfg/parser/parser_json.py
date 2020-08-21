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
"""
Convert json files to ini files.
"""
import sys
import json


def load_json(json_file):
    op_info = None
    with open(json_file) as f:
        op_info = json.load(f)
    return op_info


def parse_op_info(json_file, ini_file):
    op_info = load_json(json_file)
    with open(ini_file, "w") as f:
        for op, attrs in op_info.items():
            f.write("[%s]\n" % op)
            for attr, values in attrs.items():
                for key, value in values.items():
                    if not isinstance(value, (str, unicode)):
                        raise ValueError("str or unicode required not %s" % type(value))
                    f.write("%s.%s=%s\n" % (attr, key, value))
            

if __name__ == '__main__':

    json_file = None
    ini_file = None
    if len(sys.argv) >= 2 and len(sys.argv) <= 3:
        json_file = sys.argv[1]
        ini_file = json_file.replace(".json", ".ini")
        if len(sys.argv) == 3:
            ini_file = sys.argv[2]
    else:
        raise ValueError("Two parameter (input json and output ini) required")

    parse_op_info(json_file, ini_file) 
