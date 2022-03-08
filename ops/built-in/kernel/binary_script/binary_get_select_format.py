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
import sys
import json
import stat
import os
from importlib import import_module
from pathlib import Path
from op_test_frame.ut import OpUT
def get_select_format(self):

    opp_path = Path(os.environ.get("ASCEND_OPP_PATH", "/usr/local/Ascend/opp"))
    tbe_path = opp_path.joinpath("/op_impl/built-in/ai_core/tbe/")
    sys.path.append(tbe_path)
    select_json = "../binary_config/" + op_type + "-" + soc + ".json"
    func = getattr(import_module(select_format),'op_select_format')
    input_list = [{"shape": [-2], "format": "ND", "ori_shape":[-2], "ori_format":"ND"}] * int(tensor_num)
    dynamic_json = func(*input_list)
    dynamic_json = json.loads(dynamic_json)

    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(select_json, flags, modes), 'w') as sl:
        json.dump(dynamic_json, sl, indent=4)

if __name__ == '__main__':

    soc_version = ["ascend310", "ascend320", "ascend610", "ascend615", "ascend710",
        "ascend910", "ascend920", "hi3796cv300cs", "hi3796cv300es", "sd3403"]
    ut_soc_version = ["Ascend310", "Ascend320", "Ascend610", "Ascend615", "Ascend710",
        "Ascend910A", "Ascend920A", "Hi3796CV300CS", "Hi3796CV300ES", "SD3403"]

    soc_version_map = dict(zip(soc_version, ut_soc_version))
    soc = sys.argv[1]
    op_type = sys.argv[2]
    tensor_num = sys.argv[3]
    select_format = sys.argv[4]
    ut_case = OpUT(op_type, None, None)
    ut_case.add_cust_test_func(test_func=get_select_format)
    ut_case.run(soc_version_map.get(soc))
