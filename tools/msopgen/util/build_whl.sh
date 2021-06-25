#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

set -e
python_cmd=$1
whl_path=.

echo ${whl_path}
cd ${whl_path}

cp ../op_gen/msopgen.py ${whl_path}
mv ${whl_path}/msopgen.py ${whl_path}/msopgen
${python_cmd} setup.py bdist_wheel
cp -r ${whl_path}/dist/* ${whl_path}/build
rm -rf ${whl_path}/op_gen.egg-info
rm -rf ${whl_path}/dist
rm -rf ${whl_path}/msopgen


