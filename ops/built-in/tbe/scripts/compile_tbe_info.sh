#!/bin/bash
# Copyright 2018 Huawei Technologies Co., Ltd
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

pwd
cd ./cann/ops/built-in/tbe/op_info_cfg/ai_core
echo "-----------------------sub path = $2------------------------"
path1=./$2

if [[ "$HI_PYTHON" == "" ]]; then
  HI_PYTHON="python3.7"
fi

files=$(ls $path1)
for filename in $files
do
 if [ "${filename##*.}" = "ini" ];then
  cd $path1
  cp ../../parser/parser_ini.py ./
  $HI_PYTHON parser_ini.py $filename
  echo "----------------convert aic_c_100 ini to json------------------"
  sed -i "s/needcompile/needCompile/g" *.json
  sed -i "s/paramtype/paramType/g" *.json
 fi
done

mv *.json ../../../../../../../out/$1/host/obj/
