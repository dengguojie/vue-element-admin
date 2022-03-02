#!/bin/bash
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
# bash binary_fuzz_json.sh {op_type} {soc_version} 
set -e
main(){
  echo "[INFO]excute file: $0"
  if [ $# -ne 2 ]; then
    echo "[ERROR]input error"
    echo "[ERROR]bash $0 {op_type} {soc_version}"
    exit 1
  fi
  workdir=$(cd $(dirname $0); pwd)
  op_type=$1
  soc_version=$2
  python3 -c "import binary_json; binary_json.binary_cfg('$op_type','$soc_version')"
}
set -o pipefail
main "$@"|gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'