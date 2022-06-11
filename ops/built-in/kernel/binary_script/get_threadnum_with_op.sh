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
# get_threadnum_with_op.sh {op_type}
THREAD_CFG="build_thread_config.env"

function get_thread_num {
  local _thread_num=1
  local _op_type=$1
  _thread_num=`awk -F '=' '/'^${_op_type}='/{print $2;exit}' ${THREAD_CFG}`
  if [ "${_thread_num}" == "" ];then
     return 1
  fi
  return ${_thread_num}
}

function get_thread_num_with_json_config {
  local _thread_num=1
  local _binary_config_full_path=$1
  _thread_num=`cat ${_binary_config_full_path} | grep bin_filename | wc -l`
  if [ "${_thread_num}" == "" ];then
     return 1
  fi
  return ${_thread_num}
}
