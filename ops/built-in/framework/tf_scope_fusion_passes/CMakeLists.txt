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
if(NOT ENABLE_TEST)
  include_directories(
          ${REPO_ROOT}/ops/built-in
          ${REPO_ROOT}/inc/external
          ${REPO_ROOT}/metadef/inc/external
          ${REPO_ROOT}/ops
          ${REPO_ROOT}/metadef/inc
          ${REPO_ROOT}/third_party/protobuf/include
          ${REPO_ROOT}/open_source/protobuf/include
  )

  aux_source_directory(. SRCS)

  if("x${SRCS}" STREQUAL "x")
      add_custom_target(${TF_SCOPE_FUSION_PASS_TARGET}
              COMMAND mkdir -p ${TF_SCOPE_FUSION_PASS_TARGET_OUT_DIR}
              COMMAND echo "no source to make lib${TF_SCOPE_FUSION_PASS_TARGET}.so")
      return(0)
  endif()

  set(LIBRARY_OUTPUT_PATH ${TF_SCOPE_FUSION_PASS_TARGET_OUT_DIR})

  add_library(${TF_SCOPE_FUSION_PASS_TARGET} SHARED ${SRCS})
endif()