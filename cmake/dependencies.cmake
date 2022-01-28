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

# Ascend mode
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()
message("Search libs under install path ${ASCEND_DIR}")

set(ASCEND_ATC_DIR ${ASCEND_DIR}/compiler/lib64)
set(ASCEND_RUNTIME_DIR ${ASCEND_DIR}/runtime/lib64)

find_module(alog libalog.so ${ASCEND_ATC_DIR})
target_compile_definitions(alog INTERFACE LOG_CPP)
find_module(register libregister.so ${ASCEND_ATC_DIR})
find_module(graph libgraph.so ${ASCEND_ATC_DIR})
find_module(fmk_parser libfmk_parser.so ${ASCEND_ATC_DIR})
find_module(parser_common libparser_common.so ${ASCEND_ATC_DIR})
find_module(fmk_onnx_parser libfmk_onnx_parser.so ${ASCEND_ATC_DIR})
find_module(error_manager liberror_manager.so ${ASCEND_ATC_DIR})
find_module(te_fusion libte_fusion.so ${ASCEND_ATC_DIR})
find_module(platform libplatform.so ${ASCEND_ATC_DIR})
find_module(_caffe_parser lib_caffe_parser.so ${ASCEND_ATC_DIR})
find_module(ascend_protobuf libascend_protobuf.so.3.13.0.0 ${ASCEND_ATC_DIR})
find_module(aicore_utils libaicore_utils.so ${ASCEND_ATC_DIR})
find_module(runtime libruntime.so ${ASCEND_RUNTIME_DIR})
find_module(ascend_hal libascend_hal.so ${ASCEND_RUNTIME_DIR}/stub)
find_module(mmpa libmmpa.so ${ASCEND_ATC_DIR})
find_file(tbe_whl te-0.4.0-py3-none-any.whl ${ASCEND_ATC_DIR})
