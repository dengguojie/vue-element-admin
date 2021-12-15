/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file vector_tiling.cc
 * \brief tiling function of vector ops
 */
#include "vector_tiling.h"
#include "tiling_handler.h"

namespace optiling {
const std::vector<vector<int32_t>> OpInfo::dummy_variable;

std::shared_ptr<AutoTilingHandler> CreateAutoTilingHandler(const std::string& op_type, const std::string& pattern,
                                                           const nlohmann::json& parsed_compile_info) {
  OP_LOGI(op_type.c_str(), "Entering AutoTiling Compile Info Parser for pattern %s", pattern.c_str());
  try {
    if (pattern == "CommReduce") {
      return CreateReduceTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "ElemWise") {
      return CreateElewiseTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Broadcast") {
      return CreateBroadcastTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Norm") {
      return CreateNormTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Gather") {
        return CreateGatherTilingHandler(op_type, pattern, parsed_compile_info);
    }else if (pattern == "Transpose") {
      return CreateTransposeDslTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Transdata") {
      return CreateTransdataTilingHandler(op_type, pattern, parsed_compile_info);
    } else if (pattern == "Concat") {
      return CreateConcatDslTilingHandler(op_type, pattern, parsed_compile_info);
    } else {
      OP_LOGE(op_type.c_str(), "Pattern %s is not supported by AutoTiling Compile Info Parser", pattern.c_str());
      return std::shared_ptr<AutoTilingHandler>(nullptr);
    }
  } catch (...) {
    OP_LOGE(op_type.c_str(), "Unknown Exception encountered when parsing Compile Info of pattern %s", pattern.c_str());
    return std::shared_ptr<AutoTilingHandler>(nullptr);
  }
}
}  // namespace optiling
