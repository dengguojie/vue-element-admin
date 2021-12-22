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
 * \file tile_d.cpp
 * \brief
 */
#include <nlohmann/json.hpp>
#include <sstream>
#include <cctype>
#include "vector_tiling.h"
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling_util.h"

namespace optiling {
struct TileDCompileInfo {
  std::shared_ptr<AutoTilingHandler> tiling_handler;
  std::vector<int64_t> tiling_info;
};

bool TileDTiling(const std::string& op_type, const ge::Operator& op_paras, const TileDCompileInfo& parsed_info,
                 utils::OpRunInfo& run_info) {
  PROFILING_TILING_INIT(op_type.c_str());
  auto operator_info = ge::OpDescUtils::GetOpDescFromOperator(op_paras);
  OP_TILING_CHECK(operator_info == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "GetOpDescFromOperator return nullptr!"), return false);
  auto input_desc = operator_info->MutableInputDesc(0);
  OP_TILING_CHECK(input_desc == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "get input 0 opdesc failed"), return false);
  const std::vector<int64_t>& tiling_info = parsed_info.tiling_info;

  std::vector<int64_t> runtime_shape = input_desc->MutableShape().GetDims();
  ScalarToShape(runtime_shape);
  PROFILING_TILING_AFTER_GET_SHAPE_REG();

  // use assign init vector
  V_CHECK_GT(tiling_info.size(), 0, VECTOR_INNER_ERR_REPORT_TILIING(op_type, "tiling_info index out of range"),
             return false);
  size_t shape_size = (tiling_info.size() - tiling_info[0] - 1) / 2;
  std::vector<int64_t> broadcast_input(shape_size);
  std::vector<int64_t> broadcast_multiples(shape_size);
  broadcast_input.assign(tiling_info.begin() + tiling_info[0] + 1, tiling_info.end() - shape_size);
  broadcast_multiples.assign(tiling_info.end() - shape_size, tiling_info.end());
  int64_t count = 1;
  for (size_t i = 0; i < shape_size; i++) {
    if (broadcast_input[i] == -1) {
      broadcast_input[i] = broadcast_multiples[i] = runtime_shape[tiling_info[count]];
      count++;
    }
    if (tiling_info[0] + 1 == count) {
      break;
    }
  }
  PROFILING_TILING_AFTER_GET_COMPILE_INFO_REG();

  vector<vector<int64_t>> inputshapes = {broadcast_input, broadcast_multiples};
  ge::DataType type = input_desc->GetDataType();
  OpInfo eletwise_info(inputshapes, type);
  PROFILING_TILING_AFTER_CALCU_TILING_REG();

  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "parsed_info.tiling_handler nullptr, error!"), return false);
  bool ret = parsed_info.tiling_handler->DoTiling(op_paras, run_info, eletwise_info);
  PROFILING_TILING_END();
  return ret;
}

static bool ParseJsonCompileInfo(const std::string& op_type, const nlohmann::json& compile_info,
                                 TileDCompileInfo& parsed_info) {
  parsed_info.tiling_handler = CreateAutoTilingHandler(op_type, PATTERN_BROADCAST, compile_info);
  OP_TILING_CHECK(parsed_info.tiling_handler == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "CreateAutoTilingHandler return nullptr"), return false);
  // get core_num value
  OP_TILING_CHECK(!GetCompileValue(compile_info, "tiling_info", parsed_info.tiling_info),
                  VECTOR_INNER_ERR_REPORT_TILIING(op_type, "ParseJsonCompileInfo, get tiling_info error"),
                  return false);
  return true;
}

// register tiling interface of the TileD op.
REGISTER_OP_TILING_V3_CUSTOM(TileD, TileDTiling, ParseJsonCompileInfo, TileDCompileInfo);
}  // namespace optiling
