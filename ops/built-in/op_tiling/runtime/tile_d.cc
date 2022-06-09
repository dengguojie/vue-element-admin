/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "tile_d.h"

#include <nlohmann/json.hpp>
#include <sstream>
#include <cctype>
#include "error_log.h"
#include "vector_tiling_profiling.h"
#include "register/op_impl_registry.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"

namespace optiling {
ge::graphStatus TileDTiling(gert::TilingContext *context) {
  auto parsed_info = reinterpret_cast<const TileDCompileInfo *>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_info);
  const std::vector<int64_t>& tiling_info = parsed_info->tiling_info;

  auto x_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  const auto& shape = x_shape->GetStorageShape();
  std::vector<int64_t> runtime_shape(shape.GetDimNum());
  for (size_t i = 0; i < shape.GetDimNum(); i++) {
    runtime_shape[i] = shape.GetDim(i);
  }

  (void)ScalarToShape(runtime_shape);

  // use assign init vector
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

  vector<vector<int64_t>> inputshapes = {broadcast_input, broadcast_multiples};
  auto src_td = context->GetInputDesc(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, src_td);
  ge::DataType type = src_td->GetDataType();
  OpInfo brc_info(parsed_info->dsl_compile_info.get());
  brc_info.SetInputShape(&inputshapes);
  brc_info.SetInputType(&type);

  bool ret = DoAutoTiling(context, &brc_info);

  return ret ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus TileDParse(gert::TilingParseContext *context) {
  auto compile_info = MutableCompileInfo<TileDCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  compile_info->dsl_compile_info = ParseAutoTiling("TileD", *parsed_object_cinfo);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info->dsl_compile_info);
  // get tiling info
  OP_TILING_CHECK(!GetCompileValue(*parsed_object_cinfo, "tiling_info", compile_info->tiling_info),
                  VECTOR_INNER_ERR_REPORT_TILIING("TileD", "ParseJsonCompileInfo, get tiling_info error"),
                  return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

// register tiling interface of the TileD op.
IMPL_OP(TileD).Tiling(TileDTiling).TilingParse<TileDCompileInfo>(TileDParse);
}  // namespace optiling
