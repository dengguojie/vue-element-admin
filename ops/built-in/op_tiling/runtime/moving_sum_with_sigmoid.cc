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
#include "moving_sum_with_sigmoid.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"

using namespace ge;

namespace optiling {
constexpr int64_t OFFSET_NUMS = 3;
constexpr int64_t BATCHSIZE_MAX = 256;
constexpr int64_t INT_BTYES = 4;
constexpr size_t INDEX_OFFSET = 2;
constexpr size_t TWICE = 2;

ge::graphStatus TilingForMovingSumWithSigmoid(gert::TilingContext* context) {
  auto offset_shape = context->GetInputShape(INDEX_OFFSET);
  OPS_CHECK_NULL_WITH_CONTEXT(context, offset_shape);

  const auto& offset_storage_shape = offset_shape->GetStorageShape();
  auto compile_info = reinterpret_cast<const MovingSumWithSigmoidCompileInfo*>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  auto param = context->GetTilingData<MovingSumWithSigmoidTilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, param);

  param->batch_size = offset_storage_shape.GetDim(0) / TWICE;
  OP_TILING_CHECK(param->batch_size > BATCHSIZE_MAX,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "batch_size is over 256."),
                  return ge::GRAPH_FAILED);

  AddWorkspace(context, INT_BTYES * BATCHSIZE_MAX * OFFSET_NUMS);
  context->SetBlockDim(compile_info->core_num);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMovingSumWithSigmoid(gert::TilingParseContext* context) {
  auto compile_info = MutableCompileInfo<MovingSumWithSigmoidCompileInfo>(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OPS_CHECK_NULL_WITH_CONTEXT(context, parsed_object_cinfo);
  const nlohmann::json& vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get vars failed."),
                  return ge::GRAPH_FAILED);
  GetCompileValue(vars, "core_num", compile_info->core_num);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(MovingSumWithSigmoid)
    .Tiling(TilingForMovingSumWithSigmoid)
    .TilingParse<MovingSumWithSigmoidCompileInfo>(TilingPrepareForMovingSumWithSigmoid);
}  // namespace optiling
