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
#include "dyn_seq_outer.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"

using namespace ge;

namespace {
constexpr int64_t OFFSET_NUMS = 3;
constexpr int64_t BATCHSIZE_MAX = 256;
constexpr int64_t INT_BYTES = 4;
constexpr size_t INDEX_OFFSET = 2;
const std::string OP_NAME = "DynSeqOuter";
}  // namespace

namespace optiling {
ge::graphStatus TilingForDynSeqOuter(gert::TilingContext *context) {
  auto compile_info = reinterpret_cast<const DynSeqOuterCompileInfo *>(context->GetCompileInfo());
  OPS_CHECK_NULL_WITH_CONTEXT(context, compile_info);

  auto params = context->GetTilingData<DynSeqOuterTilingData>();
  OPS_CHECK_NULL_WITH_CONTEXT(context, params);

  auto seq_len1_shape = context->GetInputShape(INDEX_OFFSET);
  OPS_CHECK_NULL_WITH_CONTEXT(context, seq_len1_shape);
  auto seq_len1_shape_val = seq_len1_shape->GetStorageShape();
  auto x1_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
  auto x1_shape_val = x1_shape->GetStorageShape();

  params->batch_size = seq_len1_shape_val.GetDim(0);
  params->feature_dim = x1_shape_val.GetDim(1);
  OP_LOGD(OP_NAME.c_str(), "op [DynSeqOuterTilingData] : batch_size=%d.", params->batch_size);
  OP_LOGD(OP_NAME.c_str(), "op [DynSeqOuterTilingData] : feature_dim=%d.", params->feature_dim);
  OP_TILING_CHECK(params->batch_size > BATCHSIZE_MAX,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "batch_size is over 256."),
                  return ge::GRAPH_FAILED);

  AddWorkspace(context, INT_BYTES * BATCHSIZE_MAX * OFFSET_NUMS);
  context->SetBlockDim(compile_info->core_num);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDynSeqOuter(gert::TilingParseContext *context) {
  auto compile_info = MutableCompileInfo<DynSeqOuterCompileInfo>(context);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = GetJsonObj(context);
  OP_TILING_CHECK(compile_info == nullptr || parsed_object_cinfo == nullptr,
                  VECTOR_INNER_ERR_REPORT_TILIING(OP_NAME, "compile_info or json_str nullptr!"),
                  return ge::GRAPH_FAILED);
  const nlohmann::json &vars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(vars.empty(), VECTOR_INNER_ERR_REPORT_TILIING(OP_NAME, "get vars failed."), return ge::GRAPH_FAILED);
  optiling::GetCompileValue(vars, "core_num", compile_info->core_num);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynSeqOuter).Tiling(TilingForDynSeqOuter).TilingParse<DynSeqOuterCompileInfo>(TilingPrepareForDynSeqOuter);
}  // namespace optiling