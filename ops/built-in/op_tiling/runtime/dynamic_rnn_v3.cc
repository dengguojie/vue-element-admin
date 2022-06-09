/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
#include "dynamic_rnn_v3.h"
#include "runtime2_util.h"
#include "op_tiling_util.h"

using namespace ge;

namespace optiling {
constexpr int DEFAULT_SHAPE_LIST_SIZE = 3;
constexpr int DEFAULT_INDEX_TWO = 2;
constexpr int DEFAULT_RETURN = -2;
constexpr int DEFAULT_PARAS_INPUT_SIZE = 3;
constexpr int DEFAULT_XSHAPE_SIZE = 3;
constexpr int NUM_SIXTEEN = 16;
constexpr int NUM_FIFTEEN = 15;
constexpr int BLOCK_DIM = 32;
constexpr int WORKSPACE_SIZE = 4096;

static int32_t GetRnnV3LibItem(const DynamicRNNV3CompileInfo *compile_info, const gert::Shape x_shape) {
  for (const auto &tune_shape : compile_info->tune_shape_list) {
    if (tune_shape.size() < DEFAULT_SHAPE_LIST_SIZE) {
      VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "tune_shape_list's size is illegal. it's %lu.",
                                      tune_shape.size());
      return DEFAULT_RETURN;
    }
    if ((tune_shape[0] == -1) && (tune_shape[1] == -1)) {
      return static_cast<int32_t>(tune_shape[DEFAULT_INDEX_TWO]);
    }

    if ((tune_shape[0] == x_shape.GetDim(0)) &&
        (((tune_shape[1] + NUM_FIFTEEN) / NUM_SIXTEEN) == x_shape.GetDim(DEFAULT_INDEX_TWO))) {
      return static_cast<int32_t>(tune_shape[DEFAULT_INDEX_TWO]);
    }
  }
  return DEFAULT_RETURN;
}

ge::graphStatus TilingForDynamicRNNV3(gert::TilingContext *context) {
  if (context->GetComputeNodeInputNum() < DEFAULT_PARAS_INPUT_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "input shape error.");
    return ge::GRAPH_FAILED;
  }
  auto x_shape = context->GetInputShape(0);
  if (x_shape == nullptr) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "input_x is null.");
    return ge::GRAPH_FAILED;
  }
  const auto &x_storage_shape = x_shape->GetStorageShape();
  if (x_storage_shape.GetDimNum() < DEFAULT_XSHAPE_SIZE) {
    VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "x_storage_shape error.");
    return ge::GRAPH_FAILED;
  }

  auto compile_info = reinterpret_cast<const DynamicRNNV3CompileInfo *>(context->GetCompileInfo());
  OP_TILING_CHECK(compile_info == nullptr,
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "compile_info is null."),
  return ge::GRAPH_FAILED);

  auto runParams = context->GetTilingData<DynamicRnnV3TilingData>();
  OP_TILING_CHECK(runParams == nullptr,
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "runParams is null."),
  return ge::GRAPH_FAILED);

  runParams->sequenceLength = static_cast<int32_t>(x_storage_shape.GetDim(0));
  runParams->dynamicRnnBatch = static_cast<int32_t>(x_storage_shape.GetDim(DEFAULT_INDEX_TWO));
  runParams->chequeIndex = GetRnnV3LibItem(compile_info, x_storage_shape);
  OP_TILING_CHECK(runParams->chequeIndex == DEFAULT_RETURN,
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "get index fail."),
  return ge::GRAPH_FAILED);
  OP_TILING_CHECK(context->SetTilingKey(runParams->chequeIndex) != ge::GRAPH_SUCCESS,
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "set tiling fail."),
  return ge::GRAPH_FAILED);

  context->SetBlockDim(BLOCK_DIM);
  AddWorkspace(context, WORKSPACE_SIZE);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForDynamicRNNV3(gert::KernelContext *context) {
  auto compile_info = context->GetOutputPointer<DynamicRNNV3CompileInfo>(0);
  auto json_str = context->GetInputStrPointer(0);
  OP_TILING_CHECK(compile_info == nullptr || json_str == nullptr,
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "json is null."),
  return ge::GRAPH_FAILED);

  std::unique_ptr<nlohmann::json> parsed_object_cinfo(new nlohmann::json(nlohmann::json::parse(json_str)));
  OP_TILING_CHECK(parsed_object_cinfo == nullptr,
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "json object is null."),
  return ge::GRAPH_FAILED);

  const nlohmann::json& allVars = (*parsed_object_cinfo)["vars"];
  OP_TILING_CHECK(allVars.empty(),
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "get vars failed."),
  return ge::GRAPH_FAILED);

  compile_info->tune_shape_list = allVars.at("tune_shape_list").get<std::vector<std::vector<int64_t>>>();
  OP_TILING_CHECK(compile_info->tune_shape_list.empty(),
  VECTOR_INNER_ERR_REPORT_TILIING("DynamicRNNV3", "get tune_shape_list failed."),
  return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP(DynamicRNNV3)
    .Tiling(TilingForDynamicRNNV3)
    .TilingParse<DynamicRNNV3CompileInfo>(TilingPrepareForDynamicRNNV3);
}  // namespace optiling