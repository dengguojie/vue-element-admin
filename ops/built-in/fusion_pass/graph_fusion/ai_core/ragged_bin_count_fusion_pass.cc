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
 * \file ragged_bin_count_fusion_pass.cpp
 * \brief RaggedBinCount fusion pass(RaggedBinCount --> RaggedBinCount + Minimun)
 */
#include "ragged_bin_count_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/platform_info.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/fusion_common/fusion_turbo.h"
#include "securec.h"
#include "tbe_fusion_pass_util.h"
#include "op_common_util.h"

using namespace ge;
namespace fe {
static const std::string PATTERN_FUSED_NODE = "FusedRaggedBinCount";
static const std::string INSERT_NODE_NAME = "MinimumInRaggedBinCount";
static const int32_t BASE_OUTPUT_INDEX_0 = 0;
static const int32_t MIN_INPUT_INDEX_0 = 0;
static const int32_t MIN_INPUT_INDEX_1 = 1;

vector<FusionPattern*> RaggedBinCountFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("RaggedBinCountFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSED_NODE, {"RaggedBinCount"}).SetOutput(PATTERN_FUSED_NODE);
  patterns.push_back(pattern);
  return patterns;
}

bool RaggedBinCountFusionPass::IsMatch(const ge::NodePtr& ragged_bin_count_node) {
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != GRAPH_SUCCESS,
      OP_LOGD(FUSED_OP_TYPE, "Fail to get platform info."), return false);

  FUSION_PASS_CHECK(platform_info.str_info.short_soc_version != "Ascend910",
                    OP_LOGD(FUSED_OP_TYPE, "Currently only supports Ascend910, but now is %s.",
                            platform_info.str_info.short_soc_version),
                    return false);

  Operator op_ragged_bin_count = ge::OpDescUtils::CreateOperatorFromNode(ragged_bin_count_node);

  bool binary_output;
  FUSION_PASS_CHECK(GRAPH_SUCCESS != op_ragged_bin_count.GetAttr("binary_output", binary_output),
                    OP_LOGD(FUSED_OP_TYPE, "Get attr binary_output failed."), return false);

  FUSION_PASS_CHECK(!binary_output, OP_LOGD(FUSED_OP_TYPE, "Ragged Bin Count op does not need fusion with Minimum."),
                    return false);

  return true;
}

Status RaggedBinCountFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  FusionTurbo rbc_turbo(graph);
  ge::NodePtr fused_ragged_bin_count_node = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  FUSION_PASS_CHECK(
      fused_ragged_bin_count_node == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "The FusedRaggedBinCount is null, fusion is not supported."),
      return NOT_CHANGED);

  ge::OpDescPtr fused_ragged_bin_count_desc = fused_ragged_bin_count_node->GetOpDesc();

  FUSION_PASS_CHECK(!IsMatch(fused_ragged_bin_count_node),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                                   "RaggedBinCountFusionPass is not matched, fusion is not supported."),
                    return NOT_CHANGED);

  ge::NodePtr minimum_node = rbc_turbo.InsertNodeAfter(INSERT_NODE_NAME, "Minimum", fused_ragged_bin_count_node,
                                                       BASE_OUTPUT_INDEX_0, MIN_INPUT_INDEX_0);

  ge::GeTensorDesc splits_desc = fused_ragged_bin_count_desc->GetInputDesc(0);
  ge::GeShape splits_shape = splits_desc.GetShape();
  const int32_t splits_num = splits_shape.GetShapeSize();

  auto ragged_bin_count_op = OpDescUtils::CreateOperatorFromNode(fused_ragged_bin_count_node);
  std::vector<int64_t> size_data;
  FUSION_PASS_CHECK(
      !TbeFusionPassUtil::GetConstIntData(ragged_bin_count_op, "size", size_data),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Get size data from %s failed, fusion is not supported.",
                                     fused_ragged_bin_count_node->GetName().c_str()),
      return NOT_CHANGED);

  int32_t output_num = (splits_num - 1) * size_data[0];

  unique_ptr<float[]> value(new (std::nothrow) float[output_num]);
  auto data_ptr = value.get();
  for (int32_t i = 0; i < output_num; i++) {
    data_ptr[i] = 1;
  }
  WeightInfo weight_info = {ge::GeShape({splits_num - 1, size_data[0]}), ge::DT_FLOAT, ge::FORMAT_ND, value.get()};

  ge::NodePtr fused_node = rbc_turbo.AddWeight(minimum_node, MIN_INPUT_INDEX_1, weight_info);

  fusion_nodes.push_back(minimum_node);
  fusion_nodes.push_back(fused_node);

  return SUCCESS;
}

REGISTER_PASS("RaggedBinCountFusionPass", BUILT_IN_GRAPH_PASS, RaggedBinCountFusionPass);
}  // namespace fe