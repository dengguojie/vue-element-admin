/* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
 * \file batchnorm_preprocess_fusion_pass.cc
 * \brief
 */
#include "batchnorm_preprocess_fusion_pass.h"
#include <vector>
#include <memory>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {

static const string PATTERN_BATCHNORM = "batchnorm";
static const string PASS_OP_TYPE_BATCHNORM = "BatchNorm";

vector<FusionPattern*> BatchNormPreprocessFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormPreprocessFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchNormPreprocessFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_BATCHNORM, {PASS_OP_TYPE_BATCHNORM}).SetOutput(PATTERN_BATCHNORM);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormPreprocessFusionPass pattern end");
  return patterns;
}

Status BatchNormPreprocessFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormPreprocessFusionPass fusion begin");
  ge::NodePtr batchNormNode = GetNodeFromMapping(PATTERN_BATCHNORM, mapping);

  FUSION_PASS_CHECK(batchNormNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "batchnorm is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr bn_desc = batchNormNode->GetOpDesc();
  size_t outputs_size = bn_desc->GetOutputsSize();
  if (outputs_size <= 5) {
    return NOT_CHANGED;
  }

  OpDescPtr const_op_desc = std::make_shared<ge::OpDesc>("Constant", "Constant");
  
  vector<int64_t> dims{1};
  GeShape shape(dims);
  GeTensorDesc const_output_desc(shape, FORMAT_ND, DT_FLOAT);
  const_output_desc.SetOriginShape(shape);
  const_output_desc.SetOriginFormat(FORMAT_ND);
  const_op_desc->AddOutputDesc("y", const_output_desc);

  std::vector<int64_t> values {0};
  GeTensorDesc data_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
  ge::GeTensorPtr const_value = std::make_shared<ge::GeTensor>(data_desc, reinterpret_cast<uint8_t *>(values.data()),
                                                               sizeof(float)*values.size());
  AttrUtils::SetTensor(const_op_desc, "value", const_value);

  const_op_desc->SetType("Constant");
  NodePtr const_node = graph.AddNode(const_op_desc);

  // last output
  auto out_anchor = batchNormNode->GetOutDataAnchor(5);
  auto const_node_out_anchor = const_node->GetOutDataAnchor(0);
  for (auto out_anchor_peer_in : out_anchor->GetPeerInDataAnchors()) {
    ge::GraphUtils::RemoveEdge(out_anchor, out_anchor_peer_in);
    ge::GraphUtils::AddEdge(const_node_out_anchor, out_anchor_peer_in);
  }

  out_anchor->UnlinkAll();

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchNormPreprocessFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("BatchNormPreprocessFusionPass", BUILT_IN_GRAPH_PASS, BatchNormPreprocessFusionPass);
}  // namespace fe
