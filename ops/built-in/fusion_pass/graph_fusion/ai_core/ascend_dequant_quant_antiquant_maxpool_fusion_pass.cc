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

/*!
 * \file ascend_dequant_quant_antiquant_maxpool_fusion_pass.cc
 * \brief dequant_quant_antiquant_maxpool fusion pass
 */
#include "ascend_dequant_quant_antiquant_maxpool_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "pattern_fusion_util.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "fp16_t.hpp"

namespace fe {
static const string DEQUANT = "AscendDequant";
static const string QUANT = "AscendQuant";
static const string ANTIQUANT = "AscendAntiQuant";
static const string MAXPOOL = "MaxPool";
static const string PATTERN_DEQUANT = "AscendDequant";
static const string PATTERN_QUANT = "AscendQuant";
static const string PATTERN_ANTIQUANT = "AscendAntiQuant";
static const string PATTERN_MAXPOOL = "MaxPool";
static const string PATTERN_INPUT = "input";

vector<FusionPattern*> AscendDequantQuantAntiquantMaxpoolFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define AscendDequantQuantAntiquantMaxpoolFusionPass pattern begin");
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("AscendDequantQuantAntiquantMaxpoolFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_DEQUANT, {DEQUANT})
      .AddOpDesc(PATTERN_QUANT, {QUANT})
      .AddOpDesc(PATTERN_ANTIQUANT, {ANTIQUANT})
      .AddOpDesc(PATTERN_MAXPOOL, {MAXPOOL})
      .AddOpDesc(PATTERN_INPUT)
      .SetInputs(PATTERN_DEQUANT, {PATTERN_INPUT})
      .SetInputs(PATTERN_QUANT, {PATTERN_DEQUANT})
      .SetInputs(PATTERN_ANTIQUANT, {PATTERN_QUANT})
      .SetInputs(PATTERN_MAXPOOL, {PATTERN_ANTIQUANT})
      .SetOutput(PATTERN_MAXPOOL);

  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define AscendDequantQuantAntiquantMaxpoolFusionPass pattern end");
  return patterns;
}

Status AscendDequantQuantAntiquantMaxpoolFusionPass::CheckPeerAllInDataAnchors(const ge::OutDataAnchorPtr& outputAnchor,
                                                                               const size_t& expectedNum) {
  FUSION_PASS_CHECK(outputAnchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "outputAnchor must not be null"),
                    return PARAM_INVALID);
  if (outputAnchor->GetPeerInDataAnchors().size() == expectedNum) {
    return SUCCESS;
  }
  return FAILED;
}

Status AscendDequantQuantAntiquantMaxpoolFusionPass::IsMatch(ge::NodePtr& dequantNode, ge::NodePtr& quantNode,
                                                             ge::NodePtr& antiqNode, ge::NodePtr& maxpoolNode) {
  FUSION_PASS_CHECK(dequantNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "dequantNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(quantNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "quantNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(antiqNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "antiqNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maxpoolNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "maxpoolNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(CheckPeerAllInDataAnchors(dequantNode->GetOutDataAnchor(0), 1) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", dequantNode->GetName().c_str()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckPeerAllInDataAnchors(quantNode->GetOutDataAnchor(0), 1) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", quantNode->GetName().c_str()),
      return NOT_CHANGED);
  FUSION_PASS_CHECK(
      CheckPeerAllInDataAnchors(antiqNode->GetOutDataAnchor(0), 1) != SUCCESS,
      OP_LOGI(FUSED_OP_TYPE.c_str(), "%s contains more than one peer input", antiqNode->GetName().c_str()),
      return NOT_CHANGED);

  ge::DataType deq_out_dtype = dequantNode->GetOpDesc()->GetOutputDesc(0).GetDataType();
  ge::DataType antiq_out_dtype = antiqNode->GetOpDesc()->GetOutputDesc(0).GetDataType();
  if (deq_out_dtype != antiq_out_dtype) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "the output dtype of %s and %s is not equal.", dequantNode->GetName().c_str(),
            antiqNode->GetName().c_str());
    return NOT_CHANGED;
  }
  ge::Operator maxpool_op = ge::OpDescUtils::CreateOperatorFromNode(maxpoolNode);
  std::vector<int64_t> kernel_size;
  ge::AscendString op_name;
  (void) maxpool_op.GetName(op_name);
  if (maxpool_op.GetAttr("ksize", kernel_size) != GRAPH_SUCCESS) {
    OP_LOGI(op_name.GetString(), "Get attr ksize of maxpool operator failed.");
    return NOT_CHANGED;
  }
  std::vector<int64_t> strides;
  if (maxpool_op.GetAttr("strides", strides) != GRAPH_SUCCESS) {
    OP_LOGI(op_name.GetString(), "Get attr strides of maxpool operator failed.");
    return NOT_CHANGED;
  }
  // 限制maxpool的融合：ksize全部为1，strides全部为1.
  for (auto i : kernel_size) {
    if (i != 1) {
      OP_LOGI(op_name.GetString(), "the value of kernel size is not equal to 1.");
      return NOT_CHANGED;
    }
  }
  for (auto j : strides) {
    if (j != 1) {
      OP_LOGI(op_name.GetString(), "the value of strides is not equal to 1.");
      return NOT_CHANGED;
    }
  }

  size_t quant_in_nodes_size = ge::OpDescUtils::GetNonConstInputsSize(quantNode);
  if (quant_in_nodes_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "quant node need 1 inputs,but now is %d.", (int)quant_in_nodes_size);
    return NOT_CHANGED;
  }
  size_t antiq_in_nodes_size = ge::OpDescUtils::GetNonConstInputsSize(antiqNode);
  if (antiq_in_nodes_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "aintiquant node need 1 inputs,but now is %d.", (int)antiq_in_nodes_size);
    return NOT_CHANGED;
  }
  size_t maxpool_in_nodes_size = ge::OpDescUtils::GetNonConstInputsSize(maxpoolNode);
  if (maxpool_in_nodes_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "maxpool node need 1 inputs,but now is %d.", (int)maxpool_in_nodes_size);
    return NOT_CHANGED;
  }
  // 通过inAnchor拿到outAnchor，再通过outAnchor拿到对应的子图node，再获取到const输入
  auto deq_in_anchors = dequantNode->GetAllInDataAnchors();
  std::vector<ge::GeTensorPtr> deqWeights;
  for (auto& deq_in_anchor : deq_in_anchors) {
    auto deq_out_anchor = deq_in_anchor->GetPeerOutAnchor();
    if (deq_out_anchor == nullptr)
      continue;

    auto deq_in_node = deq_out_anchor->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(deq_in_node);
    if (type == "Const" || type == "Constant") {
      ge::OpDescPtr deqOpDesc = deq_in_node->GetOpDesc();
      ge::GeTensorPtr weight = nullptr;
      ge::AttrUtils::MutableTensor(deqOpDesc, ge::ATTR_NAME_WEIGHTS, weight);
      deqWeights.push_back(weight);
    }
  }
  size_t deq_in_nodes_size = dequantNode->GetOpDesc()->GetInputsSize();
  size_t deq_nonconst_size = deq_in_nodes_size - deqWeights.size();
  if (deq_nonconst_size != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "dequant node need 1 inputs,but now is %d.", (int)deq_nonconst_size);
    return NOT_CHANGED;
  }

  ge::NodePtr deqInputNode = dequantNode->GetInDataNodes().at(0);
  FUSION_PASS_CHECK(deqInputNode == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "deqInputNode is null, fusion failed."),
                    return PARAM_INVALID);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "graph node check is done. start to run fusion.");
  return SUCCESS;
}

Status AscendDequantQuantAntiquantMaxpoolFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                            vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define AscendDequantQuantAntiquantMaxpoolFusionPass fusion begin");
  ge::NodePtr dequantNode = GetNodeFromMapping(PATTERN_DEQUANT, mapping);
  ge::NodePtr quantNode = GetNodeFromMapping(PATTERN_QUANT, mapping);
  ge::NodePtr antiqNode = GetNodeFromMapping(PATTERN_ANTIQUANT, mapping);
  ge::NodePtr maxpoolNode = GetNodeFromMapping(PATTERN_MAXPOOL, mapping);

  FUSION_PASS_CHECK(dequantNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "dequantNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(quantNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "quantNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(antiqNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "antiqNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(maxpoolNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "maxPoolNode is null, fusion failed."),
                    return PARAM_INVALID);

  if (IsMatch(dequantNode, quantNode, antiqNode, maxpoolNode) != SUCCESS) {
    return NOT_CHANGED;
  }

  dequantNode->GetOpDesc()->SetType("AscendDequant");

  FUSION_PASS_CHECK(graph.RemoveNode(antiqNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove antiquant node failed."), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(quantNode) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove quant node failed."),
                    return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(maxpoolNode) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove maxPool node failed."), return FAILED);

  fusionNodes.push_back(dequantNode);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define AscendDequantQuantAntiquantMaxpoolFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("AscendDequantQuantAntiquantMaxpoolFusionPass", BUILT_IN_GRAPH_PASS,
              AscendDequantQuantAntiquantMaxpoolFusionPass);
}  // namespace fe
