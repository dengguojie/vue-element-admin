/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "dynamic_gru_v2_seqlength_fusion_pass.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "DynamicGRUV2";
static const std::string PATTERN_FUSEDNODE = "DynamicGRUV2";
static const int SEQ_LEN_INDEX = 5;

vector<FusionPattern *> DynamicGRUV2SeqFusionPass::DefinePatterns()
{
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("DynamicGRUV2SeqFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                                                "dynamicGRUV2 seqLength pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  return patterns;
}

Status DynamicGRUV2SeqFusionPass::AddRNNMaskNode(ge::NodePtr fusedNode, ge::ComputeGraph &graph,
                                                 vector<ge::NodePtr> &newNodes)
{
  int32_t seqLenIndex = SEQ_LEN_INDEX;
  int32_t xIndex = 0;
  bool rnnGenMaskExist = false;
  ge::NodePtr existRnnNode = nullptr;
  auto outDataAnchor = fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor();
  for (auto nextInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
    ge::NodePtr outputNode = nextInDataAnchor->GetOwnerNode();
    if (outputNode->GetType() == "RnnGenMask" || outputNode->GetType() == "RnnGenMaskV2") {
      rnnGenMaskExist = true;
      existRnnNode = outputNode;
      break;
    }
  }
  if (rnnGenMaskExist) {
    ge::GeTensorDesc tensorOutDesc = existRnnNode->GetOpDesc()->GetOutputDesc(0);
    fusedNode->GetOpDesc()->UpdateInputDesc("seq_length", tensorOutDesc);
    // Add Edge
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(existRnnNode->GetOutDataAnchor(0),
        fusedNode->GetInDataAnchor(seqLenIndex)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);
    return SUCCESS;
  }
  ge::OpDescPtr rnnMaskDesc = nullptr;
  ge::GeTensorDesc inputRnnMaskDesc = fusedNode->GetOpDesc()->GetInputDesc(seqLenIndex);
  std::vector <int64_t> dimLength = inputRnnMaskDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dimLength.size() != 1,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Unexcepted seqlength input shape"), return FAILED);
  int64_t batchSize = dimLength[0];
  int64_t numStep = fusedNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(0);
  int64_t hiddenSize = fusedNode->GetOpDesc()->GetInputDesc(2).GetShape().GetDim(0);
  int64_t m_size = fusedNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(1);

  std::vector <int64_t> maskDims = {numStep, batchSize, hiddenSize};
  ge::GeShape tensorMaskShape(maskDims);
  ge::GeShape tensorMaskOriginShape(maskDims);
  ge::GeTensorDesc tensorOutputMaskDesc = ge::GeTensorDesc(tensorMaskShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensorOutputMaskDesc.SetOriginShape(tensorMaskOriginShape);
  tensorOutputMaskDesc.SetOriginFormat(ge::FORMAT_ND);

  if (PatternFusionUtil::IsUnknownShape(numStep) || PatternFusionUtil::IsUnknownShape(m_size)) {
    FUSION_PASS_MAKE_SHARED(
      (rnnMaskDesc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "/RnnGenMaskV2", "RnnGenMaskV2")),
      rnnMaskDesc = nullptr; return FAILED);
    rnnMaskDesc->AddInputDesc("seq_length", inputRnnMaskDesc);
    ge::GeTensorDesc inputDesc = fusedNode->GetOpDesc()->GetInputDesc(xIndex);
    rnnMaskDesc->AddInputDesc("x", inputDesc);
    rnnMaskDesc->AddOutputDesc("seq_mask", tensorOutputMaskDesc);
    fusedNode->GetOpDesc()->UpdateInputDesc("seq_length", tensorOutputMaskDesc);
    // Set Attr
    ge::AttrUtils::SetInt(rnnMaskDesc, "hidden_size", hiddenSize);
  } else {
    FUSION_PASS_MAKE_SHARED(
      (rnnMaskDesc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "/RnnGenMask", "RnnGenMask")),
      rnnMaskDesc = nullptr; return FAILED);
    rnnMaskDesc->AddInputDesc("seq_length", inputRnnMaskDesc);
    rnnMaskDesc->AddOutputDesc("seq_mask", tensorOutputMaskDesc);
    fusedNode->GetOpDesc()->UpdateInputDesc("seq_length", tensorOutputMaskDesc);
    // Set Attr
    ge::AttrUtils::SetInt(rnnMaskDesc, "num_step", numStep);
    ge::AttrUtils::SetInt(rnnMaskDesc, "hidden_size", hiddenSize);
  }

  //Creat Mask
  ge::NodePtr maskNode = graph.AddNode(rnnMaskDesc);
  FUSION_PASS_CHECK(maskNode == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Create Mask node:%s failed", rnnMaskDesc->GetName().c_str()),
      return FAILED);
  newNodes.push_back(maskNode);

  //Add Edge
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor(),
                                          maskNode->GetInDataAnchor(0)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask input edge failed"), return FAILED);

  //Remove Edge
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor(),
                                          fusedNode->GetInDataAnchor(seqLenIndex)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge between seq_length and gruv2 failed"), return FAILED);

  //Add Edge
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(maskNode->GetOutDataAnchor(0), fusedNode->GetInDataAnchor(seqLenIndex)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);

  if (PatternFusionUtil::IsUnknownShape(numStep) || PatternFusionUtil::IsUnknownShape(m_size)) {
    //Add Edge
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(xIndex)->GetPeerOutAnchor(),
                                            maskNode->GetInDataAnchor(1)),
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add x input edge failed"), return FAILED);
  }

  return SUCCESS;
}

Status DynamicGRUV2SeqFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  // get the NodePtr of dynamic_gru_v2
  OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_gru_v2 seqLength start fusion");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "fusedNode is null, fusion failed."),
  return PARAM_INVALID);

  // get the OpDescPtr of dynamic_gru_v2
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "fusedNode OpDesc is null, fusion failed."),
  return PARAM_INVALID);

  // process seq_length
  bool hasSeqLength = fusedDesc->MutableInputDesc("seq_length") != nullptr;
  if (hasSeqLength) {
    int32_t seqLenIndex = SEQ_LEN_INDEX;
    ge::GeTensorDesc inputMaskDesc = fusedDesc->GetInputDesc(seqLenIndex);
    std::vector <int64_t> dimLength = inputMaskDesc.GetShape().GetDims();

    FUSION_PASS_CHECK(dimLength.size() == 3,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "The seqlength is already in the form of seqmask."),
                      return NOT_CHANGED);

    FUSION_PASS_CHECK(SUCCESS != AddRNNMaskNode(fusedNode, graph, newNodes),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "AddRNNMaskNode return failed"),
                      return FAILED);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_gru_v2 seqLength end fusion");
    return SUCCESS;
  } else {
    return NOT_CHANGED;
  }

}

REGISTER_PASS("DynamicGRUV2AddSeqPass", BUILT_IN_GRAPH_PASS, DynamicGRUV2SeqFusionPass);
} // namespace fe
