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

#include "dynamic_rnn_v2_seqlength_fusion_pass.h"
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
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "external/graph/operator_factory.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char *FUSED_NODE = "DynamicRNNV2";
static const std::string PATTERN_FUSEDNODE = "DynamicRNNV2";
static const int INIT_LEN = 2;
static const int W_HIDDEN_INDEX = 2;
static const int SEQ_MASK_LEN = 3;
static const int SEQ_LEN_INDEX = 4;
static const int INIT_H_LEN_INDEX = 5;
static const int INIT_C_LEN_INDEX = 6;

vector<FusionPattern *> DynamicRNNV2SeqFusionPass::DefinePatterns()
{
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define DynamicRNNV2SeqFusionPass pattern begin.");
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("DynamicRNNV2SeqFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "dynamicRNNV2 seqLength pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define DynamicRNNV2SeqFusionPass pattern end.");
  return patterns;
}

Status DynamicRNNV2SeqFusionPass::AddRNNMaskNode(ge::NodePtr fusedNode, ge::ComputeGraph &graph,
                                                 vector<ge::NodePtr> &newNodes)
{
  OP_LOGD(FUSED_OP_TYPE.c_str(), "step into AddRNNMaskNode.");
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
    ge::GeTensorDesc tensorOutDesc = existRnnNode->GetOpDesc()->GetOutputDesc(0).Clone();
    fusedNode->GetOpDesc()->UpdateInputDesc("seq_length", tensorOutDesc);
    // Add Edge
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(existRnnNode->GetOutDataAnchor(0),
                                         fusedNode->GetInDataAnchor(SEQ_LEN_INDEX)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);
    return SUCCESS;
  }
  ge::OpDescPtr rnnMaskDesc = nullptr;
  ge::GeTensorDesc inputRnnMaskDesc = fusedNode->GetOpDesc()->GetInputDesc(seqLenIndex).Clone();
  std::vector <int64_t> dimLength = inputRnnMaskDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dimLength.size() != 1,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Unexcepted seqlength input shape"), return FAILED);
  int64_t batchSize = dimLength[0];
  int64_t numStep = fusedNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(0);
  int64_t hiddenSize = fusedNode->GetOpDesc()->GetInputDesc(W_HIDDEN_INDEX).GetShape().GetDim(0);
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

  // Creat Mask
  ge::NodePtr maskNode = graph.AddNode(rnnMaskDesc);
  FUSION_PASS_CHECK(maskNode == nullptr,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Create Mask node:%s failed", rnnMaskDesc->GetName().c_str()),
    return FAILED);
  newNodes.push_back(maskNode);
  // Add Edge
  FUSION_PASS_CHECK(
    SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor(),
                                       maskNode->GetInDataAnchor(0)),
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add cast input edge failed"), return FAILED);

  // Remove Edge
  FUSION_PASS_CHECK(
    SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor(),
                                          fusedNode->GetInDataAnchor(seqLenIndex)),
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge between seq_length and rnnv2 failed"), return FAILED);

  // Add Edge
  FUSION_PASS_CHECK(
    SUCCESS != ge::GraphUtils::AddEdge(maskNode->GetOutDataAnchor(0), fusedNode->GetInDataAnchor(seqLenIndex)),
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);

  if (PatternFusionUtil::IsUnknownShape(numStep) || PatternFusionUtil::IsUnknownShape(m_size)) {
    // Add Edge
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(xIndex)->GetPeerOutAnchor(),
                                         maskNode->GetInDataAnchor(1)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add x input edge failed"), return FAILED);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "function AddRNNMaskNode end.");
  return SUCCESS;
}

Status DynamicRNNV2SeqFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  // get the NodePtr of LSTM
  OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_rnn_v2 seqLength fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // get the OpDescPtr of LSTM
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  bool is_misplaced = false;
  ge::AttrUtils::GetBool(fusedDesc, "is_misplaced", is_misplaced);

  if (is_misplaced) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_rnn_v2 init_h and init_c is misplaced to seq and init_h.");

    // Get input desc. Input init_h and init_c is misplaced. The real init_h is seq_len, init_c is init_h.
    ge::GeTensorDesc init_h_desc = fusedDesc->GetInputDesc(SEQ_LEN_INDEX).Clone();
    ge::GeTensorDesc init_c_desc = fusedDesc->GetInputDesc(INIT_H_LEN_INDEX).Clone();

    // Clear seq_length and init_h desc, set real desc to init_h and init_c.
    FUSION_PASS_CHECK(
      !ge::OpDescUtils::ClearInputDesc(fusedDesc, INIT_H_LEN_INDEX),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's clear init_h input failed.",
              fusedDesc->GetName().c_str()),
      return PARAM_INVALID);
    FUSION_PASS_CHECK(
      !ge::OpDescUtils::ClearInputDesc(fusedDesc, SEQ_LEN_INDEX),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's clear seq_length input failed.",
              fusedDesc->GetName().c_str()),
      return PARAM_INVALID);
    fusedDesc->AddInputDesc("init_h", init_h_desc);
    fusedDesc->AddInputDesc("init_c", init_c_desc);

    // Delete init_h edge and add edge to init_c
    auto InithOutAnchor = fusedNode->GetInDataAnchor(INIT_H_LEN_INDEX)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(InithOutAnchor,
                                            fusedNode->GetInDataAnchor(INIT_H_LEN_INDEX)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge between seq_length and data failed"), return FAILED);
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(InithOutAnchor,
                                         fusedNode->GetInDataAnchor(INIT_C_LEN_INDEX)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);

    // Delete seq_length edge and add edge to init_h
    auto SeqOutAnchor = fusedNode->GetInDataAnchor(SEQ_LEN_INDEX)->GetPeerOutAnchor();
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(SeqOutAnchor,
                                            fusedNode->GetInDataAnchor(SEQ_LEN_INDEX)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge between seq_length and data failed"), return FAILED);
    FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(SeqOutAnchor,
                                         fusedNode->GetInDataAnchor(INIT_H_LEN_INDEX)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);
  }

  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  bool hasInitH = fusedDesc->MutableInputDesc("init_h") != nullptr;
  bool hasInitC = fusedDesc->MutableInputDesc("init_c") != nullptr;
  if (hasInitH) {
    std::vector <int64_t> dimInitH = fusedNode->GetOpDesc()->GetInputDesc(INIT_H_LEN_INDEX).GetShape().GetDims();
    if (dimInitH.size() == INIT_LEN) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "init_h's size is 2.");
      std::vector <int64_t> initHDims = {1, dimInitH[0], dimInitH[1]};
      ge::GeShape tensorInitHOriginShape(initHDims);

      // process init_h
      auto init_h_desc = fusedDesc->MutableInputDesc("init_h");
      init_h_desc->SetOriginShape(tensorInitHOriginShape);
      init_h_desc->SetShape(tensorInitHOriginShape);
    }
  }
  if (hasInitC) {
    std::vector <int64_t> dimInitC = fusedNode->GetOpDesc()->GetInputDesc(INIT_C_LEN_INDEX).GetShape().GetDims();
    if (dimInitC.size() == INIT_LEN) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "init_c's size is 2.");
      std::vector <int64_t> initCDims = {1, dimInitC[0], dimInitC[1]};
      ge::GeShape tensorInitCOriginShape(initCDims);

      // process init_c
      auto init_c_desc = fusedDesc->MutableInputDesc("init_c");
      init_c_desc->SetOriginShape(tensorInitCOriginShape);
      init_c_desc->SetShape(tensorInitCOriginShape);
    }
  }

  // process seq_length
  bool hasSeqLength = fusedDesc->MutableInputDesc("seq_length") != nullptr;
  if (hasSeqLength) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "seq_length is not none.");
    int32_t seqLenIndex = SEQ_LEN_INDEX;
    ge::GeTensorDesc inputMaskDesc = fusedDesc->GetInputDesc(seqLenIndex);
    std::vector <int64_t> dimLength = inputMaskDesc.GetShape().GetDims();

    FUSION_PASS_CHECK(dimLength.size() == SEQ_MASK_LEN,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "The seqlength is already in the form of seqmask."),
                      return NOT_CHANGED);
    if (dimLength.size() == 0) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "seq_length is not none, but it is a scalar.");
      std::vector <int64_t> dimSeq = fusedNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
      std::vector <int64_t> seqDims = {dimSeq[1]};
      ge::GeShape tensorSeqOriginShape(seqDims);
      auto seq_length_desc = fusedDesc->MutableInputDesc("seq_length");
      seq_length_desc->SetOriginShape(tensorSeqOriginShape);
      seq_length_desc->SetShape(tensorSeqOriginShape);
    }
    FUSION_PASS_CHECK(SUCCESS != AddRNNMaskNode(fusedNode, graph, newNodes),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "AddRNNMaskNode return failed"),
                      return FAILED);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_rnn_v2 seqLength fusion end.");
    return SUCCESS;
  } else {
    return NOT_CHANGED;
  }
}

REGISTER_PASS("DynamicRNNV2SeqFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNV2SeqFusionPass);
} // namespace fe
