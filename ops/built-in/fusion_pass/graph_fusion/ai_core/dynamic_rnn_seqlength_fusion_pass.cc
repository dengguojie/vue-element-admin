/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "dynamic_rnn_seqlength_fusion_pass.h"
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
static const char *FUSED_NODE = "DynamicRNN";
static const std::string PATTERN_FUSEDNODE = "DynamicRNN";
static const int SEQ_LEN_INDEX = 3;

vector<FusionPattern *> DynamicRNNSeqFusionPass::DefinePatterns()
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicRNNSeqFusionPass pattern begin.");
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("DynamicRNNSeqFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "dynamicRNN seqLength pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, { FUSED_NODE }).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicRNNSeqFusionPass pattern end.");
  return patterns;
}

Status DynamicRNNSeqFusionPass::AddRNNMaskNode(ge::NodePtr fusedNode, ge::ComputeGraph &graph,
                                               vector<ge::NodePtr> &newNodes)
{
  OP_LOGD(FUSED_OP_TYPE.c_str(), "step into AddRNNMaskNode.");
  int32_t seqLenIndex = SEQ_LEN_INDEX;
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
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(existRnnNode->GetOutDataAnchor(0),
                        fusedNode->GetInDataAnchor(SEQ_LEN_INDEX)),
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"),
                        return FAILED);
      return SUCCESS;
  }
  ge::OpDescPtr rnnMaskDesc = nullptr;
  ge::GeTensorDesc inputRnnMaskDesc = fusedNode->GetOpDesc()->GetInputDesc(seqLenIndex).Clone();
  std::vector <int64_t> dimLength = inputRnnMaskDesc.GetShape().GetDims();
  FUSION_PASS_CHECK(dimLength.size() != 1,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Unexcepted seqlength input shape"), return FAILED);
  int64_t batchSize = dimLength[0];
  int64_t numStep = fusedNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDim(0);
  int64_t hiddenSize = fusedNode->GetOpDesc()->GetInputDesc(2).GetShape().GetDim(0) / 4;
  std::vector <int64_t> maskDims = {numStep, batchSize, hiddenSize};
  ge::GeShape tensorMaskShape(maskDims);
  ge::GeShape tensorMaskOriginShape(maskDims);
  ge::GeTensorDesc tensorOutputMaskDesc = ge::GeTensorDesc(tensorMaskShape, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensorOutputMaskDesc.SetOriginShape(tensorMaskOriginShape);
  tensorOutputMaskDesc.SetOriginFormat(ge::FORMAT_ND);
  FUSION_PASS_MAKE_SHARED(
      (rnnMaskDesc = std::make_shared<ge::OpDesc>(fusedNode->GetName() + "/RnnGenMask", "RnnGenMask")),
      rnnMaskDesc = nullptr; return FAILED);
  rnnMaskDesc->AddInputDesc("seq_length", inputRnnMaskDesc);
  rnnMaskDesc->AddOutputDesc("seq_mask", tensorOutputMaskDesc);
  fusedNode->GetOpDesc()->UpdateInputDesc("seq_length", tensorOutputMaskDesc);

  // Set Attr
  ge::AttrUtils::SetInt(rnnMaskDesc, "num_step", numStep);
  ge::AttrUtils::SetInt(rnnMaskDesc, "hidden_size", hiddenSize);

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
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add cast input edge failed"), return FAILED);

  //Remove Edge
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetInDataAnchor(seqLenIndex)->GetPeerOutAnchor(),
                                          fusedNode->GetInDataAnchor(seqLenIndex)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove edge between seq_length and gruv2 failed"), return FAILED);

  //Add Edge
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(maskNode->GetOutDataAnchor(0), fusedNode->GetInDataAnchor(seqLenIndex)),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add Mask output edge failed"), return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "function AddRNNMaskNode end.");
  return SUCCESS;
}

Status DynamicRNNSeqFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  // get the NodePtr of LSTM
  OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_rnn seqLength fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // get the OpDescPtr of LSTM
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  std::vector <int64_t> dimInitC = fusedNode->GetOpDesc()->GetInputDesc(4).GetShape().GetDims();
  if (dimInitC.size() == 2) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "init_c's size is 2.");
    std::vector <int64_t> initCDims = {1, dimInitC[0], dimInitC[1]};
    ge::GeShape tensorInitCOriginShape(initCDims);

    // process init_h
    ge::GeTensorDesc init_h_desc = *fusedDesc->MutableInputDesc("init_h");
    init_h_desc.SetOriginShape(tensorInitCOriginShape);
    init_h_desc.SetShape(tensorInitCOriginShape);
    fusedDesc->UpdateInputDesc("init_h", init_h_desc);

    // process init_c
    ge::GeTensorDesc init_c_desc = *fusedDesc->MutableInputDesc("init_c");
    init_c_desc.SetOriginShape(tensorInitCOriginShape);
    init_c_desc.SetShape(tensorInitCOriginShape);
    fusedDesc->UpdateInputDesc("init_c", init_c_desc);
  }

  // process seq_length
  bool hasSeqLength = fusedDesc->MutableInputDesc("seq_length") != nullptr;
  if (hasSeqLength) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "seq_length is not none.");
    int32_t seqLenIndex = SEQ_LEN_INDEX;
    ge::GeTensorDesc inputMaskDesc = fusedDesc->GetInputDesc(seqLenIndex);
    std::vector <int64_t> dimLength = inputMaskDesc.GetShape().GetDims();

    FUSION_PASS_CHECK(dimLength.size() == 3,
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "The seqlength is already in the form of seqmask."),
                      return NOT_CHANGED);
    if (dimLength.size() == 0) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "seq_length is not none, but is it a scalar.");
      std::vector <int64_t> dimSeq = fusedNode->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
      std::vector <int64_t> seqDims = {dimSeq[1]};
      ge::GeShape tensorSeqOriginShape(seqDims);
      ge::GeTensorDesc seq_length_desc = *fusedDesc->MutableInputDesc("seq_length");
      seq_length_desc.SetOriginShape(tensorSeqOriginShape);
      seq_length_desc.SetShape(tensorSeqOriginShape);
      fusedDesc->UpdateInputDesc("seq_length", seq_length_desc);
    }
    FUSION_PASS_CHECK(SUCCESS != AddRNNMaskNode(fusedNode, graph, newNodes),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "AddRNNMaskNode return failed"),
                      return FAILED);

    OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic_rnn seqLength fusion end.");
    return SUCCESS;
  } else {
    return NOT_CHANGED;
  }

}

REGISTER_PASS("DynamicRNNSeqFusionPass", BUILT_IN_GRAPH_PASS, DynamicRNNSeqFusionPass);
} // namespace fe