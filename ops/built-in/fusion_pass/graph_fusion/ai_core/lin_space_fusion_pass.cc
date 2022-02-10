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
 * \file lin_space_fusion_pass.cpp
 * \brief lin_space fusion pass
 */
#include "lin_space_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
const int32_t LIN_SPACE_NODE_INPUT_NUM_IDX = 3;
const int32_t LIN_SPACE_NODE_INPUT_STOP_IDX = 2;
const int32_t LIN_SPACE_NODE_INPUT_START_IDX = 1;
static const float FLOAT_NUM_ZERO = 0;
static const string PATTERN_LINSPACE = "LinSpace";
const char* LINSPACE = "LinSpace";

Status AssitHelpFloat(const int32_t n, float* output) {
  for (int32_t i = 0; i < n; i++) {
    output[i] = float(i);
  }
  return SUCCESS;
}

vector<FusionPattern*> LinSpaceFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter LinSpace fusion pass");
  vector<FusionPattern*> patterns;
  // lin_space fused to lin_space_d
  FusionPattern* pattern = new (std::nothrow) FusionPattern("LinSpaceFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_LINSPACE, {LINSPACE}).SetOutput(PATTERN_LINSPACE);
  patterns.push_back(pattern);

  return patterns;
}

Status LinSpaceFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get the lin_space node
  ge::NodePtr linspaceVNode = GetNodeFromMapping(PATTERN_LINSPACE, mapping);
  FUSION_PASS_CHECK(linspaceVNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "linspaceVNode is null, fusion failed."), return PARAM_INVALID);

  // get the desc of lin_space node
  ge::OpDescPtr linspaceDesc = linspaceVNode->GetOpDesc();
  FUSION_PASS_CHECK(linspaceDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "linspaceVNode's OpDesc is null, fusion failed."), return PARAM_INVALID);
  vector<int64_t> dims = linspaceDesc->GetOutputDesc("output").GetShape().GetDims();
  for (int64_t ele : dims) {
    if (ele == UNKNOWN_DIM) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "It is unknown shape, not changed");
      return NOT_CHANGED;
    }
  }

  // desc copy
  ge::OpDescPtr linSpaceDDesc = AttrUtils::CopyOpDesc(linspaceDesc);
  FUSION_PASS_CHECK(linSpaceDDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "linSpaceDDesc's OpDesc is null, fusion failed."), return PARAM_INVALID);

  ge::GeTensorDesc tensorDesc0 = linspaceVNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc tensorDesc1 = linspaceVNode->GetOpDesc()->GetInputDesc(LIN_SPACE_NODE_INPUT_START_IDX);
  ge::GeTensorDesc tensorDesc2 = linspaceVNode->GetOpDesc()->GetInputDesc(LIN_SPACE_NODE_INPUT_STOP_IDX);

  // find the parent node of lin_space
  ge::InDataAnchorPtr linspaceAnchorPtr2 = linspaceVNode->GetInDataAnchor(LIN_SPACE_NODE_INPUT_STOP_IDX);
  ge::OutDataAnchorPtr constAnchorPtr2 = linspaceAnchorPtr2->GetPeerOutAnchor();
  ge::NodePtr constNode2 = constAnchorPtr2->GetOwnerNode();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Success to get the father node\n");
  // get the output desc of parent node of lin_space
  ge::GeTensorDesc linspaceInputTensor = constNode2->GetOpDesc()->GetOutputDesc(0);

  ge::ConstGeTensorPtr constTensor2 = nullptr;
  ge::AttrUtils::GetTensor(constNode2->GetOpDesc(), "value", constTensor2);
  FUSION_PASS_CHECK(constTensor2 == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                    "constTensor is nullptr, does not need fusion"),
                    return NOT_CHANGED);
  const uint8_t* constData = constTensor2->GetData().GetData();
  FUSION_PASS_CHECK(constData == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constData is NULL"),
                    return PARAM_INVALID);
  int32_t num = *(reinterpret_cast<const int32_t*>(constData));
  OP_LOGD(FUSED_OP_TYPE.c_str(), "The num is %d\n", num);

  Format assistFormat = linspaceInputTensor.GetFormat();

  vector<int64_t> assistShapeVec;
  assistShapeVec.push_back(num);
  ge::GeShape assistShape(assistShapeVec);

  ge::GeTensorPtr assitPtr = nullptr;
  unique_ptr<float[]> inputAssit(new (std::nothrow) float[num]());
  FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "inputAssit is NULL"), return PARAM_INVALID);
  Status ret = NnSet(num, FLOAT_NUM_ZERO, *reinterpret_cast<float*>(inputAssit.get()));
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

  ret = AssitHelpFloat(num, inputAssit.get());
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  tensorDesc.SetShape(assistShape);
  tensorDesc.SetFormat(assistFormat);

  FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                               tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), num * sizeof(float))),
                          assitPtr = nullptr;
                          return PARAM_INVALID);

  OpDescUtils::ClearInputDesc(linSpaceDDesc, LIN_SPACE_NODE_INPUT_STOP_IDX);
  OpDescUtils::ClearInputDesc(linSpaceDDesc, LIN_SPACE_NODE_INPUT_START_IDX);
  OpDescUtils::ClearInputDesc(linSpaceDDesc, 0);

  // new the assist node
  std::shared_ptr<ge::OpDesc> newConstantOp = std::make_shared<ge::OpDesc>(linSpaceDDesc->GetName() +
                                                                           "_assist", "Constant");
  FUSION_PASS_MAKE_SHARED(newConstantOp, return PARAM_INVALID);
  ge::AttrUtils::SetTensor(newConstantOp, "value", assitPtr);
  (void)newConstantOp->AddOutputDesc(assitPtr->GetTensorDesc());
  ge::NodePtr assistNode = graph.AddNode(newConstantOp);

  linSpaceDDesc->AddInputDesc(0, newConstantOp->GetOutputDesc(0));
  linSpaceDDesc->AddInputDesc(LIN_SPACE_NODE_INPUT_START_IDX, linspaceDesc->GetInputDesc(0));
  linSpaceDDesc->AddInputDesc(LIN_SPACE_NODE_INPUT_STOP_IDX,
                              linspaceDesc->GetInputDesc(LIN_SPACE_NODE_INPUT_START_IDX));
  linSpaceDDesc->AddInputDesc(LIN_SPACE_NODE_INPUT_NUM_IDX, linspaceDesc->GetInputDesc(LIN_SPACE_NODE_INPUT_STOP_IDX));

  ge::NodePtr linSpaceDNode = graph.AddNode(linSpaceDDesc);
  FUSION_PASS_CHECK(
      linSpaceDNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNodeis null, fusion failed."),
      return PARAM_INVALID);
  fusionNodes.push_back(linSpaceDNode);

  linSpaceDDesc->SetType("LinSpaceD");
  OP_LOGD(FUSED_OP_TYPE.c_str(), "The size of linspaced's indataanchor is %d\n",
          linSpaceDNode->GetAllInDataAnchors().size());

  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(assistNode->GetOutDataAnchor(0), linSpaceDNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add edge from %s's index[%d] to %s's index[%d] failed.",
              assistNode->GetName().c_str(), 0, linSpaceDNode->GetName().c_str(), 0),
      return FAILED);

  for (unsigned int i = 0; i < linspaceVNode->GetAllInDataAnchors().size(); i++) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(linspaceVNode->GetInDataAnchor(i)->GetPeerOutAnchor(),
                                                         linSpaceDNode->GetInDataAnchor(i + 1)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                              "Add edge from %s's index[%d] to %s's index[%d] failed.",
                              linspaceVNode->GetName().c_str(), i, linSpaceDNode->GetName().c_str(), i + 1),
                      return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Success to add edge from %s's index[%d] to %s's index[%d].",
            linspaceVNode->GetName().c_str(), i, linSpaceDNode->GetName().c_str(), i + 1);
  }

  if (linspaceVNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "The size of linspaceVNode is [%d].",
            linspaceVNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : linspaceVNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(linSpaceDNode->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from %s's index to %s's 1st index failed.",
                                linSpaceDNode->GetName().c_str(), inAnchorPtr->GetOwnerNode()->GetName().c_str()),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from %s's 1st index to %s's 1st index.",
              linspaceVNode->GetName().c_str(), linSpaceDNode->GetName().c_str());
    }
  }

  for (auto inAnchor : linspaceVNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(linspaceVNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
      linspaceVNode->GetName().c_str()), return FAILED);

  return SUCCESS;
}

REGISTER_PASS("LinSpaceFusionPass", BUILT_IN_GRAPH_PASS, LinSpaceFusionPass);
}  // namespace fe
