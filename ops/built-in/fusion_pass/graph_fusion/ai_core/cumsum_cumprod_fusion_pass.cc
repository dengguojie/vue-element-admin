/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file cumsum_cumprod_fusion_pass.cpp
 * \brief
 */
#include "cumsum_cumprod_fusion_pass.h"
#include <vector>
#include <string>

#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE_CUMPROD = "CumprodD";
static const std::string PATTERN_CUMPROD = "Cumprod";
static const char* FUSED_NODE_CUMSUM = "CumsumD";
static const std::string PATTERN_CUMSUM = "Cumsum";

vector<FusionPattern*> CumFusionPass::DefinePatterns()
{
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define CumFusionPass pattern begin");
  vector<FusionPattern*> patterns;

  FusionPattern* pattern0 = new (std::nothrow) FusionPattern("CumFusionPass");
  FUSION_PASS_CHECK(pattern0 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new pattern0 object failed."),
                    return patterns);

  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("CumFusionPass");
  FUSION_PASS_CHECK(pattern1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new pattern1 object failed."),
                    return patterns);

  pattern0->AddOpDesc(PATTERN_CUMSUM, {FUSED_NODE_CUMSUM}).SetOutput(PATTERN_CUMSUM);
  patterns.push_back(pattern0);

  pattern1->AddOpDesc(PATTERN_CUMPROD, {FUSED_NODE_CUMPROD}).SetOutput(PATTERN_CUMPROD);
  patterns.push_back(pattern1);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define CumFusionPass pattern end");
  return patterns;
}

Status CumFusionPass::AddTransposeNode(ge::NodePtr cumNode, ge::ComputeGraph &graph, vector<ge::NodePtr> &newNodes,
                                       vector<int64_t> &dimInfo, vector<int64_t> &permListIn,
                                       vector<int64_t> &permListOut)
{
  ge::InDataAnchorPtr cumAnchorPtr0 = cumNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr preAnchorPtr0 = cumAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr preNode = preAnchorPtr0->GetOwnerNode();
  // creat a transposeD node
  std::shared_ptr<ge::OpDesc> transposeDDesc = nullptr;
  transposeDDesc = std::make_shared<ge::OpDesc>(cumNode->GetName() + "_transposeD_layer", "TransposeD");
  FUSION_PASS_CHECK(transposeDDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransposeD failed."),
                    return PARAM_INVALID);
  // add input
  ge::GeTensorDesc input_desc = preNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(transposeDDesc->AddInputDesc(input_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc input failed."),
                    return FAILED);
  // add output
  ge::GeTensorDesc output_desc = cumNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(transposeDDesc->AddOutputDesc(output_desc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc output failed."),
                    return FAILED);
  // add node
  ge::NodePtr transposeDNode = graph.AddNode(transposeDDesc);
  FUSION_PASS_CHECK(transposeDNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Create Mask node:%s failed", transposeDDesc->GetName().c_str()),
                    return FAILED);
  newNodes.push_back(transposeDNode);

  ge::AttrUtils::SetListInt(transposeDDesc, "perm", permListIn);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preAnchorPtr0, cumAnchorPtr0) != SUCCESS,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between quant and pooling failed!"), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preAnchorPtr0, transposeDNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                    "Add edge between node %s and node %s failed.",
                                                    preNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode->GetOutDataAnchor(0), cumAnchorPtr0) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                    "Add edge between node %s and node %s failed.",
                                                    cumNode->GetName().c_str(), transposeDNode->GetName().c_str()),
                    return FAILED);
  // creat a transposeD node
  ge::OutDataAnchorPtr cumAnchorPtr1 = cumNode->GetOutDataAnchor(0);
  auto transposeDpPtr1 = cumAnchorPtr1->GetPeerInDataAnchors().at(0);

  std::shared_ptr<ge::OpDesc> transposeDDesc1 = nullptr;
  transposeDDesc1 = std::make_shared<ge::OpDesc>(cumNode->GetName() + "_transposeD1_layer", "TransposeD");
  FUSION_PASS_CHECK(transposeDDesc1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transposeD is null, TransposeD failed."),
                    return PARAM_INVALID);
  // add input
  ge::GeTensorDesc input_desc1 = cumNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(transposeDDesc1->AddInputDesc(input_desc1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 input failed."),
                    return FAILED);
  // add output
  ge::GeTensorDesc output_desc1 = transposeDpPtr1->GetOwnerNode()->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(transposeDDesc1->AddOutputDesc(output_desc1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transposeDDesc1 output failed."),
                    return FAILED);
  ge::GeTensorDesc transposeD1OutputOpDesc = transposeDDesc1->GetOutputDesc(0);
  ge::GeShape assitShape1(dimInfo);
  ge::GeShape assitShapeOrigin1(dimInfo);
  transposeD1OutputOpDesc.SetShape(assitShape1);
  transposeD1OutputOpDesc.SetOriginShape(assitShapeOrigin1);
  Status ret = transposeDDesc1->UpdateOutputDesc(0, transposeD1OutputOpDesc);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc failed."),
                    return FAILED);
  // add node
  ge::NodePtr transposeDNode1 = graph.AddNode(transposeDDesc1);
  FUSION_PASS_CHECK(transposeDNode1 == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Create Mask node:%s failed", transposeDDesc1->GetName().c_str()),
                    return FAILED);
  newNodes.push_back(transposeDNode1);
  ge::AttrUtils::SetListInt(transposeDDesc1, "perm", permListOut);
  // add edge between cumsum and transdateD
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(cumAnchorPtr1, transposeDNode1->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Add edge between node %s and node %s failed.",
                                                   cumNode->GetName().c_str(),
                                                   transposeDNode1->GetName().c_str()),
                    return FAILED);
  for (auto postAnchorPtr0 : cumAnchorPtr1->GetPeerInDataAnchors()) {
    if (postAnchorPtr0->GetOwnerNode()->GetName() != cumNode->GetName() + "_transposeD1_layer") {
      // remove edge between cumop and next node
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(postAnchorPtr0, cumAnchorPtr1) != SUCCESS,
                        OP_LOGI(FUSED_OP_TYPE.c_str(), "remove edge between cumsum and next node failed!"),
                        return FAILED);
      // add edge between transdateD and post
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeDNode1->GetOutDataAnchor(0), postAnchorPtr0) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                       "Add edge between node %s and node %s failed.",
                                                       transposeDDesc1->GetName().c_str(),
                                                       cumNode->GetName().c_str()),
                        return FAILED);
    }
  }
  return SUCCESS;
}

Status CumFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes)
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter CumFusionPass!");
  ge::NodePtr cumprodNode = GetNodeFromMapping(PATTERN_CUMPROD, mapping);
  ge::NodePtr cumsumNode = GetNodeFromMapping(PATTERN_CUMSUM, mapping);
  ge::NodePtr cumNode;
  if (cumprodNode == nullptr && cumsumNode == nullptr) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "cumsum and cumprod node both are null.");
    return PARAM_INVALID;
  } else if (cumprodNode == nullptr) {
    cumNode = cumsumNode;
  } else {
    cumNode = cumprodNode;
  }
  ge::OpDescPtr cumOpDesc = cumNode->GetOpDesc();
  FUSION_PASS_CHECK(cumOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "cum opdesc is null."),
                    return PARAM_INVALID);
  ge::GeTensorDesc cumInputOpDesc = cumOpDesc->GetInputDesc(0);
  ge::GeTensorDesc cumOutputOpDesc = cumOpDesc->GetOutputDesc(0);
  ge::GeShape cumInputShape = cumInputOpDesc.GetShape();
  vector<int64_t> dimInfo = cumInputShape.GetDims();
  int64_t input_dim = dimInfo.size();
  OP_LOGD(FUSED_OP_TYPE.c_str(), "cumop input size is %d.", input_dim);
  int64_t axis = 0;
  if (!ge::AttrUtils::GetInt(cumOpDesc, "axis", axis)) {
    ge::AttrUtils::SetInt(cumOpDesc, "axis", axis);
  }
  if (axis < 0) {
    axis = axis + input_dim;
  }
  if (axis == input_dim - 1 && input_dim > 1) {
    int64_t inputLast = dimInfo[axis];
    vector<int64_t> inputDimInfo = {inputLast};
    vector<int64_t> permListIn = {axis};
    vector<int64_t> permListOut;
    for (int64_t i = 0; i < input_dim - 1; i++) {
      inputDimInfo.push_back(dimInfo[i]);
      permListIn.push_back(i);
      permListOut.push_back(i + 1);
    }
    permListOut.push_back(0);
    ge::GeShape assitShape(inputDimInfo);
    ge::GeShape assitShapeOrigin(inputDimInfo);
    cumInputOpDesc.SetShape(assitShape);
    cumInputOpDesc.SetOriginShape(assitShapeOrigin);
    cumOutputOpDesc.SetShape(assitShape);
    cumOutputOpDesc.SetOriginShape(assitShapeOrigin);
    Status ret = cumOpDesc->UpdateInputDesc(0, cumInputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateInputDesc failed."),
                      return FAILED);
    ret = cumOpDesc->UpdateOutputDesc(0, cumOutputOpDesc);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "UpdateOutputDesc failed."),
                      return FAILED);
    int64_t axis_new = 0;
    FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(cumNode->GetOpDesc(), "axis", axis_new),
                      OP_LOGI(FUSED_OP_TYPE.c_str(), "Set axis attr failed."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != AddTransposeNode(cumNode, graph, newNodes, dimInfo, permListIn, permListOut),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "AddTransposeNode return failed"),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "CumFusionPass success end!");
    return SUCCESS;
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "CumFusionPass is not match.");
    return NOT_CHANGED;
  }
}
REGISTER_PASS("CumFusionPass", BUILT_IN_GRAPH_PASS, CumFusionPass);
}  // namespace fe
