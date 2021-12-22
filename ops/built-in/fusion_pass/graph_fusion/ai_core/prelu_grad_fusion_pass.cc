/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief PReluGrad fusion pass(PReluGrad --> PReluGrad & reducesum)
 *
 */

#include "prelu_grad_fusion_pass.h"

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
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "PReluGrad";
static const std::string PATTERN_FUSEDNODE = "PReluGrad";

vector<FusionPattern*> PReluGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("PReluGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr PReluGradFusionPass::AddPReluGradNoneNode(ge::NodePtr prelugradNode, ge::ComputeGraph& graph,
                                                      vector<ge::NodePtr>& newNodes, bool& failStatus) {
  ge::OpDescPtr prelugradDesc = prelugradNode->GetOpDesc();

  // create prelugrad_none desc
  ge::OpDescPtr prelugradNoneDesc = AttrUtils::CloneOpDesc(prelugradDesc);
  FUSION_PASS_CHECK(prelugradNoneDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "prelugradNone's OpDesc is null, fusion failed."),
                    failStatus = true);

  // input
  ge::GeTensorDesc inputTensorDesc = prelugradNoneDesc->GetInputDesc(0);

  // update output shape
  ge::GeTensorDesc outputTensorDesc = prelugradNoneDesc->GetOutputDesc(1);
  outputTensorDesc.SetOriginShape(inputTensorDesc.GetShape());
  outputTensorDesc.SetShape(inputTensorDesc.GetShape());
  prelugradNoneDesc->UpdateOutputDesc(1, outputTensorDesc);

  // create prelugrad_none node
  ge::NodePtr prelugradNoneNode = graph.AddNode(prelugradNoneDesc);
  FUSION_PASS_CHECK(
      prelugradNoneNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                     prelugradNoneNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(prelugradNoneNode);

  // Edge
  for (unsigned int i = 0; i < prelugradNode->GetAllInDataAnchors().size(); i++) {
    ge::GraphUtils::AddEdge(prelugradNode->GetInDataAnchor(i)->GetPeerOutAnchor(),
                            prelugradNoneNode->GetInDataAnchor(i));
  }

  FUSION_PASS_CHECK(prelugradNode->GetOutDataAnchor(0) == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "prelugrad node output anchor is null"),
                                                   failStatus = true);
  if (prelugradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : prelugradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(prelugradNoneNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  return prelugradNoneNode;
}

ge::NodePtr PReluGradFusionPass::AddReduceNode(ge::NodePtr prelugradNode, ge::NodePtr prelugradNoneNode,
                                               ge::ComputeGraph& graph, vector<ge::NodePtr>& newNodes,
                                               bool& failStatus) {
  // create reduce desc
  ge::OpDescPtr reduceDesc;

  FUSION_PASS_MAKE_SHARED(
      (reduceDesc = std::make_shared<ge::OpDesc>(prelugradNode->GetName() + "ReduceSumD", "ReduceSumD")),
      failStatus = true; return nullptr);

  // input
  ge::GeTensorDesc inputTensorDesc = prelugradNoneNode->GetOpDesc()->GetOutputDesc(1).Clone();
  if (inputTensorDesc.GetDataType() == ge::DT_FLOAT16) {
    inputTensorDesc.SetDataType(ge::DT_FLOAT);
  }
  reduceDesc->AddInputDesc("input_reduce", inputTensorDesc);

  // output
  ge::GeTensorDesc outputTensorDesc = prelugradNode->GetOpDesc()->GetOutputDesc(1).Clone();
  reduceDesc->AddOutputDesc("y", outputTensorDesc);

  // compute axes and set attr
  ge::GeTensorDesc tensor_input = prelugradNoneNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc weight_input = prelugradNoneNode->GetOpDesc()->GetInputDesc(2);
  vector<int64_t> tensor_info = tensor_input.GetShape().GetDims();
  vector<int64_t> weight_info = weight_input.GetShape().GetDims();
  size_t tensor_size = tensor_input.GetShape().GetDimNum();

  std::vector<int64_t> axes;
  axes.push_back(0);
  for (size_t i = 1; i < tensor_size; ++i) {
    if (tensor_info[i] != weight_info[i - 1]) {
      axes.push_back(i);
    }
  }
  ge::AttrUtils::SetListInt(reduceDesc, "axes", axes);
  ge::AttrUtils::SetBool(reduceDesc, "keep_dims", true);

  // create reduce node
  ge::NodePtr reduceNode = graph.AddNode(reduceDesc);
  FUSION_PASS_CHECK(
      reduceNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                                     reduceNode->GetName().c_str()),
      failStatus = true);
  newNodes.push_back(reduceNode);

  // Edge
  ge::GraphUtils::AddEdge(prelugradNoneNode->GetOutDataAnchor(1), reduceNode->GetInDataAnchor(0));
  FUSION_PASS_CHECK(prelugradNode->GetOutDataAnchor(1) == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "prelugrad node output anchor is null"),
                                                   failStatus = true);
  if (prelugradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : prelugradNode->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(0), inAnchorPtr);
    }
  }

  return reduceNode;
}

Status PReluGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  bool failStatus = false;

  // get prelugradNode
  ge::NodePtr prelugradNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(prelugradNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "prelugradNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr prelugradDesc = prelugradNode->GetOpDesc();
  FUSION_PASS_CHECK(prelugradDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "prelugrad's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc tensor_input = prelugradNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc weight_input = prelugradNode->GetOpDesc()->GetInputDesc(2);
  size_t tensor_size = tensor_input.GetShape().GetDimNum();
  size_t axis_size = weight_input.GetShape().GetDimNum();
  if (axis_size == 1 || axis_size != tensor_size - 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "This is not weight generalization, will not changed");
    return NOT_CHANGED;
  }

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(prelugradNode);
  ge::NodePtr prelugradNoneNode = AddPReluGradNoneNode(prelugradNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "AddPReluGradNoneNode:check failed, fusion failed."),
                    return FAILED);

  AddReduceNode(prelugradNode, prelugradNoneNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "AddReduceNode:check failed, fusion failed."),
                    return FAILED);

  // unlink all control input of prelugradNode
  if (prelugradNode->GetInControlAnchor() != nullptr) {
    prelugradNode->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of prelugradNode
  for (auto inAnchor : prelugradNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  // remove prelugradNode from graph
  FUSION_PASS_CHECK(
      ge::GRAPH_SUCCESS != graph.RemoveNode(prelugradNode),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove fusedNode node[%s] failed",
                                     prelugradNode->GetName().c_str()),
      return FAILED);

  return SUCCESS;
}

REGISTER_PASS("PReluGradFusionPass", BUILT_IN_GRAPH_PASS, PReluGradFusionPass);
}  // namespace fe
