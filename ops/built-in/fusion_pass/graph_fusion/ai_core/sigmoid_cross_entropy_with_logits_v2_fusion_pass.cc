/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * @brief SigmoidCrossEntropyWithLogitsV2 fusion pass(SigmoidCrossEntropyWithLogitsV2 --> SigmoidCrossEntropyWithLogitsV2 & reduce(sum/mean))
 *
 */

#include "sigmoid_cross_entropy_with_logits_v2_fusion_pass.h"

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

static const char* FUSED_NODE = "SigmoidCrossEntropyWithLogitsV2";
static const std::string PATTERN_FUSEDNODE = "SigmoidCrossEntropyWithLogitsV2";

vector<FusionPattern*> SigmoidCrossEntropyWithLogitsV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("SigmoidCrossEntropyWithLogitsV2FusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr SigmoidCrossEntropyWithLogitsV2FusionPass::AddSigmoidNoneNode(ge::NodePtr sigmoidNode,
                                                                          ge::ComputeGraph& graph,
                                                                          vector<ge::NodePtr>& newNodes,
                                                                          bool &failStatus) {
  ge::OpDescPtr sigmoidDesc = sigmoidNode->GetOpDesc();

  // create sigmoid_none desc
  ge::OpDescPtr sigmoidNoneDesc = AttrUtils::CloneOpDesc(sigmoidDesc);
  std::map<string, uint32_t> input_name_id;
  // update node inputname
  input_name_id["predict"] = 0;
  input_name_id["target"] = 1;
  input_name_id["weight"] = 2;
  input_name_id["pos_weight"] = 3;
  sigmoidNoneDesc->UpdateInputName(input_name_id);
  std::map<string, uint32_t> out_name_idx;
  // update node output name
  out_name_idx["loss"] = 0;
  sigmoidNoneDesc->UpdateOutputName(out_name_idx);
  // input
  ge::GeTensorDesc siginput_tensor_desc = sigmoidNoneDesc->GetInputDesc(0);
  ge::Format input_origin_format = siginput_tensor_desc.GetOriginFormat();
  ge::Format input_format = siginput_tensor_desc.GetFormat();
  // update output shape
  ge::GeTensorDesc sigm_output_tensor_desc = sigmoidNoneDesc->GetOutputDesc(0);
  sigm_output_tensor_desc.SetOriginShape(siginput_tensor_desc.GetOriginShape());
  sigm_output_tensor_desc.SetShape(siginput_tensor_desc.GetShape());
  sigm_output_tensor_desc.SetOriginFormat(input_origin_format);
  sigm_output_tensor_desc.SetFormat(input_format);
  sigmoidNoneDesc->UpdateOutputDesc("loss", sigm_output_tensor_desc);
  ge::AttrUtils::SetStr(sigmoidNoneDesc, "reduction", "none");

  // create sigmoid_none node
  ge::NodePtr sigmoidNoneNode = graph.AddNode(sigmoidNoneDesc);

  FUSION_PASS_CHECK(sigmoidNoneNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                    sigmoidNoneNode->GetName().c_str()), failStatus=true);

  newNodes.push_back(sigmoidNoneNode);

  // Edge
  for (unsigned int i = 0; i < sigmoidNode->GetAllInDataAnchors().size(); i++) {
    ge::GraphUtils::AddEdge(sigmoidNode->GetInDataAnchor(i)->GetPeerOutAnchor(), sigmoidNoneNode->GetInDataAnchor(i));
  }

  for (unsigned int i = 0; i < sigmoidNode->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    ge::GraphUtils::AddEdge(sigmoidNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                            sigmoidNoneNode->GetInControlAnchor());
  }
  
  return sigmoidNoneNode;
}

ge::NodePtr SigmoidCrossEntropyWithLogitsV2FusionPass::AddReduceNode(ge::NodePtr sigmoidNode,
                                                                     ge::NodePtr sigmoidNoneNode,
                                                                     ge::ComputeGraph& graph,
                                                                     vector<ge::NodePtr>& newNodes,
                                                                     bool &failStatus, string reduction) {
  // create reduce desc
  ge::OpDescPtr reduceDesc;

  if (reduction == "sum") {
    FUSION_PASS_MAKE_SHARED(
        (reduceDesc = std::make_shared<ge::OpDesc>(sigmoidNode->GetName() + "ReduceSumD", "ReduceSumD")),
        failStatus = true; return nullptr);
  } else {
    FUSION_PASS_MAKE_SHARED(
        (reduceDesc = std::make_shared<ge::OpDesc>(sigmoidNode->GetName() + "ReduceMeanD", "ReduceMeanD")),
        failStatus = true; return nullptr);
  }

  // input
  ge::GeTensorDesc inputTensorDesc = sigmoidNoneNode->GetOpDesc()->GetOutputDesc(0).Clone();
  inputTensorDesc.SetDataType(ge::DT_FLOAT);
  reduceDesc->AddInputDesc("x", inputTensorDesc);

  // output
  ge::GeTensorDesc outputTensorDesc = sigmoidNode->GetOpDesc()->GetOutputDesc(0).Clone();
  outputTensorDesc.SetDataType(ge::DT_FLOAT);
  reduceDesc->AddOutputDesc("y", outputTensorDesc);

  // attr
  ge::AttrUtils::SetListInt(reduceDesc, "axes", {});
  ge::AttrUtils::SetBool(reduceDesc, "keep_dims", false);

  // create reduce node
  ge::NodePtr reduceNode = graph.AddNode(reduceDesc);
  FUSION_PASS_CHECK(reduceNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                    reduceNode->GetName().c_str()), failStatus=true);
  newNodes.push_back(reduceNode);
  
  // Edge
  ge::GraphUtils::AddEdge(sigmoidNoneNode->GetOutDataAnchor(0), reduceNode->GetInDataAnchor(0));
  for (unsigned int i = 0; i < sigmoidNode->GetAllOutDataAnchors().size(); i++) {
    if (sigmoidNode->GetOutDataAnchor(i)->GetPeerInDataAnchors().size() > 0) {
      for (InDataAnchorPtr inAnchorPtr : sigmoidNode->GetOutDataAnchor(i)->GetPeerInDataAnchors()) {
        inAnchorPtr->UnlinkAll();
        ge::GraphUtils::AddEdge(reduceNode->GetOutDataAnchor(0), inAnchorPtr);
      }
    }
  }
  
  return reduceNode;
}

Status SigmoidCrossEntropyWithLogitsV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                         vector<ge::NodePtr>& newNodes) {
  bool failStatus = false;
  string reduction = "";
  string reductionAttr = "reduction";
  string reductionNone = "none";

  // get sigmoidNode
  ge::NodePtr sigmoidNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(sigmoidNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "sigmoidNode is null, fusion failed."),
                    return PARAM_INVALID);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(sigmoidNode);

  if (GRAPH_SUCCESS != op.GetAttr(reductionAttr, reduction)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "can't get reduction attr.");
    return FAILED;
  }

  if (reduction == reductionNone) {
    return SUCCESS;
  }

  ge::NodePtr sigmoidNoneNode = AddSigmoidNoneNode(sigmoidNode, graph, newNodes, failStatus);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                    "AddSigmoidNoneNode:check failed, fusion failed."), return FAILED);

  AddReduceNode(sigmoidNode, sigmoidNoneNode, graph, newNodes, failStatus, reduction);
  FUSION_PASS_CHECK(failStatus, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                    "AddReduceNode:check failed, fusion failed."), return FAILED);

  // unlink all control input of sigmoidNode
  if (sigmoidNode->GetInControlAnchor() != nullptr) {
    sigmoidNode->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of sigmoidNode
  for (auto inAnchor : sigmoidNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  // remove sigmoidNode from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(sigmoidNode), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "remove fusedNode node[%s] failed", sigmoidNode->GetName().c_str()), return FAILED);

  return SUCCESS;
}

REGISTER_PASS("SigmoidCrossEntropyWithLogitsV2FusionPass", BUILT_IN_GRAPH_PASS, SigmoidCrossEntropyWithLogitsV2FusionPass);
}
