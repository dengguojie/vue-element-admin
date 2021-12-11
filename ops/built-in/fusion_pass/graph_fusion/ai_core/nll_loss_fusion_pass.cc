/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 *
 * @brief NLLLoss fusion pass(NLLLoss --> NLLLoss(sum) & reduce(mean))
 *
 */

#include "nll_loss_fusion_pass.h"

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
#include "common/util/platform_info.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "NLLLoss";
static const std::string PATTERN_FUSEDNODE = "NLLLoss";

static const vector<vector<int64_t>> SUPPORT_CASES = {
    {4210704, 21},
    {8388608, 19},
};

static const vector<string> NOT_SUPPORT_AIC = {
    "AIC-C-100",
};

vector<FusionPattern*> NLLLossFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("NLLLossFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

ge::NodePtr NLLLossFusionPass::AddNLLLossSumNode(ge::NodePtr nll_loss_node,
                                                 ge::ComputeGraph& graph,
                                                 vector<ge::NodePtr>& new_nodes,
                                                 bool &fail_status) {
  ge::OpDescPtr nll_loss_desc = nll_loss_node->GetOpDesc();

  // create nll_loss_sum desc
  ge::OpDescPtr nll_loss_sum_desc = AttrUtils::CloneOpDesc(nll_loss_desc);
  nll_loss_sum_desc->UpdateInputDesc(0, nll_loss_desc->GetInputDesc(0));
  nll_loss_sum_desc->UpdateInputDesc(1, nll_loss_desc->GetInputDesc(1));
  nll_loss_sum_desc->UpdateInputDesc(2, nll_loss_desc->GetInputDesc(2));
  nll_loss_sum_desc->UpdateOutputDesc(0, nll_loss_desc->GetOutputDesc(0));
  nll_loss_sum_desc->UpdateOutputDesc(1, nll_loss_desc->GetOutputDesc(1));
    
  // attr
  ge::AttrUtils::SetStr(nll_loss_sum_desc, "reduction", "sum");

  // create nll_loss_sum node
  ge::NodePtr nll_loss_sum_node = graph.AddNode(nll_loss_sum_desc);
  FUSION_PASS_CHECK(nll_loss_sum_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                    nll_loss_sum_node->GetName().c_str()), fail_status=true);
  new_nodes.push_back(nll_loss_sum_node);
  
  // Edge
  for (unsigned int i = 0; i < nll_loss_node->GetAllInDataAnchors().size(); i++) {
    ge::GraphUtils::AddEdge(nll_loss_node->GetInDataAnchor(i)->GetPeerOutAnchor(), nll_loss_sum_node->GetInDataAnchor(i));
  }

  for (unsigned int i = 0; i < nll_loss_node->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
    ge::GraphUtils::AddEdge(nll_loss_node->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                            nll_loss_sum_node->GetInControlAnchor());
  }
  
  return nll_loss_sum_node;
}

ge::NodePtr NLLLossFusionPass::AddDivNode(ge::NodePtr nll_loss_node,
                                          ge::NodePtr nll_loss_sum_node,
                                          ge::ComputeGraph& graph,
                                          vector<ge::NodePtr>& new_nodes,
                                          bool &fail_status) {
  // create div desc
  ge::OpDescPtr div_desc;

  FUSION_PASS_MAKE_SHARED(
        (div_desc = std::make_shared<ge::OpDesc>(nll_loss_node->GetName() + "Div", "Div")),
        fail_status = true; return nullptr);
  
  // input
  ge::GeTensorDesc x1_tensor_desc = nll_loss_sum_node->GetOpDesc()->GetOutputDesc(0).Clone();
  ge::GeTensorDesc x2_tensor_desc = nll_loss_sum_node->GetOpDesc()->GetOutputDesc(1).Clone();

  div_desc->AddInputDesc("x1", x1_tensor_desc);
  div_desc->AddInputDesc("x2", x2_tensor_desc);

  // output
  ge::GeTensorDesc y_tensor_desc = nll_loss_node->GetOpDesc()->GetOutputDesc(0).Clone();
  div_desc->AddOutputDesc("y", y_tensor_desc);

  // create div node
  ge::NodePtr div_node = graph.AddNode(div_desc);
  FUSION_PASS_CHECK(div_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusionNode:%s is null, fusion failed.",
                    div_node->GetName().c_str()), fail_status=true);
  new_nodes.push_back(div_node);
  
  // Edge
  ge::GraphUtils::AddEdge(nll_loss_sum_node->GetOutDataAnchor(0), div_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(nll_loss_sum_node->GetOutDataAnchor(1), div_node->GetInDataAnchor(1));
  
  if (nll_loss_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : nll_loss_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(div_node->GetOutDataAnchor(0), inAnchorPtr);
    }
  }
  
  if (nll_loss_node->GetOutDataAnchor(1)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : nll_loss_node->GetOutDataAnchor(1)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      ge::GraphUtils::AddEdge(nll_loss_sum_node->GetOutDataAnchor(1), inAnchorPtr);
    }
  }
  
  return div_node;
}

Status NLLLossFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  bool fail_status = false;
  bool is_support_aic = true;
  bool is_unknown_shape = false;
  bool is_match = false;
  string reduction = "";
  string reduction_attr = "reduction";

  // get nll_loss_node
  ge::NodePtr nll_loss_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(nll_loss_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "nll_loss_node is null, fusion failed."),
                    return PARAM_INVALID);
  
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(nll_loss_node);
  if (GRAPH_SUCCESS != op.GetAttr(reduction_attr, reduction)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "can't get reduction attr.");
    return FAILED;
  }

  if (reduction != "mean") {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "reduction is not mean.");
    return NOT_CHANGED;
  }

  PlatformInfo platform_info;
  OptionalInfo optional_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info,
                                                                                     optional_info) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get platform info failed."), return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "aic version is [%s].", platform_info.str_info.aic_version.c_str());
  for (auto aic_version : NOT_SUPPORT_AIC) {
    if (platform_info.str_info.aic_version == aic_version) {
      is_support_aic = false;
      break;
    }
  }

  if (!is_support_aic) {
    // check nllloss support dynamic
    if (GRAPH_SUCCESS != ge::NodeUtils::GetNodeUnknownShapeStatus(*(nll_loss_node.get()), is_unknown_shape)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "can't get unknown shape status.");
      return FAILED;
    }

    if (!is_unknown_shape) {
      GeTensorDesc x_desc = nll_loss_node->GetOpDesc()->GetInputDesc(0);
      vector<int64_t> x_shape = x_desc.GetShape().GetDims();
      for (auto shape : SUPPORT_CASES) {
        if (x_shape == shape) {
          is_match = true;
          break;
        }
      }

      if (!is_match) {
        auto weight_desc = nll_loss_node->GetOpDesc()->MutableInputDesc(2);
        // weight optional
        if (weight_desc != nullptr) {
          OP_LOGD(FUSED_OP_TYPE.c_str(), "is not dynamic shape or not match static shape or not optional weight.");
          return NOT_CHANGED;
        }
      }
    }
  }

  ge::NodePtr nll_loss_sum_node = AddNLLLossSumNode(nll_loss_node, graph, new_nodes, fail_status);
  FUSION_PASS_CHECK(fail_status, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                    "AddNLLLossSumNode:check failed, fusion failed."), return FAILED);

  AddDivNode(nll_loss_node, nll_loss_sum_node, graph, new_nodes, fail_status);
  FUSION_PASS_CHECK(fail_status, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                    "AddDivNode:check failed, fusion failed."), return FAILED);

  // unlink all control input of nll_loss_node
  if (nll_loss_node->GetInControlAnchor() != nullptr) {
    nll_loss_node->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all input of nll_loss_node
  for (auto inAnchor : nll_loss_node->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  // remove nll_loss_node from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(nll_loss_node), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "remove fusedNode node[%s] failed", nll_loss_node->GetName().c_str()), return FAILED);

  return SUCCESS;
}

REGISTER_PASS("NLLLossFusionPass", BUILT_IN_GRAPH_PASS, NLLLossFusionPass);
}
