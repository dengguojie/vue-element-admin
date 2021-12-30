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
 * \file fusedbatchnormgrad_fusion_pass.cpp
 * \brief fusedbatchnormgrad fusion pass
 *   (fusedbatchnormgrad --> BNTrainingReduceGrad & BNTrainingUpdateGrad)
 */
#include "fusedbatchnormgrad_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

namespace fe {
static const uint32_t NUMBER_2 = 2;
static const uint32_t NUMBER_3 = 3;
static const uint32_t NUMBER_4 = 4;
static const uint32_t NUMBER_0 = 0;
static const uint32_t NUMBER_5 = 5;
static const uint32_t NUMBER_1 = 1;

vector<FusionPattern*> FusedBatchNormGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern(PATTERN_FUSEDBATCHNORMGRAD);
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(
                    FUSED_OP_TYPE.c_str(), "new a pattern object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDBATCHNORMGRAD, {PASS_OP_TYPE_BATCHNORMGRAD}).SetOutput(PATTERN_FUSEDBATCHNORMGRAD);
  patterns.push_back(pattern);
  return patterns;
}

vector<ge::NodePtr> FusedBatchNormGradFusionPass::GetNodesFromMapping(const string& id, Mapping& mapping) {
  vector<ge::NodePtr> nodes;
  for (auto& item : mapping) {
    std::shared_ptr<OpDesc> opDesc = item.first;
    if (opDesc != nullptr && opDesc->id == id) {
      nodes = item.second;
    }
  }
  return nodes;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status FusedBatchNormGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                            vector<ge::NodePtr>& fusionNodes) {
  auto nodes = GetNodesFromMapping(PATTERN_FUSEDBATCHNORMGRAD, mapping);

  string streamLabelTmp;
  for (auto node : nodes) {
    string streamLabel;
    if (ge::AttrUtils::GetStr(node->GetOpDesc(), STREAM_LABEL, streamLabel)) {
      if (streamLabelTmp.empty()) {
        streamLabelTmp = streamLabel;
      } else if (!streamLabelTmp.empty() && streamLabelTmp != streamLabel) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "_stream_label(attr) not equal in all nodes and can not fusion.");
        return NOT_CHANGED;
      }
    } else {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNorm not have _stream_label attr.");
    }
  }

  for (auto& fusedBatchNormGrad : nodes) {
    if (!ge::AttrUtils::HasAttr(fusedBatchNormGrad->GetOpDesc(), BATCHNORMGRAD_ATTR_TRAINING)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The fused batch norm grad node does not have is training attr.");
      return NOT_CHANGED;
    }
    bool isTraining;
    ge::AttrUtils::GetBool(fusedBatchNormGrad->GetOpDesc(), BATCHNORMGRAD_ATTR_TRAINING, isTraining);

    FUSION_PASS_CHECK(isTraining == false, OP_LOGI(FUSED_OP_TYPE.c_str(), "is training must be true for fusion pass."),
                      return NOT_CHANGED);

    size_t outDataAnchorSize = fusedBatchNormGrad->GetAllOutDataAnchors().size();
    if (outDataAnchorSize < NUMBER_3) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNormGrad should have more than three outputs.");
      return NOT_CHANGED;
    }
    if (outDataAnchorSize == NUMBER_4) {
      OutDataAnchorPtr fusedOutAnchorThree = fusedBatchNormGrad->GetOutDataAnchor(NUMBER_3);
      if (!fusedOutAnchorThree->GetPeerInDataAnchors().empty()) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "only support three real outputs for fusedBatchNormGrad.");
        return NOT_CHANGED;
      }
    }
    if (outDataAnchorSize == NUMBER_5) {
      OutDataAnchorPtr fusedOutAnchorThree = fusedBatchNormGrad->GetOutDataAnchor(NUMBER_3);
      OutDataAnchorPtr fusedOutAnchorFour = fusedBatchNormGrad->GetOutDataAnchor(NUMBER_4);
      if (!fusedOutAnchorThree->GetPeerInDataAnchors().empty() ||
          !fusedOutAnchorFour->GetPeerInDataAnchors().empty()) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "only support three real outputs for fusedBatchNormGrad.");
        return NOT_CHANGED;
      }
    }
    if (outDataAnchorSize > NUMBER_5) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchNormGrad only should have less than five outputs.");
      return NOT_CHANGED;
    }

    ge::OpDescPtr xUpdateBnOp;
    std::string fusedBatchNormGradName = fusedBatchNormGrad->GetOpDesc()->GetName();

    FUSION_PASS_MAKE_SHARED(
        xUpdateBnOp = std::make_shared<ge::OpDesc>(fusedBatchNormGradName + "_Update", PASS_OP_TYPE_BNUPDATEGRAD),
        return INTERNAL_ERROR);

    ge::Node::Vistor<ge::NodePtr> input_node_vector = fusedBatchNormGrad->GetInDataNodes();
    ge::GeTensorDesc inputDesc0 = input_node_vector.at(NUMBER_0)->GetOpDesc()->GetOutputDesc(NUMBER_0);
    ge::GeTensorDesc inputDesc1 = input_node_vector.at(NUMBER_1)->GetOpDesc()->GetOutputDesc(NUMBER_0);
    ge::GeTensorDesc inputDesc3 = input_node_vector.at(NUMBER_3)->GetOpDesc()->GetOutputDesc(NUMBER_0);
    ge::GeTensorDesc inputDesc4 = input_node_vector.at(NUMBER_4)->GetOpDesc()->GetOutputDesc(NUMBER_0);
    xUpdateBnOp->AddInputDesc("grads", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_1));
    xUpdateBnOp->AddInputDesc("x", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_1));
    xUpdateBnOp->AddInputDesc("batch_mean", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xUpdateBnOp->AddInputDesc("batch_variance", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xUpdateBnOp->AddOutputDesc("diff_scale", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xUpdateBnOp->AddOutputDesc("diff_offset", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));

    ge::NodePtr xUpdateOp = graph.AddNode(xUpdateBnOp);
    fusionNodes.push_back(xUpdateOp);

    ge::OpDescPtr xReduceBnOp;
    FUSION_PASS_MAKE_SHARED(
        xReduceBnOp = std::make_shared<ge::OpDesc>(fusedBatchNormGradName + "_Reduce", PASS_OP_TYPE_BNREDUCEGRAD),
        return INTERNAL_ERROR);

    ge::GeTensorDesc inputDesc2 = input_node_vector.at(NUMBER_2)->GetOpDesc()->GetOutputDesc(NUMBER_0);
    xReduceBnOp->AddInputDesc("grads", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_1));
    xReduceBnOp->AddInputDesc("x", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_1));
    xReduceBnOp->AddInputDesc("diff_scale", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xReduceBnOp->AddInputDesc("diff_offset", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xReduceBnOp->AddInputDesc("scale", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xReduceBnOp->AddInputDesc("batch_mean", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xReduceBnOp->AddInputDesc("batch_variance", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_3));
    xReduceBnOp->AddOutputDesc("y", fusedBatchNormGrad->GetOpDesc()->GetInputDesc(NUMBER_1));
    ge::NodePtr xReduceOp = graph.AddNode(xReduceBnOp);
    fusionNodes.push_back(xReduceOp);

    // add edge for input of xUpdate
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(xUpdateOp->GetOutDataAnchor(NUMBER_0),
                      xReduceOp->GetInDataAnchor(NUMBER_2)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge between xUpdateOp output(0) and xReduceOp "
                                                     "input(2) failed."),
                      return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(xUpdateOp->GetOutDataAnchor(NUMBER_1),
                                                         xReduceOp->GetInDataAnchor(NUMBER_3)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                     "Add edge between xUpdateOp output(1) and xReduceOp "
                                                     "input(3) failed."),
                      return FAILED);

    float attrEpsilon = 0.0;
    FUSION_PASS_CHECK(
        !(ge::AttrUtils::GetFloat(fusedBatchNormGrad->GetOpDesc(), BATCHNORMGRAD_ATTR_EPSILON, attrEpsilon)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get epsilon attr failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(!(ge::AttrUtils::SetFloat(xUpdateOp->GetOpDesc(), BATCHNORMGRAD_ATTR_EPSILON, attrEpsilon)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                              "Set xUpdateOp Epsilon attr failed, exit "
                              "FusedBATCHNORMGRADGradFusionPass with status: failed."),
                      return FAILED);

    FUSION_PASS_CHECK(!(ge::AttrUtils::SetFloat(xReduceOp->GetOpDesc(), BATCHNORMGRAD_ATTR_EPSILON, attrEpsilon)),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                              "Set xReduceOp Epsilon attr failed, exit "
                              "FusedBATCHNORMGRADGradFusionPass with status: failed."),
                      return FAILED);

    // change fused_batch_norm_grad to x_update + x_reduce
    for (auto inDataAnchor : fusedBatchNormGrad->GetAllInDataAnchors()) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "fusedBatchNormGrad node[%s] has [%u] input anchors.",
              fusedBatchNormGrad->GetName().c_str(), fusedBatchNormGrad->GetAllInDataAnchors().size());
      FUSION_PASS_CHECK(nullptr == inDataAnchor->GetPeerOutAnchor(),
                        OP_LOGW(FUSED_OP_TYPE.c_str(), "inDataAnchor is null"), continue);
      if (inDataAnchor->GetIdx() == NUMBER_0) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), 
                                               xUpdateOp->GetInDataAnchor(NUMBER_0)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                                           "Add edge between grads input node and x_update node failed."),
                                           return FAILED);

        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xReduceOp->GetInDataAnchor(NUMBER_0)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), 
                                           "Add edge between grads input node and x_reduce node failed."),
                                           return FAILED);
      }
      if (inDataAnchor->GetIdx() == NUMBER_1) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xUpdateOp->GetInDataAnchor(1)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "Add edge between x input node and x_update node failed."), return FAILED);
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xReduceOp->GetInDataAnchor(1)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "Add edge between x input node and x_reduce node failed."), return FAILED);
      }
      if (inDataAnchor->GetIdx() == NUMBER_2) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xReduceOp->GetInDataAnchor(4)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
            "Add edge between scale input node and x_reduce node failed."),
            return FAILED);
      }
      if (inDataAnchor->GetIdx() == NUMBER_3) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xUpdateOp->GetInDataAnchor(NUMBER_2)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Add edge between batch_mean input node and x_update "
                    "node failed."),
            return FAILED);
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xReduceOp->GetInDataAnchor(NUMBER_5)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Add edge between batch_mean input node and x_reduce "
                    "node failed."),
            return FAILED);
      }
      if (inDataAnchor->GetIdx() == NUMBER_4) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xUpdateOp->GetInDataAnchor(NUMBER_3)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Add edge between batch_variance input node and "
                    "x_update node failed."),
            return FAILED);
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inDataAnchor->GetPeerOutAnchor(), xReduceOp->GetInDataAnchor(6)),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Add edge between batch_variance input node and "
                    "x_reduce node failed."),
            return FAILED);
      }
    }
    for (auto outDataAnchor : fusedBatchNormGrad->GetAllOutDataAnchors()) {
      auto outputIdex = outDataAnchor->GetIdx();
      if (static_cast<uint32_t>(outputIdex) == NUMBER_0) {
        for (auto peerInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::RemoveEdge(fusedBatchNormGrad->GetOutDataAnchor(outputIdex), peerInDataAnchor),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "remove edge between fusedBatchNormGrad output and "
                      "graph node failed."),
              return FAILED);
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(xReduceOp->GetOutDataAnchor(outputIdex), peerInDataAnchor),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add edge between xReduceOp input and graph node failed."), return FAILED);
        }
        continue;
      } else if (static_cast<uint32_t>(outputIdex) < NUMBER_3) {
        for (auto peerInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::RemoveEdge(fusedBatchNormGrad->GetOutDataAnchor(outputIdex), peerInDataAnchor),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "remove edge between fusedBatchNormGrad output and "
                      "graph node failed."),
              return FAILED);
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::AddEdge(xUpdateOp->GetOutDataAnchor(outputIdex - 1), peerInDataAnchor),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add edge between xUpdateOp input and graph node failed."), return FAILED);
        }
      } else if (static_cast<uint32_t>(outputIdex) > NUMBER_2) {
        for (auto peerInDataAnchor : outDataAnchor->GetPeerInDataAnchors()) {
          FUSION_PASS_CHECK(
              SUCCESS != ge::GraphUtils::RemoveEdge(fusedBatchNormGrad->GetOutDataAnchor(outputIdex), peerInDataAnchor),
              VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "remove edge between fusedBatchNormGrad output and "
                      "graph node failed."),
              return FAILED);
        }
      }
    }
    for (auto fusedToReducePeerOutControlAnchor :
         fusedBatchNormGrad->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(fusedToReducePeerOutControlAnchor, xReduceOp->GetInControlAnchor()),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's control edge to fusion node:%s's control edge failed.",
                  fusedBatchNormGrad->GetName().c_str(), xReduceOp->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control edge to fusion node:%s's control edge.",
              fusedBatchNormGrad->GetName().c_str(), xReduceOp->GetName().c_str());
    }
    for (auto fusedToUpdatePeerOutControlAnchor :
         fusedBatchNormGrad->GetInControlAnchor()->GetPeerOutControlAnchors()) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(fusedToUpdatePeerOutControlAnchor, xUpdateOp->GetInControlAnchor()),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's control edge to fusion node:%s's control edge failed.",
                  fusedBatchNormGrad->GetName().c_str(), xUpdateOp->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control edge to fusion node:%s's control edge.",
              fusedBatchNormGrad->GetName().c_str(), xUpdateOp->GetName().c_str());
    }
    for (auto fusedPeerInControlAnchor : fusedBatchNormGrad->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(xUpdateOp->GetOutControlAnchor(), fusedPeerInControlAnchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "Add edge from fused node:%s's control edge to fusion node:%s's control edge failed.",
                          fusedBatchNormGrad->GetName().c_str(), xUpdateOp->GetName().c_str()),
      return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control edge to fusion node:%s's control edge.",
              fusedBatchNormGrad->GetName().c_str(), xUpdateOp->GetName().c_str());

      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(xReduceOp->GetOutControlAnchor(), fusedPeerInControlAnchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "Add edge from fused node:%s's control edge to fusion node:%s's control edge failed.",
                          fusedBatchNormGrad->GetName().c_str(), xReduceOp->GetName().c_str()),
      return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's control edge to fusion node:%s's control edge.",
              fusedBatchNormGrad->GetName().c_str(), xReduceOp->GetName().c_str());
    }

    if (fusedBatchNormGrad->GetInControlAnchor() != nullptr) {
      fusedBatchNormGrad->GetInControlAnchor()->UnlinkAll();
    }
    if (fusedBatchNormGrad->GetOutControlAnchor() != nullptr) {
      fusedBatchNormGrad->GetOutControlAnchor()->UnlinkAll();
    }
    // set attr(_stream_label)
    if (!streamLabelTmp.empty()) {
      if (!ge::AttrUtils::SetStr(xUpdateOp->GetOpDesc(), STREAM_LABEL, streamLabelTmp)) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "BNTrainingUpdateGrad set _stream_label error, fusion failed.");
        return FAILED;
      }
      if (!ge::AttrUtils::SetStr(xReduceOp->GetOpDesc(), STREAM_LABEL, streamLabelTmp)) {
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "BNTrainingReduceGrad set _stream_label error, fusion failed.");
        return FAILED;
      }
    }

    FUSION_PASS_CHECK(SUCCESS != graph.RemoveNode(fusedBatchNormGrad),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Remove fusedBatchNormGrad node failed."), return FAILED);
    FUSION_PASS_CHECK(xUpdateOp->GetOpDesc() == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "xUpdateOp is null, fusion failed."), return PARAM_INVALID);
    FUSION_PASS_CHECK(xReduceOp->GetOpDesc() == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "xReduceOp is null, fusion failed."), return PARAM_INVALID);
  }
  return SUCCESS;
}
REGISTER_PASS("FusedBatchNormGradFusionPass", BUILT_IN_GRAPH_PASS, FusedBatchNormGradFusionPass);
}  // namespace fe
