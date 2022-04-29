/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file batch_multi_class_nms_insert_nms_bucketize_fusion_pass.cpp
 * \brief batch_multi_class_nms_insert_nms_bucketize fusion pass
 */
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include "op_log.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"
#include "batch_multi_class_nms_insert_nms_bucketize_fusion_pass.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeBatchMultiClassNonMaxSuppressionInsertNMSBucketize";
static const string FUSED_NODE = "BatchMultiClassNonMaxSuppression";

vector<FusionPattern*> BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      new (std::nothrow) FusionPattern("BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

void BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass::GetNMSNodeIndex(ge::ComputeGraph& graph,
                                                                                   const ge::NodePtr& checkNode,
                                                                                   int64_t& nmsNodeIndex) {
  for (const auto& node : graph.GetDirectNode()) {
    if (node->GetType() == "BatchMultiClassNonMaxSuppression") {
      nmsNodeIndex++;
      if (node->GetOpDesc()->GetName() == checkNode->GetOpDesc()->GetName()) {
        break;
      }
    }
  }
}

Status BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass::UpdateConstNodeValueInCaseSubgraph(
    const ge::NodePtr& bucketizeNode, const int64_t outputIndex, ge::InDataAnchorPtr& inDataAnchor,
    ge::ComputeGraph& graph) {
  auto outNode = inDataAnchor->GetOwnerNode();
  auto outShape = bucketizeNode->GetOpDesc()->GetOutputDesc(outputIndex).GetShape();
  std::vector<std::pair<int64_t, int64_t>> shapeRange;
  bucketizeNode->GetOpDesc()->GetOutputDesc(outputIndex).GetShapeRange(shapeRange);
  auto inDesc = outNode->GetOpDesc()->MutableInputDesc(inDataAnchor->GetIdx());
  inDesc->SetShape(outShape);

  FUSION_PASS_CHECK(inDesc->SetShapeRange(shapeRange) == ge::GRAPH_FAILED,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set bucketizeNode peerout Node shaperange failed, fusion failed."),
                    return FAILED);
  for (auto& nameIter : outNode->GetOpDesc()->GetSubgraphInstanceNames()) {
    auto subgraph = graph.GetSubgraph(nameIter);
    for (auto& node : subgraph->GetDirectNode()) {
      if (node->GetType() == "Data") {
        int32_t parentNodeIndex = 0;
        ge::AttrUtils::GetInt(node->GetOpDesc(), "_parent_node_index", parentNodeIndex);
        if (parentNodeIndex == inDataAnchor->GetIdx()) {
          node->GetOpDesc()->MutableInputDesc(0)->SetShape(outShape);
          node->GetOpDesc()->MutableOutputDesc(0)->SetShape(outShape);
          node->GetOpDesc()->MutableInputDesc(0)->SetShapeRange(shapeRange);
          node->GetOpDesc()->MutableOutputDesc(0)->SetShapeRange(shapeRange);
          for (auto peer : node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
            auto peerNode = peer->GetOwnerNode();
            peerNode->GetOpDesc()->MutableInputDesc(peer->GetIdx())->SetShape(outShape);
            FUSION_PASS_CHECK(
                peerNode->GetOpDesc()->MutableInputDesc(peer->GetIdx())->SetShapeRange(shapeRange) == ge::GRAPH_FAILED,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "Set shaperange failed, fusion failed."), return FAILED);
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                                            vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);

  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "The fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  int64_t nmsNodeIndex = 0;
  GetNMSNodeIndex(graph, fusedNode, nmsNodeIndex);
  // Only handle the second nmsNode
  if (nmsNodeIndex == 2U) {
    // Define NonMaxSuppressionBucketize's input tensorDesc
    auto transposeNode = fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode();
    auto reduceNode = fusedNode->GetOutDataAnchor(3)->GetPeerInDataAnchors().at(0)->GetOwnerNode();

    ge::GeTensorDesc nmsBucketizeInputDesc0 = transposeNode->GetOpDesc()->GetOutputDesc(0);
    ge::GeTensorDesc nmsBucketizeInputDesc1 = fusedNode->GetOpDesc()->GetOutputDesc(1);
    ge::GeTensorDesc nmsBucketizeInputDesc2 = fusedNode->GetOpDesc()->GetOutputDesc(2);
    ge::GeTensorDesc nmsBucketizeInputDesc3 = reduceNode->GetOpDesc()->GetOutputDesc(0);

    // define NonMaxSuppressionBucketize OpDesc
    std::shared_ptr<ge::OpDesc> nmsBucketizeDesc = nullptr;
    std::string nmsBucketizeName = fusedNode->GetName() + "_NonMaxSupressionBucketize";
    nmsBucketizeDesc = std::make_shared<ge::OpDesc>(nmsBucketizeName, "NonMaxSuppressionBucketize");
    FUSION_PASS_CHECK(nmsBucketizeDesc == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add nmsBucketizeDesc failed, fusion failed."), return FAILED);

    // Add InputDesc to NonMaxSuppressionBucketize OpDesc
    FUSION_PASS_CHECK(nmsBucketizeDesc->AddInputDesc("input_nmsed_boxes", nmsBucketizeInputDesc0) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add input_nmsed_boxes failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(nmsBucketizeDesc->AddInputDesc("input_nmsed_score", nmsBucketizeInputDesc1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add input_nmsed_score failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(nmsBucketizeDesc->AddInputDesc("input_nmsed_class", nmsBucketizeInputDesc2) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add input_nmsed_class failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(nmsBucketizeDesc->AddInputDesc("input_nmsed_num", nmsBucketizeInputDesc3) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add input_nmsed_num failed, fusion failed."), return FAILED);
    // Add OutputDesc to NonMaxSuppressionBucketize OpDesc
    FUSION_PASS_CHECK(nmsBucketizeDesc->AddOutputDesc("output_nmsed_boxes", nmsBucketizeInputDesc0) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add output_nmsed_boxes failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(nmsBucketizeDesc->AddOutputDesc("output_nmsed_score", nmsBucketizeInputDesc1) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add output_nmsed_score failed, fusion failed."), return FAILED);
    FUSION_PASS_CHECK(nmsBucketizeDesc->AddOutputDesc("output_nmsed_class", nmsBucketizeInputDesc2) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add output_nmsed_class failed, fusion failed."), return FAILED);

    // Add NonMaxSupressionBucketize node, infer output shape
    auto nmsBucketizeNode = graph.AddNode(nmsBucketizeDesc);
    FUSION_PASS_CHECK(!nmsBucketizeNode, OP_LOGE(FUSED_OP_TYPE.c_str(), "Add nmsBucketizeNode failed, fusion failed."),
                      return FAILED);
    FUSION_PASS_CHECK(nmsBucketizeNode->InferShapeAndType() != ge::GRAPH_SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "The nmsBucketizeNode InferShape failed, fusion failed."),
                      return FAILED);

    // Insert NonMaxSupressionBucketize
    auto transposeNodeOutputAnchor = transposeNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(transposeNodeOutputAnchor == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "The transposeNodeOutputAnchor is nullptr, fusion failed."),
                      return FAILED);
    for (auto inDataAnchor : transposeNodeOutputAnchor->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(transposeNodeOutputAnchor, inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveEdge failed, fusion failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(nmsBucketizeNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed, fusion failed."), return FAILED);
      FUSION_PASS_CHECK(UpdateConstNodeValueInCaseSubgraph(nmsBucketizeNode, 0, inDataAnchor, graph) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateConstNodeValueInCaseSubgraph failed, fusion failed."),
                        return FAILED);
    }
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(transposeNodeOutputAnchor, nmsBucketizeNode->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed, fusion failed."), return FAILED);

    auto fusedNodeOutputAnchor1 = fusedNode->GetOutDataAnchor(1);
    FUSION_PASS_CHECK(fusedNodeOutputAnchor1 == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "The fusedNodeOutputAnchor1 is nullptr, fusion failed."),
                      return FAILED);
    for (auto inDataAnchor : fusedNodeOutputAnchor1->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNodeOutputAnchor1, inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveEdge failed, fusion failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(nmsBucketizeNode->GetOutDataAnchor(1), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed, fusion failed."), return FAILED);
      FUSION_PASS_CHECK(UpdateConstNodeValueInCaseSubgraph(nmsBucketizeNode, 1, inDataAnchor, graph) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateConstNodeValueInCaseSubgraph failed, fusion failed."),
                        return FAILED);
    }
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNodeOutputAnchor1, nmsBucketizeNode->GetInDataAnchor(1)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed, fusion failed."), return FAILED);

    auto fusedNodeOutputAnchor2 = fusedNode->GetOutDataAnchor(2);
    FUSION_PASS_CHECK(fusedNodeOutputAnchor2 == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "The fusedNodeOutputAnchor2 is nullptr, fusion failed."),
                      return FAILED);
    for (auto inDataAnchor : fusedNodeOutputAnchor2->GetPeerInDataAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNodeOutputAnchor2, inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveEdge failed, fusion failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(nmsBucketizeNode->GetOutDataAnchor(2), inDataAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed, fusion failed."), return FAILED);
      FUSION_PASS_CHECK(UpdateConstNodeValueInCaseSubgraph(nmsBucketizeNode, 2, inDataAnchor, graph) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "UpdateConstNodeValueInCaseSubgraph failed, fusion failed."),
                        return FAILED);
    }
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNodeOutputAnchor2, nmsBucketizeNode->GetInDataAnchor(2)) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed, fusion failed."), return FAILED);

    auto strideSliceDNodeOutputAnchor = reduceNode->GetOutDataAnchor(0);
    FUSION_PASS_CHECK(strideSliceDNodeOutputAnchor == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "The strideSliceDNodeOutputAnchor is nullptr, fusion failed."),
                      return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(strideSliceDNodeOutputAnchor, nmsBucketizeNode->GetInDataAnchor(3)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "AddEdge failed, fusion failed."), return FAILED);
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass fusion end.");
  return SUCCESS;
}

REGISTER_PASS("BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass", BUILT_IN_GRAPH_PASS,
              BatchMultiClassNonMaxSuppressionInsertNMSBucketizeFusionPass);
}  // namespace fe