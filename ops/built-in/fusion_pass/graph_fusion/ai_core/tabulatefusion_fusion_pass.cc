/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file tabulatefusion_fusion_pass.cc
 * \brief TabulateFusion fusion pass(Parallel TabulateFusion)
 */
#include <cstdint>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "deep_md_fusion_pass_util.h"
#include "tabulatefusion_fusion_pass.h"

namespace fe {
static const std::string PATTERN_TABULATEFUSIONA = "TabulateFusion";
static const std::string OP_TYPE_TABULATEFUSION = "TabulateFusion";
static const std::string ATTR_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
static const std::string ATTR_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";
static const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
static const std::string ATTR_SPLIT_COUNT = "split_count";
static const std::string ATTR_SPLIT_INDEX = "split_index";
static const int SPLIT_COUNT = 2;
static const int NUM_3 = 3;

/*!
 * @brief Define pattern.
 * The graph struct need to adapt and target is shown as follows:
 *
 *        inputs                         inputs
 *          |                            /     \
 *    TabulateFusion    ==>    TabulateFusion   TabulateFusion
 *          |                           \        /
 *       outputs                         \      /
 *                                        \    /
 *                                        Concat
 *                                           |
 *                                       outputs
 * @return vector<FusionPattern*> All valid patterns.
 */
vector<FusionPattern *> TabulateFusionFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  string passName = "TabulateFusionFusionPass";
  FusionPattern *pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define pass pattern");
  pattern->AddOpDesc(PATTERN_TABULATEFUSIONA, {OP_TYPE_TABULATEFUSION})
      .SetOutput(PATTERN_TABULATEFUSIONA);
  patterns.push_back(pattern);
  return patterns;
}

Status TabulateFusionFusionPass::UpdateTabulateFusionNode(ge::NodePtr &tabulateNodeAic,
                                                          ge::NodePtr &tabulateNodeVec) const {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into Update TabulateFusion node.");
  ge::OpDescPtr tabulateAicDesc = tabulateNodeAic->GetOpDesc();
  ge::OpDescPtr tabulateVecDesc = tabulateNodeVec->GetOpDesc();
  FUSION_PASS_CHECK(
    tabulateAicDesc == nullptr || tabulateVecDesc == nullptr,
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get Aic or Vec OpDesc of TabulateFusion."),
    return PARAM_INVALID);

  ge::GeTensorDesc tabulateAicOutput = tabulateAicDesc->GetOutputDesc(0);
  ge::GeTensorDesc tabulateVecOutput = tabulateVecDesc->GetOutputDesc(0);
  ge::GeShape tabulateAicOutputShape = tabulateAicOutput.GetShape();
  ge::GeShape tabulateVecOutputShape = tabulateVecOutput.GetShape();
  FUSION_PASS_CHECK(tabulateAicOutputShape.GetDimNum() != NUM_3 || tabulateVecOutputShape.GetDimNum() != NUM_3,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "The output shape must be 3."),
                    return PARAM_INVALID);

  if (tabulateAicOutputShape.GetDim(0) > 0) {
    int64_t nloc = tabulateAicOutputShape.GetDim(0);
    int64_t ceilValue = (nloc + SPLIT_COUNT - 1) / SPLIT_COUNT;

    tabulateAicOutputShape.SetDim(0, ceilValue);
    tabulateAicOutput.SetShape(tabulateAicOutputShape);
    tabulateAicOutput.SetOriginShape(tabulateAicOutputShape);
    tabulateAicDesc->UpdateOutputDesc(0, tabulateAicOutput);

    tabulateVecOutputShape.SetDim(0, nloc - ceilValue);
    tabulateVecOutput.SetShape(tabulateVecOutputShape);
    tabulateVecOutput.SetOriginShape(tabulateVecOutputShape);
    tabulateVecDesc->UpdateOutputDesc(0, tabulateVecOutput);
  }

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to Update TabulateFusion node.");
  return SUCCESS;
}

Status TabulateFusionFusionPass::CreateConcatNodes(ge::ComputeGraph &graph, ge::NodePtr &tabulateNode,
                                                   vector<ge::NodePtr> &newTabulateNodes, ge::NodePtr &concatNode) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into create Concat nodes.");

  std::string concatNodeName = tabulateNode->GetName() + "/concatD";
  std::shared_ptr<ge::OpDesc> concatDesc = std::make_shared<ge::OpDesc>(concatNodeName, "ConcatD");
  FUSION_PASS_CHECK(concatDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Concat Node."),
                    return FAILED);

  ge::GeTensorDesc outputDesc = tabulateNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc outputAicDesc = newTabulateNodes[0]->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc outputVecDesc = newTabulateNodes[1]->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(concatDesc->AddInputDesc("x1", outputAicDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add x1 desc for Concat."),
                    return FAILED);
  FUSION_PASS_CHECK(concatDesc->AddInputDesc("x2", outputVecDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add x2 desc for Concat."),
                    return FAILED);
  FUSION_PASS_CHECK(concatDesc->AddOutputDesc(outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add y desc for Concat."),
                    return FAILED);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", 0);
  ge::AttrUtils::SetInt(concatDesc, "N", SPLIT_COUNT);

  concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add ConcatD Node to graph"),
                    return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newTabulateNodes[0]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output to concat x1."),
      return FAILED);
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newTabulateNodes[1]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(1)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from output to concat x2."),
      return FAILED);

  for (auto inAnchor : tabulateNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchor->UnlinkAll();
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(concatNode->GetOutDataAnchor(0), inAnchor),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from concat output to output node."),
        return FAILED);
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to create Add nodes.");
  return SUCCESS;
}

/*!
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TabulateFusionFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into TabulateFusion fusion pass.");

  ge::NodePtr tabulateNode = GetNodeFromMapping(PATTERN_TABULATEFUSIONA, mapping);
  FUSION_PASS_CHECK(tabulateNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get TabulateFusion Node"),
                    return PARAM_INVALID);
  if (DeepMdFusionPassUtil::CheckSplitInitInfo(FUSED_OP_TYPE, tabulateNode) != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "TabulateFusion fusion not changed");
    return NOT_CHANGED;
  }

  ge::NodePtr tabulateNodeAic = nullptr;
  ge::NodePtr tabulateNodeVec = nullptr;
  FUSION_PASS_CHECK(DeepMdFusionPassUtil::SplitNodeToAICoreAndVectorCore(FUSED_OP_TYPE, graph, tabulateNode,
                                                                         tabulateNodeAic, tabulateNodeVec) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to split TabulateFusion node"),
                    return FAILED);
  FUSION_PASS_CHECK(UpdateTabulateFusionNode(tabulateNodeAic, tabulateNodeVec) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to update TabulateFusion node"),
                    return FAILED);

  ge::NodePtr concatNode = nullptr;
  vector<ge::NodePtr> newTabulateNodes = {tabulateNodeAic, tabulateNodeVec};
  Status concatRet = CreateConcatNodes(graph, tabulateNode, newTabulateNodes, concatNode);
  FUSION_PASS_CHECK(concatRet != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create Concat nodes."),
                    return concatRet);

  FUSION_PASS_CHECK(DeepMdFusionPassUtil::ClearFusedNode(FUSED_OP_TYPE, graph, tabulateNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to clear TabulateFusion node"),
                    return FAILED);

  newNodes.push_back(tabulateNodeAic);
  newNodes.push_back(tabulateNodeVec);
  newNodes.push_back(concatNode);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to TabulateFusion fusion pass.");
  return SUCCESS;
}

REGISTER_PASS("TabulateFusionFusionPass", BUILT_IN_GRAPH_PASS, TabulateFusionFusionPass);
} // namespace fe
