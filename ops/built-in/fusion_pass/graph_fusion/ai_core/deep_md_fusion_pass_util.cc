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
 * \file deep_md_fusion_pass_util.cc
 * \brief Deep MD fusion pass util
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

namespace fe {
static const std::string ATTR_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
static const std::string ATTR_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";
static const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
static const std::string ATTR_SPLIT_COUNT = "split_count";
static const std::string ATTR_SPLIT_INDEX = "split_index";
const int64_t SPLIT_COUNT_VALUE = 2;
Status DeepMdFusionPassUtil::CheckSplitInitInfo(const std::string& fusedOpType, const ge::NodePtr& node) {
  OP_LOGD(fusedOpType.c_str(), "Enter into CheckSplitInitInfo");

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  int32_t splitCount;
  FUSION_PASS_CHECK(op.GetAttr(ATTR_SPLIT_COUNT.c_str(), splitCount) == ge::GRAPH_SUCCESS && splitCount == 2,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "split_count should not be 2 before fusion"),
                    return PARAM_INVALID);

  OP_LOGD(fusedOpType.c_str(), "End to CheckSplitInitInfo");
  return SUCCESS;
}

Status DeepMdFusionPassUtil::SplitNodeToAICoreAndVectorCore(const std::string& fusedOpType, ge::ComputeGraph& graph,
                                                            const ge::NodePtr& fusedNode, ge::NodePtr& nodeAic,
                                                            ge::NodePtr& nodeVec) {
  OP_LOGD(fusedOpType.c_str(), "Enter into SplitNodeToAICoreAndVectorCore");

  ge::OpDescPtr opDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(opDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to get OpDesc of fused node"),
                    return PARAM_INVALID);

  ge::OpDescPtr opDescAic = AttrUtils::CopyOpDesc(opDesc);
  FUSION_PASS_CHECK(opDescAic == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Faile to create opDesc(AI Core)"),
                    return FAILED);
  opDescAic->SetName(opDesc->GetName() + "_aicore");

  // Set attribute of AICore flag
  ge::AttrUtils::SetStr(opDescAic, ATTR_OP_SPECIFIED_ENGINE_NAME, "AIcoreEngine");
  ge::AttrUtils::SetStr(opDescAic, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "AIcoreEngine");
  ge::AttrUtils::SetInt(opDescAic, ATTR_SPLIT_COUNT, SPLIT_COUNT_VALUE);
  ge::AttrUtils::SetInt(opDescAic, ATTR_SPLIT_INDEX, 0);

  nodeAic = graph.AddNode(opDescAic);
  FUSION_PASS_CHECK(nodeAic == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add node(AI Core) to graph"),
                    return FAILED);

  ge::OpDescPtr opDescVec = AttrUtils::CopyOpDesc(opDesc);
  FUSION_PASS_CHECK(opDescVec == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Faile to create opDesc(Vector Core)"),
                    return FAILED);
  opDescVec->SetName(opDesc->GetName() + "_vectorcore");

  // Set attribute of VectorCore flag
  ge::AttrUtils::SetStr(opDescVec, ATTR_OP_SPECIFIED_ENGINE_NAME, "VectorEngine");
  ge::AttrUtils::SetStr(opDescVec, ATTR_OP_SPECIFIED_KERNEL_LIB_NAME, "VectorEngine");
  ge::AttrUtils::SetInt(opDescVec, ATTR_SPLIT_COUNT, SPLIT_COUNT_VALUE);
  ge::AttrUtils::SetInt(opDescVec, ATTR_SPLIT_INDEX, (int64_t)1);
  ge::AttrUtils::SetStr(opDescVec, ATTR_NAME_STREAM_LABEL, "VectorEngine");

  nodeVec = graph.AddNode(opDescVec);
  FUSION_PASS_CHECK(nodeVec == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add node(Vector Core) to graph"),
                    return FAILED);
  uint32_t index = 0;
  for (auto inputDesc : opDesc->GetAllInputsDesc()) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(index)->GetPeerOutAnchor(), nodeAic->GetInDataAnchor(index)),
        VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add edge from fused node to AICore node"),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(index)->GetPeerOutAnchor(),
                                nodeVec->GetInDataAnchor(index)) != ge::GRAPH_SUCCESS,
        VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add edge from fused node to VectorCore node"),
        return FAILED);
    index++;
  }

  OP_LOGD(fusedOpType.c_str(), "End to SplitNodeToAICoreAndVectorCore");
  return SUCCESS;
}

Status DeepMdFusionPassUtil::CreateAddNodeAfterSplitNode(const std::string& fusedOpType, ge::ComputeGraph& graph,
                                                         ge::NodePtr& addNode, vector<ge::NodePtr>& preNodes,
                                                         const uint32_t& preNodeOutputIdx) {
  OP_LOGD(fusedOpType.c_str(), "Enter into create Add node");
  FUSION_PASS_CHECK(preNodes.size() != 3 || preNodes[0] == nullptr || preNodes[1] == nullptr || preNodes[2] == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to check preNodes"),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedOpDesc = preNodes[0]->GetOpDesc();
  FUSION_PASS_CHECK(fusedOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to get OpDesc of fused node"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(preNodeOutputIdx >= fusedOpDesc->GetAllOutputsDescSize(),
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to check preNodeOutputIdx"),
                    return PARAM_INVALID);

  std::string addNodeName = preNodes[0]->GetName() + "/Add/" + fusedOpDesc->GetOutputNameByIndex(preNodeOutputIdx);
  OP_LOGD(fusedOpType.c_str(), "Name for new Add node is: %s", addNodeName);

  ge::OpDescPtr opDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(opDesc = std::make_shared<ge::OpDesc>(addNodeName, "Add"), return INTERNAL_ERROR);
  FUSION_PASS_CHECK(opDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to create Add op desc"), return FAILED);

  ge::GeTensorDesc outputDesc = fusedOpDesc->GetOutputDesc(preNodeOutputIdx);
  FUSION_PASS_CHECK(opDesc->AddInputDesc(0, outputDesc) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add x1 desc for add node"),
                    return FAILED);
  FUSION_PASS_CHECK(opDesc->AddInputDesc(1, outputDesc) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add x2 desc for add node"),
                    return FAILED);
  FUSION_PASS_CHECK(opDesc->AddOutputDesc(outputDesc) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add y desc for add node"),
                    return FAILED);

  addNode = graph.AddNode(opDesc);
  FUSION_PASS_CHECK(addNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add Add node to graph"),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preNodes[1]->GetOutDataAnchor(preNodeOutputIdx),
                                            addNode->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add edge from preNode output to x1"),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preNodes[SPLIT_COUNT_VALUE]->GetOutDataAnchor(preNodeOutputIdx),
                                            addNode->GetInDataAnchor(1)) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add edge from preNode output to x2"),
                    return FAILED);

  OP_LOGD(fusedOpType.c_str(), "End to create Add node");
  return SUCCESS;
}

Status DeepMdFusionPassUtil::CreateConcatNodeAfterSplitNode(const std::string& fusedOpType, ge::ComputeGraph& graph,
                                                            ge::NodePtr& concatNode,
                                                            const vector<ge::NodePtr>& preNodes,
                                                            const uint32_t& preNodeOutputIdx,
                                                            const vector<int32_t>& concatAttrs) {
  OP_LOGD(fusedOpType.c_str(), "Enter into create Concat node");

  FUSION_PASS_CHECK(preNodes.size() != 3 || preNodes[0] == nullptr || preNodes[1] == nullptr || preNodes[2] == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to check preNodes"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(preNodes[0]->GetOpDesc() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to get OpDesc of fused node"),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedOpDesc = preNodes[0]->GetOpDesc();
  FUSION_PASS_CHECK(preNodeOutputIdx >= fusedOpDesc->GetAllOutputsDescSize(),
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to check preNodeOutputIdx"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(concatAttrs.size() < 2 || concatAttrs[0] < 0 || concatAttrs[1] < 1,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to check ConcatD attrs"),
                    return PARAM_INVALID);

  std::string concatNodeName = preNodes[0]->GetName() + "/Concat/"+ fusedOpDesc->GetOutputNameByIndex(preNodeOutputIdx);
  OP_LOGD(fusedOpType.c_str(), "Name for new ConcatD node is: %s", concatNodeName);

  ge::OpDescPtr concatDesc = nullptr;
  FUSION_PASS_MAKE_SHARED(concatDesc = std::make_shared<ge::OpDesc>(concatNodeName, "ConcatD"), return INTERNAL_ERROR);
  FUSION_PASS_CHECK(concatDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to create Concat op desc."),
                    return FAILED);

  ge::GeTensorDesc outputDesc = fusedOpDesc->GetOutputDesc(preNodeOutputIdx);
  ge::GeTensorDesc outputDescAic = preNodes[1]->GetOpDesc()->GetOutputDesc(preNodeOutputIdx);
  ge::GeTensorDesc outputDescVec = preNodes[SPLIT_COUNT_VALUE]->GetOpDesc()->GetOutputDesc(preNodeOutputIdx);
  FUSION_PASS_CHECK(concatDesc->AddInputDesc("x1", outputDescAic) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add x1 desc for Concat."),
                    return FAILED);
  FUSION_PASS_CHECK(concatDesc->AddInputDesc("x2", outputDescVec) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add x2 desc for Concat."),
                    return FAILED);
  FUSION_PASS_CHECK(concatDesc->AddOutputDesc(outputDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add y desc for Concat."),
                    return FAILED);

  ge::AttrUtils::SetInt(concatDesc, "concat_dim", concatAttrs[0]);
  ge::AttrUtils::SetInt(concatDesc, "N", concatAttrs[1]);

  concatNode = graph.AddNode(concatDesc);
  FUSION_PASS_CHECK(concatNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add ConcatD Node to graph"),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(preNodes[1]->GetOutDataAnchor(preNodeOutputIdx),
                                                       concatNode->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add edge from output to concat x1."),
                    return FAILED);
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(preNodes[SPLIT_COUNT_VALUE]->GetOutDataAnchor(preNodeOutputIdx),
                                                       concatNode->GetInDataAnchor(1)),
                    VECTOR_FUSION_INNER_ERR_REPORT(fusedOpType.c_str(), "Failed to add edge from output to concat x2."),
                    return FAILED);

  OP_LOGD(fusedOpType.c_str(), "End to create Concat node");
  return SUCCESS;
}

Status DeepMdFusionPassUtil::ClearFusedNode(const std::string& fusedOpType, ge::ComputeGraph& graph,
                                            ge::NodePtr& node) {
  OP_LOGD(fusedOpType.c_str(), "Enter into clear fused node");

  std::string nodeName = node->GetName();
  OP_LOGD(fusedOpType.c_str(), "Node name: %s", nodeName.c_str());
  for (auto inAnchor : node->GetAllInDataAnchors()) {
    if (inAnchor == nullptr) {
      OP_LOGE(fusedOpType.c_str(), "Failed to UnlinkAll of %s for inAnchor is nullptr.", nodeName.c_str());
    } else {
      inAnchor->UnlinkAll();
      OP_LOGD(fusedOpType.c_str(), "Finished to UnlinkAll inDataAnchor of node[%s].", nodeName.c_str());
    }
  }

  FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fusedOpType.c_str(), "Failed to remove node[%s].", nodeName.c_str()), return FAILED);

  OP_LOGD(fusedOpType.c_str(), "End to clear fused node");
  return SUCCESS;
}
}  // namespace fe
