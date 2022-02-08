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
 * \file tabulatefusiongrad_transpose_fusion_pass.cc
 * \brief TabulateFusionGrad transpose fusion pass, reorder and fills input0 (table)
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
#include "tbe_ops_pass_util.h"
#include "tabulatefusiongrad_transpose_fusion_pass.h"

namespace fe {
static const std::string PATTERN_TABULATEFUSIONGRAD = "TabulateFusionGrad";
static const std::string OP_TYPE_TABULATEFUSIONGRAD = "TabulateFusionGrad";
static const int NUM_5 = 5;
static const int NUM_6 = 6;
static const int NUM_2 = 2;

vector<FusionPattern *> ATabulateFusionGradTransposeFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  string passName = "ATabulateFusionGradTransposeFusionPass";
  FusionPattern *pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define pass pattern");
  pattern->AddOpDesc(PATTERN_TABULATEFUSIONGRAD, {OP_TYPE_TABULATEFUSIONGRAD})
      .SetOutput(PATTERN_TABULATEFUSIONGRAD);
  patterns.push_back(pattern);
  return patterns;
}

/*!
 * @brief: parse nodes matched in mapping and call graph DoFusion
 * @param [in] graph: original graph
 * @param [in] mapping: matched pattern
 * @param [out] newNodes: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status ATabulateFusionGradTransposeFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                                      vector<ge::NodePtr> &newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ATabulateFusionGradTranspose fusion pass.");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_TABULATEFUSIONGRAD, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get TabulateFusionGrad Node"),
                    return PARAM_INVALID);

  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorDesc tableTensorDesc = fusedDesc->GetInputDesc(0);
  ge::GeShape tableShape = tableTensorDesc.GetShape();
  FUSION_PASS_CHECK(IsUnknownShape(tableShape.GetDims()) || (tableShape.GetDims().size() != NUM_2),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Not support, table is dynamic shape"),
                    return PARAM_INVALID);
  int64_t tableDim0 = tableShape.GetDim(0);
  int64_t tableDim1 = tableShape.GetDim(1);

  ge::GeTensorDesc descriptorTensorDesc = fusedDesc->GetInputDesc(NUM_5);
  ge::GeShape descriptorShape = descriptorTensorDesc.GetShape();
  int32_t lastLayerSize = descriptorShape.GetDim(NUM_2);
  
  Operator fusedOp = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  FUSION_PASS_CHECK((tableDim1 != lastLayerSize * NUM_6),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Failed to check input shape. tableDim[1] vs descriptorDim[2]*6 (%ld vs %d)",
                    tableDim1, lastLayerSize * NUM_6),
                    return PARAM_INVALID);

  // input0: table, is a const tensor
  ge::InDataAnchorPtr dataAnchorPtr = fusedNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr tableAnchorPtr = dataAnchorPtr->GetPeerOutAnchor();
  ge::NodePtr tableNode = tableAnchorPtr->GetOwnerNode();

  ge::OpDescPtr tableDesc = tableNode->GetOpDesc();
  FUSION_PASS_CHECK(tableDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "tableDesc is null, fusion failed."),
                    return FAILED);
  ge::OpDescPtr newTableDesc = AttrUtils::CopyOpDesc(tableDesc);
  FUSION_PASS_CHECK(newTableDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to create new table Desc"),
                    return FAILED);
  newTableDesc->SetName(newTableDesc->GetName() + "_grad_new");
  ge::NodePtr newTableNode = graph.AddNode(newTableDesc);
  FUSION_PASS_CHECK(newTableNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add new table to graph"),
                    return FAILED);

  fusedNode->GetInDataAnchor(0)->UnlinkAll();
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newTableNode->GetOutDataAnchor(0), fusedNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from new table to fused node."),
      return FAILED);

  vector<ge::GeTensorPtr> tableWeights = ge::OpDescUtils::MutableWeights(newTableNode);
  FUSION_PASS_CHECK(tableWeights.empty(),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Weights of table is empty, fusion failed."),
                    return PARAM_INVALID);
  ge::GeTensorPtr tableTensorPtr = tableWeights[0];
  FUSION_PASS_CHECK(tableTensorPtr == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "table tensor is NULL"),
                    return PARAM_INVALID);

  tableTensorPtr->SetTensorDesc(tableTensorDesc);
  newTableNode->GetOpDesc()->UpdateOutputDesc(0, tableTensorDesc);

  int64_t tableDataSize = tableDim0 * tableDim1;
  std::unique_ptr<float[]> newTableData(new (std::nothrow) float[tableDataSize]());
  FUSION_PASS_CHECK(newTableData.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newTableData is NULL"),
                    return FAILED);
  auto retMem = memset_s(newTableData.get(), tableDataSize, 0, tableDataSize);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "memset_s function failed!"),
                    return FAILED);
  
  float* tableNewData = newTableData.get();
  float* tableData = (float*)(tableTensorPtr->GetData().data());
  FUSION_PASS_CHECK(tableData == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "table Data is NULL"),
                    return PARAM_INVALID);

  for (int64_t i = 0; i < tableDim0; i++) {
    for (int64_t j = 0; j < tableDim1; j++) {
      *(tableNewData + i * tableDim1 + ((j % NUM_6) * lastLayerSize + (j / NUM_6))) =
          *(tableData + i * tableDim1 + j);
    }
  }

  ge::GeTensorPtr weightTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
    weightTensor = std::make_shared<GeTensor>(tableTensorDesc, reinterpret_cast<uint8_t*>(newTableData.get()),
                                              tableDataSize * sizeof(float)),
    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetTensor(newTableNode->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, weightTensor),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Failed to set ATTR_NAME_WEIGHTS, const: table"),
                    return FAILED);

  if (tableNode->GetOutAllNodes().size() == 0) {
    FUSION_PASS_CHECK(graph.RemoveNode(tableNode) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to remove origin table node."),
                      return FAILED);
  }
  newNodes.push_back(newTableNode);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ATabulateFusionGradTranspose fusion pass.");
  return SUCCESS;
}

REGISTER_PASS("ATabulateFusionGradTransposeFusionPass", BUILT_IN_GRAPH_PASS, ATabulateFusionGradTransposeFusionPass);
} // namespace fe
