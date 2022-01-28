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
 * \file tabulatefusion_transpose_fusion_pass.cc
 * \brief TabulateFusion transpose fusion pass, reorder and fills input0 (table)
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
#include "tabulatefusion_transpose_fusion_pass.h"

namespace fe {
static const std::string PATTERN_TABULATEFUSION = "TabulateFusion";
static const std::string OP_TYPE_TABULATEFUSION = "TabulateFusion";
static const std::string ATTR_LAST_LAYER_SIZE = "last_layer_size";
static const int NUM_64 = 64;
static const int NUM_6 = 6;
static const int NUM_2 = 2;

vector<FusionPattern *> ATabulateFusionTransposeFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  string passName = "ATabulateFusionTransposeFusionPass";
  FusionPattern *pattern = new (std::nothrow) FusionPattern(passName);
  FUSION_PASS_CHECK(pattern == nullptr, CommonRuntimeErrLog(FUSED_OP_TYPE, "Failed to new a pattern object."),
                    return patterns);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define pass pattern");
  pattern->AddOpDesc(PATTERN_TABULATEFUSION, {OP_TYPE_TABULATEFUSION})
      .SetOutput(PATTERN_TABULATEFUSION);
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
Status ATabulateFusionTransposeFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                                  vector<ge::NodePtr> &newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter into ATabulateFusionTranspose fusion pass.");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_TABULATEFUSION, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get TabulateFusion Node"),
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
  int32_t lastLayerSize;
  Operator fusedOp = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  FUSION_PASS_CHECK((fusedOp.GetAttr(ATTR_LAST_LAYER_SIZE.c_str(), lastLayerSize) != ge::GRAPH_SUCCESS) ||
                    (tableDim1 != lastLayerSize * NUM_6),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to get attr last_layer_size"),
                    return PARAM_INVALID);

  int64_t lastLayerSizeAlign = lastLayerSize;
  if (lastLayerSize % NUM_64 != 0) {
    lastLayerSizeAlign = (lastLayerSize + NUM_64 - 1) / NUM_64 * NUM_64;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "lastLayerSize is: %d, lastLayerSizeAlign is: %d", lastLayerSize, lastLayerSizeAlign);

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
  newTableDesc->SetName(newTableDesc->GetName() + "_new");
  ge::NodePtr newTableNode = graph.AddNode(newTableDesc);
  FUSION_PASS_CHECK(newTableNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add new table to graph"),
                    return FAILED);

  fusedNode->GetInDataAnchor(0)->UnlinkAll();
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(newTableNode->GetOutDataAnchor(0), fusedNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Failed to add edge from new table to fused node."),
      return FAILED);

  int64_t newTableDim1 = lastLayerSizeAlign * NUM_6;
  ge::GeShape newTableShape({tableDim0, newTableDim1});
  tableTensorDesc.SetOriginShape(newTableShape);
  tableTensorDesc.SetShape(newTableShape);
  fusedDesc->UpdateInputDesc(0, tableTensorDesc);

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

  int64_t newTableDataSize = tableDim0 * newTableDim1;
  std::unique_ptr<float[]> newTableData(new (std::nothrow) float[newTableDataSize]());
  FUSION_PASS_CHECK(newTableData.get() == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "newTableData is NULL"),
                    return FAILED);
  auto retMem = memset_s(newTableData.get(), newTableDataSize, 0, newTableDataSize);
  FUSION_PASS_CHECK(retMem != EOK, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "memset_s function failed!"),
                    return FAILED);
  float* tableNewData = newTableData.get();
  float* tableData = (float*)(tableTensorPtr->GetData().data());
  FUSION_PASS_CHECK(tableData == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "table Data is NULL"),
                    return PARAM_INVALID);

  for (int64_t i = 0; i < tableDim0; i++) {
    for (int64_t j = 0; j < tableDim1; j++) {
      *(tableNewData + i * newTableDim1 + ((j % NUM_6) * lastLayerSizeAlign + (j / NUM_6))) =
          *(tableData + i * tableDim1 + j);
    }
  }

  ge::GeTensorPtr weightTensor = nullptr;
  FUSION_PASS_MAKE_SHARED(
    weightTensor = std::make_shared<GeTensor>(tableTensorDesc, reinterpret_cast<uint8_t*>(newTableData.get()),
                                              newTableDataSize * sizeof(float)),
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

  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to ATabulateFusionTranspose fusion pass.");
  return SUCCESS;
}

REGISTER_PASS("ATabulateFusionTransposeFusionPass", BUILT_IN_GRAPH_PASS, ATabulateFusionTransposeFusionPass);
} // namespace fe
