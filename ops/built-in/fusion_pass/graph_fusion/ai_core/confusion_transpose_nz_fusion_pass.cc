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
 * \file confusion_transpose_nz_fusion_pass.cc
 * \brief
 */
#include "confusion_transpose_nz_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_CONFUSIONTRANSPOSE = "FusedConfusionTransposeD";
static const string FUSED_NODE = "ConfusionTransposeD";
static const int64_t PERM_VALUE_2 = 2;
static const int64_t PERM_VALUE_3 = 3;
static const size_t DIM_SIZE_3 = 3;
static const size_t DIM_SIZE_4 = 4;
static const size_t PERM_SIZE = 4;
static const size_t DIM_IDX_2 = 2;
static const size_t DIM_IDX_3 = 3;
/*
confusion_transpose_d nz format if final_perm = [0,1,2,3,4,5] can delete
*/

Status ConfusionTransposeNzFusionPass::RemoveNode(ge::NodePtr node, ge::ComputeGraph& graph) {
  // remove input data edge
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); ++i) {
    auto inDataAnchor = node->GetInDataAnchor(i);
    FUSION_PASS_CHECK(inDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                                     "inDataAnchor is null, remove node failed."),
                      return FAILED);
    auto preOutDataAnchor = inDataAnchor->GetPeerOutAnchor();
    FUSION_PASS_CHECK(preOutDataAnchor == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                                     "preOutDataAnchor is null, remove node failed."),
                      return FAILED);

    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(preOutDataAnchor, inDataAnchor) != ge::GRAPH_SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "remove node failed."),
                      return FAILED);
    OP_LOGI(FUSED_OP_TYPE, "remove edge %zu of node %s", i, node->GetName().c_str());
  }
  // delete the node
  FUSION_PASS_CHECK(graph.RemoveNode(node) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "remove node failed"),
                    return FAILED);
  return SUCCESS;
}

vector<FusionPattern*> ConfusionTransposeNzFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE, "Define ConfusionTransposeNzFusionPass pattern begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConfusionTransposeNzFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new an object failed"),
                    return patterns);

  pattern->AddOpDesc(PATTERN_CONFUSIONTRANSPOSE, {FUSED_NODE}).SetOutput(PATTERN_CONFUSIONTRANSPOSE);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE, "Define ConfusionTransposeNzFusionPass pattern end");

  return patterns;
}

Status ConfusionTransposeNzFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                              vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE, "Define ConfusionTransposeNzFusionPass fusion begin");
  ge::NodePtr confusionTransposeD = GetNodeFromMapping(PATTERN_CONFUSIONTRANSPOSE, mapping);

  FUSION_PASS_CHECK(confusionTransposeD == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "confusionTransposeD is null"),
                    return PARAM_INVALID);
  // must be NZ
  ge::OpDescPtr ConfusionTransposeOpDesc = confusionTransposeD->GetOpDesc();
  FUSION_PASS_CHECK(ConfusionTransposeOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "opdesc is null"), return PARAM_INVALID);
  ge::GeTensorDesc ConfusionTransposeInputTensor = ConfusionTransposeOpDesc->GetInputDesc(0);
  ge::GeTensorDesc ConfusionTransposeOutputTensor = ConfusionTransposeOpDesc->GetOutputDesc(0);
  if (ConfusionTransposeInputTensor.GetFormat() != ge::FORMAT_FRACTAL_NZ ||
      ConfusionTransposeOutputTensor.GetFormat() != ge::FORMAT_FRACTAL_NZ) {
    OP_LOGI(FUSED_OP_TYPE,"input format is not FRACTAL_NZ , not support fusion,"
            "ConfusionTransposeNzFusionPass fusion end");
    return NOT_CHANGED;
  }
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(confusionTransposeD);
  std::vector<int64_t> permList;
  if (op.GetAttr("perm", permList) != ge::GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op), "GetOpAttr perm failed!");
    return NOT_CHANGED;
  }
  // perm must be [0, 2, 1, 3]
  if (permList.size() != PERM_SIZE || permList[0] != 0 || permList[1] != PERM_VALUE_2 ||
      permList[DIM_IDX_2] != 1 || permList[DIM_IDX_3] != PERM_VALUE_3) {
    OP_LOGI(TbeGetName(op), "length of perm not equal to 4!");
    return NOT_CHANGED;
  }
  ge::GeShape Shape_in = ConfusionTransposeInputTensor.GetOriginShape();
  ge::GeShape Shape_out = ConfusionTransposeOutputTensor.GetOriginShape();
  size_t Shape_in_size = Shape_in.GetDimNum();
  size_t Shape_out_size = Shape_out.GetDimNum();
  if (!((Shape_in_size == DIM_SIZE_3 && Shape_out_size == DIM_SIZE_4) ||
        (Shape_in_size == DIM_SIZE_4 && Shape_out_size == DIM_SIZE_3))) {
    OP_LOGI(TbeGetName(op), "the dims of in shape size %zu out shape size %zu not match the fusion condition!",
            Shape_in_size, Shape_out_size);
    
    return NOT_CHANGED;
  }
  if (Shape_in_size == DIM_SIZE_3) {
    if (!(Shape_in.GetDim(0) == Shape_out.GetDim(0) &&
        Shape_in.GetDim(1) == Shape_out.GetDim(DIM_IDX_2) &&
        Shape_in.GetDim(DIM_IDX_2) == Shape_out.GetDim(1) * Shape_out.GetDim(DIM_IDX_3))) {
      OP_LOGI(TbeGetName(op), "Shape_in == 3 not match the fusion condition!");
      return NOT_CHANGED;
    }
  }
  if (Shape_in_size == DIM_SIZE_4) {
    if (!(Shape_out.GetDim(0) == Shape_in.GetDim(0) &&
        Shape_out.GetDim(1) == Shape_in.GetDim(DIM_IDX_2) &&
        Shape_out.GetDim(DIM_IDX_2) == Shape_in.GetDim(1) * Shape_in.GetDim(DIM_IDX_3))) {
      OP_LOGI(TbeGetName(op), "Shape_in == 4 not match the fusion condition!");
      return NOT_CHANGED;
    }
  }
  // add edge from the input to the output
  size_t output_link_nums = confusionTransposeD->GetOutDataAnchor(0)->GetPeerInDataAnchors().size();
  OP_LOGI(FUSED_OP_TYPE, "ConfusionTranposeD output connect [%d] node.", output_link_nums);
  if (confusionTransposeD->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() == 0) {
    OP_LOGI(FUSED_OP_TYPE, "output connect 0 node, not changed");
    return NOT_CHANGED;
  }
  for (InDataAnchorPtr inAnchorPtr : confusionTransposeD->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    inAnchorPtr->UnlinkAll();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(confusionTransposeD->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                          inAnchorPtr),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Add edge failed"), return FAILED);
  }
  // delete confusionTransposeD node
  FUSION_PASS_CHECK(RemoveNode(confusionTransposeD, graph) == FAILED,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "remove confusionTransposeD node failed"),
                    return FAILED);
  OP_LOGI(FUSED_OP_TYPE, "Define ConfusionTransposeNzFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("ConfusionTransposeNzFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              ConfusionTransposeNzFusionPass);
}  // namespace fe
