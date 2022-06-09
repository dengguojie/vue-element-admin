/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file matmulv2_fusion_pass.cpp
 * \brief MatMulV2 fusion pass
 */
#include "matmulv2_fusion_pass.h"

#include <memory>
#include <string>

#include "anchor_util.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_INPUTS1 = "input1";
static const string PATTERN_INPUTS2 = "input2";
static const string PATTERN_INPUTS3 = "input3";
static const string PATTERN_MATMULV2 = "matmulv2";
static const string MATMULV2 = "MatMulV2";
static const string TRANSPOSEB = "transpose_b";
static const string TRANSPOSEBX2 = "transpose_x2";
static const string TRANSPOSED_TYPE = "TransposeD";
static const string PERM = "perm";
static const int CONST_INDEX = 1;
static const int CONST_DIM_NUM = 2;
vector<FusionPattern*> MatMulV2FusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MatMulV2FusionPass");

  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "new MatMulV2FusionPass pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_MATMULV2, {MATMULV2})
      .AddOpDesc(PATTERN_INPUTS1)
      .AddOpDesc(PATTERN_INPUTS2)
      .AddOpDesc(PATTERN_INPUTS3)
      .SetInputs(PATTERN_MATMULV2, {PATTERN_INPUTS1, PATTERN_INPUTS2, PATTERN_INPUTS3})
      .SetOutput(PATTERN_MATMULV2);
  patterns.push_back(pattern);
  return patterns;
}

Status MatMulV2FusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr matMulV2Node = GetNodeFromMapping(PATTERN_MATMULV2, mapping);
  FUSION_PASS_CHECK(matMulV2Node == nullptr,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "matMulV2Node Node is null, fusion failed."),
                    return NOT_CHANGED);

  bool transposeB = false;
  ge::OpDescPtr matMulV2NodeDescPtr = matMulV2Node->GetOpDesc();
  FUSION_PASS_CHECK(matMulV2NodeDescPtr == nullptr,
                    OP_LOGW(matMulV2Node, "matMulV2NodeDescPtr is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(matMulV2Node->GetOpDesc(), TRANSPOSEB, transposeB),
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get transpose_b attr failed."), return NOT_CHANGED);

  if (!transposeB) {
    return NOT_CHANGED;
  }
  size_t constInputDimNum = matMulV2Node->GetOpDesc()->GetInputDesc(CONST_INDEX).GetShape().GetDimNum();
  if (constInputDimNum != CONST_DIM_NUM) {
    return NOT_CHANGED;
  }

  ge::DataType constDataType = matMulV2Node->GetOpDesc()->GetInputDesc(CONST_INDEX).GetDataType();
  if (constDataType != ge::DT_INT8) {
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(matMulV2Node->GetOpDesc(), TRANSPOSEB, false),
                    OP_LOGW(matMulV2Node, "Set transpose_b attr failed."), return NOT_CHANGED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(matMulV2Node->GetOpDesc(), TRANSPOSEBX2, false),
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "Set transpose_x2 attr failed."), return FAILED);

  std::vector<int64_t> oldDims = matMulV2Node->GetOpDesc()->GetInputDesc(CONST_INDEX).GetShape().GetDims();
  FUSION_PASS_CHECK(oldDims.size() < 2,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "matmul shape length is not 2."),
                    return FAILED);
  vector<int64_t> newDims;
  newDims.push_back(oldDims[1]);
  newDims.push_back(oldDims[0]);
  ge::GeShape newShape(newDims);
  matMulV2Node->GetOpDesc()->MutableInputDesc(CONST_INDEX)->SetShape(newShape);
  matMulV2Node->GetOpDesc()->MutableInputDesc(CONST_INDEX)->SetOriginShape(newShape);

  ge::NodePtr constNode = GetPeerOutNodeWithInDataAnchor(matMulV2Node, CONST_INDEX);
  FUSION_PASS_CHECK(constNode == nullptr,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "failed to get const node"),
                    return FAILED);

  std::shared_ptr<ge::OpDesc> transposeOpdesc = nullptr;
  FUSION_PASS_MAKE_SHARED(
      (transposeOpdesc = std::make_shared<ge::OpDesc>(constNode->GetName() + "_transpose_b", TRANSPOSED_TYPE)),
      return FAILED);
  FUSION_PASS_CHECK(transposeOpdesc == nullptr,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "failed to create transpose node"),
                    return FAILED);

  vector<int64_t> perm;
  perm.push_back(1);
  perm.push_back(0);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(transposeOpdesc, PERM, perm),
                    CUBE_INNER_ERR_REPORT(matMulV2Node,
                    "Set perm to %s failed.", transposeOpdesc->GetName().c_str()), return FAILED);

  ge::GeTensorDesc inputDesc = constNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc outputDesc = matMulV2Node->GetOpDesc()->GetInputDesc(CONST_INDEX);

  FUSION_PASS_CHECK(transposeOpdesc->AddInputDesc("x", inputDesc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "%s add inputDesc failed.",
                    transposeOpdesc->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(transposeOpdesc->AddOutputDesc("y", outputDesc) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "%s add outputDesc failed.",
                    transposeOpdesc->GetName().c_str()), return FAILED);

  ge::NodePtr transposeNode = graph.AddNode(transposeOpdesc);
  FUSION_PASS_CHECK(transposeNode == nullptr,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "failed to add transpose to graph"), return FAILED);

  ge::OutDataAnchorPtr src = constNode->GetOutDataAnchor(0);
  ge::InDataAnchorPtr dst = matMulV2Node->GetInDataAnchor(CONST_INDEX);

  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "remove %s input0 edge error",
                    matMulV2Node->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(src, transposeNode->GetInDataAnchor(0)) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "Add edge between node %s. and node %s failed.",
                            constNode->GetName().c_str(), transposeNode->GetName().c_str()),
                    return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), dst) != SUCCESS,
                    CUBE_INNER_ERR_REPORT(matMulV2Node, "Add edge between node %s. and node %s failed.",
                            transposeNode->GetName().c_str(), matMulV2Node->GetName().c_str()),
                    return FAILED);

  fusionNodes.push_back(transposeNode);
  OP_LOGI(matMulV2Node, "MatMulV2FusionPass fusion success!");
  return SUCCESS;
}
REGISTER_PASS("MatMulV2FusionPass", BUILT_IN_GRAPH_PASS, MatMulV2FusionPass);
}  // namespace fe
