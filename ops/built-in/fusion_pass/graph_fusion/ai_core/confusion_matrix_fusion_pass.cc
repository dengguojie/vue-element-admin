/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file confusion_matrix_fusion_pass.cpp
 * \brief confusion_matrix fusion pass( --> confusion_matrix)
 */
#include "confusion_matrix_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const char* CAST = "Cast";
static const char* SPARSETENSORDENSEADD = "SparseTensorDenseAdd";
static const char* TRANSPOSE = "Transpose";
static const char* PACK = "Pack";
static const string PATTERN_CONFUSION_MATRIX = "ConfusionMatrix";
static const string PATTERN_CAST = "Cast";
static const string PATTERN_CAST_1 = "Cast_1";
static const string PATTERN_SPARSE_TENSOR_DENSE_ADD = "SparseTensorDenseAdd";
static const string PATTERN_TRANSPOSE = "Transpose";
static const string PATTERN_PACK = "Pack";
/*

          SparseTensorDenseAdd
            /     |
           /  transpose
          /       |              --->  confusion_matrix
   input edge   pack
                /   \
               /     \
           cast_1    cast
*/
vector<FusionPattern*> ConfusionMatrixFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // tf confusion_matrix api fused to tbe confusion_matrix
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConfusionMatrixFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_CAST, {CAST})
      .AddOpDesc(PATTERN_CAST_1, {CAST})
      .AddOpDesc(PATTERN_SPARSE_TENSOR_DENSE_ADD, {SPARSETENSORDENSEADD})
      .AddOpDesc(PATTERN_TRANSPOSE, {TRANSPOSE})
      .AddOpDesc(PATTERN_PACK, {PACK})
      .SetInputs(PATTERN_PACK, {PATTERN_CAST_1, PATTERN_CAST})
      .SetInputs(PATTERN_TRANSPOSE, {PATTERN_PACK})
      .SetInputs(PATTERN_SPARSE_TENSOR_DENSE_ADD, {PATTERN_TRANSPOSE})
      .SetOutput(PATTERN_SPARSE_TENSOR_DENSE_ADD);

  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status ConfusionMatrixFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // get orig graph confusion_matrix node and desc of confuion_matrix node
  ge::NodePtr castNode = GetNodeFromMapping(PATTERN_CAST, mapping);
  ge::NodePtr cast1Node = GetNodeFromMapping(PATTERN_CAST_1, mapping);
  ge::NodePtr sparseNode = GetNodeFromMapping(PATTERN_SPARSE_TENSOR_DENSE_ADD, mapping);
  ge::NodePtr transposeNode = GetNodeFromMapping(PATTERN_TRANSPOSE, mapping);
  ge::NodePtr packNode = GetNodeFromMapping(PATTERN_PACK, mapping);
  FUSION_PASS_CHECK(castNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "castNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(sparseNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "sparseNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(cast1Node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "cast1Node is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(transposeNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "transposeNode is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(packNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "packNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::GeTensorDesc inputDesc1 = cast1Node->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc inputDesc2 = castNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc inputDesc3 = sparseNode->GetOpDesc()->GetInputDesc(1);
  ge::GeTensorDesc outputDesc = sparseNode->GetOpDesc()->GetOutputDesc(0);

  ge::OpDescPtr confusionMatrixOp;
  FUSION_PASS_MAKE_SHARED((confusionMatrixOp = std::make_shared<ge::OpDesc>("confusion_matrix", "ConfusionMatrix")),
                          return INTERNAL_ERROR);
  confusionMatrixOp->AddInputDesc("x", inputDesc1);
  confusionMatrixOp->AddInputDesc("y", inputDesc2);
  confusionMatrixOp->AddInputDesc("z", inputDesc3);
  confusionMatrixOp->AddOutputDesc("output", outputDesc);
  ge::NodePtr confusionMatrixNode = graph.AddNode(confusionMatrixOp);
  FUSION_PASS_CHECK(confusionMatrixNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "confusionMatrixNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OutDataAnchorPtr newInAnchorPtr0 = cast1Node->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::OutDataAnchorPtr newInAnchorPtr1 = castNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  ge::OutDataAnchorPtr newInAnchorPtr2 = sparseNode->GetInDataAnchor(1)->GetPeerOutAnchor();
  ge::GraphUtils::AddEdge(newInAnchorPtr0, confusionMatrixNode->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(newInAnchorPtr1, confusionMatrixNode->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(newInAnchorPtr2, confusionMatrixNode->GetInDataAnchor(2));

  for (auto inDataAnchor : sparseNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(sparseNode->GetOutDataAnchor(0), inDataAnchor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(confusionMatrixNode->GetOutDataAnchor(0), inDataAnchor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }

  ge::OpDescPtr confusionMatrixDesc = confusionMatrixNode->GetOpDesc();
  ge::InDataAnchorPtr tointAnchorPtr = sparseNode->GetInDataAnchor(2);
  ge::OutDataAnchorPtr constAnchorPtr = tointAnchorPtr->GetPeerOutAnchor();
  ge::NodePtr constNode = constAnchorPtr->GetOwnerNode();
  ge::ConstGeTensorPtr constTensor = nullptr;
  ge::AttrUtils::GetTensor(constNode->GetOpDesc(), "value", constTensor);
  const uint8_t* constData = constTensor->GetData().GetData();
  ge::AttrUtils::SetInt(confusionMatrixDesc, "num_classes", *(int64_t*)constData);

  int64_t outtype;
  std::string outtype_str = "float32";
  if (confusionMatrixOp->GetInputDesc(0).GetDataType() != confusionMatrixOp->GetInputDesc(0).GetDataType()) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "labels and predictions dtype should be same.");
    return NOT_CHANGED;
  }
  outtype = confusionMatrixOp->GetInputDesc(2).GetDataType();
  if (outtype == 0) {
    outtype_str = "float32";
  } else if (outtype == 1) {
    outtype_str = "float16";
  } else if (outtype == 2) {
    outtype_str = "int8";
  } else if (outtype == 4) {
    outtype_str = "uint8";
  } else if (outtype == 3) {
    outtype_str = "int32";
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Output_dtype can not support this dtype.");
    return NOT_CHANGED;
  }
  ge::AttrUtils::SetStr(confusionMatrixDesc, "dtype", outtype_str);

  graph.RemoveNode(castNode);
  graph.RemoveNode(cast1Node);
  graph.RemoveNode(sparseNode);
  graph.RemoveNode(transposeNode);
  graph.RemoveNode(packNode);

  fusionNodes.push_back(confusionMatrixNode);

  return SUCCESS;
}

REGISTER_PASS("ConfusionMatrixFusionPass", BUILT_IN_GRAPH_PASS, ConfusionMatrixFusionPass);
}  // namespace fe
