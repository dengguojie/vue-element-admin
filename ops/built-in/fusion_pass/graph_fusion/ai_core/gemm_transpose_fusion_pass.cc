/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file gemm_transpose_fusion_pass.cpp
 * \brief gemm transpose fusion pass
 */
#include "gemm_transpose_fusion_pass.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
namespace {
static const char GEMM[] = "GEMM";
static const char PATTERN_GEMM[] = "GEMM";
static const int ALIGN_LENGTH = 16;
}  // namespace

vector<FusionPattern*> GemmTransFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GemmTransFusionPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      new (std::nothrow) FusionPattern("GemmTransFusionPass");
  FUSION_PASS_CHECK(
      pattern == nullptr,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
      return patterns);

  pattern->AddOpDesc(PATTERN_GEMM, {GEMM}).SetOutput(PATTERN_GEMM);

  patterns.push_back(pattern);

  return patterns;
}

static Status GenerateTransposeNode(ge::ComputeGraph* graph,
                                    const ge::GeTensorDesc& prevOutDesc,
                                    ge::GeTensorDesc* nextInDesc,
                                    const vector<int64_t>& perm,
                                    ge::NodePtr* transposeNode,
                                    const std::string& basename) {
  vector<int64_t> nextInShape(2);
  for (size_t i = 0; i < perm.size(); ++i) {
    nextInShape[i] = prevOutDesc.GetShape().GetDim(perm[i]);
  }
  ge::OpDescPtr transposeDesc;
  FUSION_PASS_MAKE_SHARED((transposeDesc = std::make_shared<ge::OpDesc>(
                               basename + "_transpose", "TransposeD")),
                          return FAILED);
  transposeDesc->AddInputDesc("x", prevOutDesc);
  nextInDesc->SetShape(ge::GeShape(nextInShape));
  nextInDesc->SetOriginShape(ge::GeShape(nextInShape));
  transposeDesc->AddOutputDesc("y", *nextInDesc);
  ge::AttrUtils::SetListInt(transposeDesc, "perm", perm);
  *transposeNode = graph->AddNode(transposeDesc);
  return SUCCESS;
}

Status GemmTransFusionPass::Relink(ge::NodePtr aNode,
                                   ge::NodePtr transposeANode,
                                   ge::NodePtr gemmNode, const int Anchor) {
  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(aNode->GetOutDataAnchor(0),
                                 gemmNode->GetInDataAnchor(Anchor)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to remove edge between aNode and gemmNode"),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(aNode->GetOutDataAnchor(0),
                              transposeANode->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to add edge between aNode and transposeANode"),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(transposeANode->GetOutDataAnchor(0),
                              gemmNode->GetInDataAnchor(Anchor)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to add edge between transposeANode and gemmNode"),
      return FAILED);

  FUSION_PASS_CHECK(
      gemmNode->GetOpDesc()->UpdateInputDesc(
          Anchor, transposeANode->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(),
              "fail to update input description of transdataANode"),
      return FAILED);

  return SUCCESS;
}

Status GemmTransFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                   vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter GemmTransFusionPass.");
  ge::NodePtr gemmNode = GetNodeFromMapping(PATTERN_GEMM, mapping);

  int aAnchor = 0;
  int bAnchor = 1;
  int cAnchor = 2;

  // get transpose flag
  bool transposeA = false;
  bool transposeB = false;
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(gemmNode);

  if (op.GetAttr("transpose_a", transposeA) != GRAPH_SUCCESS) {
    OP_LOGI(
        FUSED_OP_TYPE.c_str(),
        "op gemm get attribute transpose_a failed or transpose_a not exist");
  }

  if (op.GetAttr("transpose_b", transposeB) != GRAPH_SUCCESS) {
    OP_LOGI(
        FUSED_OP_TYPE.c_str(),
        "op gemm get attribute transpose_b failed or transpose_b not exist");
  }

  // prerequisite
  ge::NodePtr aNode =
      gemmNode->GetInDataAnchor(aAnchor)->GetPeerOutAnchor()->GetOwnerNode();
  int aIdx = gemmNode->GetInDataAnchor(aAnchor)->GetPeerOutAnchor()->GetIdx();

  ge::NodePtr bNode =
      gemmNode->GetInDataAnchor(bAnchor)->GetPeerOutAnchor()->GetOwnerNode();
  int bIdx = gemmNode->GetInDataAnchor(bAnchor)->GetPeerOutAnchor()->GetIdx();

  ge::NodePtr cNode =
      gemmNode->GetInDataAnchor(cAnchor)->GetPeerOutAnchor()->GetOwnerNode();
  int cIdx = gemmNode->GetInDataAnchor(cAnchor)->GetPeerOutAnchor()->GetIdx();

  // get info of Node
  ge::GeTensorDesc aOutDesc = aNode->GetOpDesc()->GetOutputDesc(aIdx);
  ge::GeTensorDesc bOutDesc = bNode->GetOpDesc()->GetOutputDesc(bIdx);
  ge::GeTensorDesc cOutDesc = cNode->GetOpDesc()->GetOutputDesc(cIdx);

  ge::GeTensorDesc gemmAInDesc = gemmNode->GetOpDesc()->GetInputDesc(aAnchor);
  ge::GeTensorDesc gemmBInDesc = gemmNode->GetOpDesc()->GetInputDesc(bAnchor);

  // get format and shape of a,b,c
  ge::Format aFormat = aOutDesc.GetFormat();
  ge::Format bFormat = bOutDesc.GetFormat();
  ge::Format cFormat = cOutDesc.GetFormat();
  ge::GeShape aShape = aOutDesc.GetShape();
  ge::GeShape bShape = bOutDesc.GetShape();

  // get nDirectionLength
  int nDirectionLength = 0;
  std::vector<int64_t> bShapeVector = bShape.GetDims();

  if (transposeB) {
    nDirectionLength = bShapeVector[0];
  } else {
    nDirectionLength = bShapeVector[1];
  }

  bool needTranspose = true;

  if (aFormat == ge::FORMAT_ND && bFormat == ge::FORMAT_ND &&
      cFormat == ge::FORMAT_ND && (nDirectionLength % ALIGN_LENGTH == 0)) {
    needTranspose = false;
  }

  // 2. transpose
  ge::NodePtr transposeANode = nullptr;
  ge::NodePtr transposeBNode = nullptr;
  auto basenameA = aNode->GetName();
  auto basenameB = bNode->GetName();
  vector<int64_t> transPerm({1, 0});

  if (transposeA && needTranspose) {
    // transpose a
    FUSION_PASS_CHECK(
        GenerateTransposeNode(&graph, aOutDesc, &gemmAInDesc, transPerm,
                              &transposeANode, basenameA) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate transpose node A"),
        return FAILED);
    // relink a
    FUSION_PASS_CHECK(
        Relink(aNode, transposeANode, gemmNode, aAnchor) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to relink nodes"), return FAILED);
    fusionNodes.push_back(transposeANode);
    op.SetAttr("transpose_a", false);
  }

  if (transposeB && needTranspose) {
    // transpose b
    FUSION_PASS_CHECK(
        GenerateTransposeNode(&graph, bOutDesc, &gemmBInDesc, transPerm,
                              &transposeBNode, basenameB) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to generate transpose node B"),
        return FAILED);
    // relink b
    FUSION_PASS_CHECK(
        Relink(bNode, transposeBNode, gemmNode, bAnchor) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "fail to relink nodes"), return FAILED);
    fusionNodes.push_back(transposeBNode);
    op.SetAttr("transpose_b", false);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End GemmTransFusionPass.");
  return SUCCESS;
}

REGISTER_PASS("GemmTransFusionPass", BUILT_IN_GRAPH_PASS, GemmTransFusionPass);
}  // namespace fe
