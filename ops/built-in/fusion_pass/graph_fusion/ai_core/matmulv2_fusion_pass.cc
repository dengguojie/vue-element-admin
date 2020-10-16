/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief MatMulV2 fusion pass
 *
 */

#include <memory>
#include <string>
#include "matmulv2_fusion_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"

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
vector<FusionPattern *> MatMulV2FusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MatMulV2FusionPass pattern begin");
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("MatMulV2FusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_MATMULV2, {MATMULV2})
      .AddOpDesc(PATTERN_INPUTS1)
      .AddOpDesc(PATTERN_INPUTS2)
      .AddOpDesc(PATTERN_INPUTS3)
      .SetInputs(PATTERN_MATMULV2,
                 {PATTERN_INPUTS1, PATTERN_INPUTS2, PATTERN_INPUTS3})
      .SetOutput(PATTERN_MATMULV2);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MatMulV2FusionPass pattern end");
  return patterns;
}

Status MatMulV2FusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                    vector<ge::NodePtr> &fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MatMulV2FusionPass fusion begin");
  ge::NodePtr matMulV2Node = GetNodeFromMapping(PATTERN_MATMULV2, mapping);
  FUSION_PASS_CHECK(matMulV2Node == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "padD Node is null, fusion failed."), return PARAM_INVALID);

  bool transposeB = false;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(matMulV2Node->GetOpDesc(), TRANSPOSEB,
                                   transposeB),
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Get transpose_b attr failed."), return NOT_CHANGED);

  if (transposeB == false) {
    return NOT_CHANGED;
  }
  size_t constInputDimNum = matMulV2Node->GetOpDesc()
                                ->GetInputDesc(CONST_INDEX)
                                .GetShape()
                                .GetDimNum();
  if (constInputDimNum != CONST_DIM_NUM) {
    return NOT_CHANGED;
  }

  ge::DataType constDataType =
      matMulV2Node->GetOpDesc()->GetInputDesc(CONST_INDEX).GetDataType();
  if (constDataType != ge::DT_INT8) {
    return NOT_CHANGED;
  }

  FUSION_PASS_CHECK(
      !ge::AttrUtils::SetBool(matMulV2Node->GetOpDesc(), TRANSPOSEB, false),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Set transpose_b attr failed."), return FAILED);

  FUSION_PASS_CHECK(
      !ge::AttrUtils::SetBool(matMulV2Node->GetOpDesc(), TRANSPOSEBX2, false),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Set transpose_x2 attr failed."), return FAILED);

  std::vector<int64_t> oldDims =
      matMulV2Node->GetOpDesc()->GetInputDesc(CONST_INDEX).GetShape().GetDims();
  vector<int64_t> newDims;
  newDims.push_back(oldDims[1]);
  newDims.push_back(oldDims[0]);
  ge::GeShape newShape(newDims);
  matMulV2Node->GetOpDesc()->MutableInputDesc(CONST_INDEX)->SetShape(newShape);
  matMulV2Node->GetOpDesc()->MutableInputDesc(CONST_INDEX)->SetOriginShape(newShape);

  ge::NodePtr constNode = matMulV2Node->GetInDataAnchor(CONST_INDEX)
                              ->GetPeerOutAnchor()
                              ->GetOwnerNode();

  std::shared_ptr<ge::OpDesc> transposeOpdesc = std::make_shared<ge::OpDesc>(
      constNode->GetName() + "_transpose_b", TRANSPOSED_TYPE);

  vector<int64_t> perm;
  perm.push_back(1);
  perm.push_back(0);

  FUSION_PASS_CHECK(
      !ge::AttrUtils::SetListInt(transposeOpdesc, PERM, perm),
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set perm to %s failed.", transposeOpdesc->GetName().c_str()),
      return FAILED);

  ge::GeTensorDesc inputDesc = constNode->GetOpDesc()->GetOutputDesc(0);
  ge::GeTensorDesc outputDesc =
      matMulV2Node->GetOpDesc()->GetInputDesc(CONST_INDEX);

  FUSION_PASS_CHECK(
      transposeOpdesc->AddInputDesc("x", inputDesc) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "%s add inputDesc failed.", transposeOpdesc->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(
      transposeOpdesc->AddOutputDesc("y", outputDesc) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "%s add outputDesc failed.", transposeOpdesc->GetName().c_str()),
      return FAILED);

  ge::NodePtr transposeNode = graph.AddNode(transposeOpdesc);

  ge::OutDataAnchorPtr src = constNode->GetOutDataAnchor(0);
  ge::InDataAnchorPtr dst = matMulV2Node->GetInDataAnchor(CONST_INDEX);

  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "remove %s input0 edge error", matMulV2Node->GetName().c_str()),
      return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(src, transposeNode->GetInDataAnchor(0)) !=
               SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   constNode->GetName().c_str(),
                   transposeNode->GetName().c_str()),
           return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposeNode->GetOutDataAnchor(0), dst) !=
               SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                   transposeNode->GetName().c_str(),
                   matMulV2Node->GetName().c_str()),
           return FAILED);

  fusionNodes.push_back(transposeNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define MatMulV2FusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("MatMulV2FusionPass", BUILT_IN_GRAPH_PASS, MatMulV2FusionPass);
}  // namespace fe
