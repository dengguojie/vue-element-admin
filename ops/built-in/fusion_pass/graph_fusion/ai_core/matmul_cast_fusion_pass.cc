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
 * \file matmul_cast_fusion_pass.cpp
 * \brief matmul cast fusion (Matmul--Cast)
 */
#include "matmul_cast_fusion_pass.h"
#include <memory>
#include <string>
#include <vector>

#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"
#include "error_util.h"

namespace fe {
static const string PATTERN_MATMUL = "matmul";
static const string PATTERN_CAST = "cast";
static const string OUT_T = "out_T";
static const string DATA_TYPE_FIXED = "DataTypeFixed";
static const uint8_t MATMUL_OUTPUT_NUM = 1;
static const string MATMUL_DATATYPE_ATTR_KEY = "T";
static const string CAST_DATATYPE_DES_ATTR_KEY = "DstT";

static const char* TF_MATMUL = "MatMul";
static const char* TF_MATMULV2 = "MatMulV2";

/*
    fusion pattern
            node
                \
                 \
                Matmul---Cast---
                /
               /
            node
*/
vector<FusionPattern*> MatmulCastFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("matmulCastFusion");
  if (pattern == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr,Create pattern not success!");
    return patterns;
  }
  pattern->AddOpDesc(PATTERN_MATMUL, {TF_MATMUL, TF_MATMULV2})
      .AddOpDesc(PATTERN_CAST, {CAST})
      .SetInputs(PATTERN_CAST, {PATTERN_MATMUL})
      .SetOutput(PATTERN_CAST);
  patterns.push_back(pattern);
  return patterns;
}

Status MatmulCastFusionPass::IsMatch(const ge::NodePtr &matmulNode, const ge::NodePtr &castNode) {
  std::shared_ptr<ge::OpDesc> matmulOp = matmulNode->GetOpDesc();
  std::shared_ptr<ge::OpDesc> castOp = castNode->GetOpDesc();
  FUSION_PASS_CHECK(matmulOp == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "MATMUL ops is null"), return FAILED);
  FUSION_PASS_CHECK(castOp == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Cast ops is null"), return FAILED);
  if (ge::AttrUtils::HasAttr(matmulOp, MATMUL_DATATYPE_ATTR_KEY) &&
      ge::AttrUtils::HasAttr(castOp, CAST_DATATYPE_DES_ATTR_KEY)) {
    // get matmul datatype
    ge::DataType matmulDataType = ge::DT_UNDEFINED;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetDataType(matmulOp, MATMUL_DATATYPE_ATTR_KEY, matmulDataType),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Fail to get attr T from matmul."), return FAILED);
    // get cast dest datatype
    ge::DataType castDesDataType = ge::DT_UNDEFINED;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetDataType(castOp, CAST_DATATYPE_DES_ATTR_KEY, castDesDataType),
                      OP_LOGD(FUSED_OP_TYPE.c_str(), "Fail to get attr DstT from cast."), return FAILED);
    if (matmulDataType != ge::DT_FLOAT16 || castDesDataType != ge::DT_FLOAT) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Matmul datatype is %u Cast data type is %u", matmulDataType, castDesDataType);
      return FAILED;
    }
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Do not have attr %s or %s", MATMUL_DATATYPE_ATTR_KEY.c_str(),
            CAST_DATATYPE_DES_ATTR_KEY.c_str());
    return FAILED;
  }
  if (matmulNode->GetOutDataNodes().size() != MATMUL_OUTPUT_NUM) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "matmul outputs num shoubld be 1");
    return FAILED;
  }
  return SUCCESS;
}

Status MatmulCastFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr matmulNode = GetNodeFromMapping(PATTERN_MATMUL, mapping);
  ge::NodePtr castNode = GetNodeFromMapping(PATTERN_CAST, mapping);

  FUSION_PASS_CHECK(matmulNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "matmul node is null"), return FAILED);
  FUSION_PASS_CHECK(castNode == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "cast node is null"), return FAILED);

  FUSION_PASS_CHECK(!CheckOpSupported(matmulNode->GetOpDesc()),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Matmul[%s] is not supported by FE, fusion abort.",
                            matmulNode->GetOpDesc()->GetName().c_str()),
                    return NOT_CHANGED);

  if (IsMatch(matmulNode, castNode) != SUCCESS) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s] and node[%s] don't match Matmul + Cast fusion pattern.",
            matmulNode->GetName().c_str(), castNode->GetName().c_str());
    return NOT_CHANGED;
  }
  if (DoFusion(matmulNode) == FAILED) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "matmul and cast fusion failed!");
    return FAILED;
  }
  ge::GeTensorDesc castOutputDesc = castNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(matmulNode->GetOpDesc()->UpdateOutputDesc(0, castOutputDesc) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to update output desc of MatMul node[%s].",
                            matmulNode->GetOpDesc()->GetName().c_str()),
                    return FAILED);
  if (PatternFusionUtil::RemoveInputEdge(castNode) == FAILED) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "link output edge Failed.");
    return FAILED;
  }
  if (LinkOutputEdgeWithoutControl(castNode, matmulNode) == FAILED) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "link output edge Failed.");
    return FAILED;
  }
  // link matmul output with cast output and remove cast node
  if (graph.RemoveNode(castNode) == ge::GRAPH_FAILED) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "cast node remove failed");
    return FAILED;
  }

  fusionNodes.push_back(matmulNode);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Node[%s] do Matmul + Cast fusion success!", matmulNode->GetName().c_str());
  return SUCCESS;
}

Status MatmulCastFusionPass::DoFusion(const ge::NodePtr &matmulNode) {
  std::shared_ptr<ge::OpDesc> matmulOp = matmulNode->GetOpDesc();
  FUSION_PASS_CHECK(matmulOp == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "matmul op is null"), return FAILED);
  if (ge::AttrUtils::SetInt(matmulOp, OUT_T, ge::DT_FLOAT) == false) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set Matmul[%s]'s out_T DT_FLOAT failed", matmulOp->GetName().c_str());
    return FAILED;
  }

  if (ge::AttrUtils::SetBool(matmulOp, DATA_TYPE_FIXED, true) == false) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "set Matmul[%s]'s DataTypeFixed true failed", matmulOp->GetName().c_str());
    return FAILED;
  }

  ge::GeTensorDescPtr tensorDesc = matmulOp->MutableOutputDesc(0);
  FUSION_PASS_CHECK(tensorDesc == nullptr, OP_LOGD(FUSED_OP_TYPE.c_str(), "tensor desc is null"), return FAILED);
  return SUCCESS;
}

Status MatmulCastFusionPass::LinkOutputEdgeWithoutControl(const ge::NodePtr &oldNode, const ge::NodePtr &newNode) {
  ge::OutDataAnchorPtr newOutDataAnchor = newNode->GetOutDataAnchor(0);
  if (newOutDataAnchor == nullptr) {
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[newOutDataAnchor] must not be null.");
    return fe::PARAM_INVALID;
  }
  for (ge::OutDataAnchorPtr &anchor : oldNode->GetAllOutDataAnchors()) {
    if (anchor == nullptr) {
      CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Parameter[anchor] must not be null.");
      return fe::PARAM_INVALID;
    }
    for (ge::InDataAnchorPtr &dstAnchor : anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(anchor, dstAnchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(newOutDataAnchor, dstAnchor) != ge::GRAPH_SUCCESS) {
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace out data anchor Failed.");
        return FAILED;
      }
    }
  }
  auto outControlAnchor = oldNode->GetOutControlAnchor();
  if (outControlAnchor != nullptr) {
    for (ge::InControlAnchorPtr &dstAnchor : outControlAnchor->GetPeerInControlAnchors()) {
      if (ge::GraphUtils::RemoveEdge(outControlAnchor, dstAnchor) != ge::GRAPH_SUCCESS ||
          ge::GraphUtils::AddEdge(newNode->GetOutControlAnchor(), dstAnchor) != ge::GRAPH_SUCCESS) {
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Replace input control anchor Failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
REGISTER_PASS("MatmulCastFusionPass", BUILT_IN_GRAPH_PASS, MatmulCastFusionPass);
}  // namespace fe
