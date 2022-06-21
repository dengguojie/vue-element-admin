/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
 * \file swin_attention_score_fusion_pass.cc
 * pattern:
 *      
 *             
 *    x1    const0    x2    
 *     \     /        |            
 *       mul     transpose
 *         \       /
 *        batch_matmul  add    
 *                 \    /        
 *                  add       
 *                   |                          x1  x2  x3    add add2 const0  
 *                reshape    add2         ->      \   \   \    /   /   /
 *                    \     /                      swin_attention_score
 *                      add       
 *                       |               
 *                    reshape      
 *                       |
 *                    softmax       x3
 *                         \        /
 *                        batch_matmul
 *                             |
 *                          transpose
 *                          
 */
#include <stdlib.h>
#include "swin_attention_score_fusion_pass.h"
#include "common/util/platform_info.h"
#include "anchor_util.h"
#include "error_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "external/graph/operator_factory.h"

using namespace std;
using namespace ge;
namespace fe {
static const string PATTERN_INPUT0 = "input0";
static const string PATTERN_INPUT1 = "input1";
static const string PATTERN_INPUT2 = "input2";
static const string PATTERN_INPUT3 = "input3";
static const string PATTERN_CONST = "const";
static const string PATTERN_MUL = "mul";
static const string PATTERN_BATCHMATMUL = "batch_matmul";
static const string PATTERN_ADD = "add";
static const string PATTERN_RESHAPE = "reshape";
static const string PATTERN_ADD2 = "add2";
static const string PATTERN_RESHAPE2 = "reshape2";
static const string PATTERN_SOFTMAXV2WITHDROPOUT = "softmax_v2_with_dropout";
static const string PATTERN_SOFTMAXV2 = "softmax_v2";
static const string PATTERN_BATCHMATMUL2 = "batch_matmul2";
static const string PATTERN_TRANSPOSED = "transpose_d";
static const string PATTERN_TRANSPOSED2 = "transpose_d2";
static const string PATTERN_RESHAPE3 = "reshape3";

static const string MUL = "Mul";
static const string CONST = "Const";
static const string BATCHMATMULV2 = "BatchMatMulV2";
static const string ADD = "Add";
static const string RESHAPE = "Reshape";
static const string SOFTMAXV2WITHDROPOUTDOMASKV3D = "SoftmaxV2WithDropOutDoMaskV3D";
static const string SOFTMAXV2 = "SoftmaxV2";
static const string TRANSPOSED = "TransposeD";
static const string TRANSPOSE = "Transpose";
static const int64_t ALIGN_UNIT = 16;
static const int64_t ALIGN_UNIT_BASE = 12;
static const int64_t CONFUSION_DIM_ONE = 12288;
static const int64_t CONFUSION_DIM_TWO = 768;
static const int64_t NUM_TWO = 2;
static const int64_t NUM_THREE = 3;
static const int64_t NUM_FOUR = 4;
static const int64_t NUM_FIVE = 5;
static const int64_t NUM_SIX = 6;
static bool is_inference_plateform = false;
static const std::vector<std::string> SUPPORT_PLATFORM_PATTERN = {"Ascend310P"};
static const string kNameFusionPass = "SwinAttentionScoreFusionPass";

vector<FusionPattern *> SwinAttentionScoreFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern."),
                    return patterns);

  OP_LOGD(kNameFusionPass, "Start to define pattern");
  pattern->AddOpDesc(PATTERN_INPUT0)
    .AddOpDesc(PATTERN_MUL, {MUL})
    .AddOpDesc(PATTERN_INPUT1)
    .AddOpDesc(PATTERN_TRANSPOSED, {TRANSPOSED, TRANSPOSE})
    .AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_ADD, {ADD})
    .AddOpDesc(PATTERN_SOFTMAXV2, {SOFTMAXV2})
    .AddOpDesc(PATTERN_INPUT2)
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_TRANSPOSED2, {TRANSPOSED, TRANSPOSE})
    .AddOpDesc(PATTERN_RESHAPE3, {RESHAPE})
    .SetInputs(PATTERN_MUL, {PATTERN_INPUT0})
    .SetInputs(PATTERN_TRANSPOSED, {PATTERN_INPUT1})
    .SetInputs(PATTERN_BATCHMATMUL, {PATTERN_MUL, PATTERN_TRANSPOSED})
    .SetInputs(PATTERN_ADD, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_SOFTMAXV2, {PATTERN_ADD})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXV2, PATTERN_INPUT2})
    .SetInputs(PATTERN_TRANSPOSED2, {PATTERN_BATCHMATMUL2})
    .SetInputs(PATTERN_RESHAPE3, {PATTERN_TRANSPOSED2})
    .SetOutput(PATTERN_RESHAPE3);
  patterns.push_back(pattern);
  OP_LOGD(kNameFusionPass, "End to define pattern.");

  FusionPattern *pattern1 = new (std::nothrow) FusionPattern(kNameFusionPass);
  FUSION_PASS_CHECK(pattern1 == nullptr, OP_LOGW(kNameFusionPass, "Failed to create pattern1."),
                    return patterns);
  OP_LOGD(kNameFusionPass, "Start to define pattern1.");
  pattern1->AddOpDesc(PATTERN_INPUT0)
    .AddOpDesc(PATTERN_MUL, {MUL})
    .AddOpDesc(PATTERN_INPUT1)
    .AddOpDesc(PATTERN_TRANSPOSED, {TRANSPOSED, TRANSPOSE})
    .AddOpDesc(PATTERN_BATCHMATMUL, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_ADD, {ADD})
    .AddOpDesc(PATTERN_RESHAPE, {RESHAPE})
    .AddOpDesc(PATTERN_ADD2, {ADD})
    .AddOpDesc(PATTERN_RESHAPE2, {RESHAPE})
    .AddOpDesc(PATTERN_SOFTMAXV2, {SOFTMAXV2})
    .AddOpDesc(PATTERN_INPUT2)
    .AddOpDesc(PATTERN_BATCHMATMUL2, {BATCHMATMULV2})
    .AddOpDesc(PATTERN_TRANSPOSED2, {TRANSPOSED, TRANSPOSE})
    .AddOpDesc(PATTERN_RESHAPE3, {RESHAPE})
    .SetInputs(PATTERN_MUL, {PATTERN_INPUT0})
    .SetInputs(PATTERN_TRANSPOSED, {PATTERN_INPUT1})
    .SetInputs(PATTERN_BATCHMATMUL, {PATTERN_MUL, PATTERN_TRANSPOSED})
    .SetInputs(PATTERN_ADD, {PATTERN_BATCHMATMUL})
    .SetInputs(PATTERN_RESHAPE, {PATTERN_ADD})
    .SetInputs(PATTERN_ADD2, {PATTERN_RESHAPE})
    .SetInputs(PATTERN_RESHAPE2, {PATTERN_ADD2})
    .SetInputs(PATTERN_SOFTMAXV2, {PATTERN_RESHAPE2})
    .SetInputs(PATTERN_BATCHMATMUL2, {PATTERN_SOFTMAXV2, PATTERN_INPUT2})
    .SetInputs(PATTERN_TRANSPOSED2, {PATTERN_BATCHMATMUL2})
    .SetInputs(PATTERN_RESHAPE3, {PATTERN_TRANSPOSED2})
    .SetOutput(PATTERN_RESHAPE3);
  patterns.push_back(pattern1);
  OP_LOGD(kNameFusionPass, "End to define pattern1.");

  return patterns;
}

bool SwinAttentionScoreFusionPass::IsTargetPlateform(const std::string plateform) {
  OP_LOGD(kNameFusionPass, "IsTargetPlateform begin");
  PlatformInfo platformInfo;
  OptionalInfo optionalInfo;
  FUSION_PASS_CHECK(
      PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != fe::SUCCESS,
      OP_LOGW(kNameFusionPass, "Failed to get platform info"), return NOT_CHANGED);

  std::string socVersion = optionalInfo.soc_version;
  bool is_target = false;
  if (socVersion == plateform || socVersion.find(plateform) != string::npos) {
    is_target = true;
  }

  OP_LOGD(kNameFusionPass, "IsTargetPlateform end");
  return is_target;
}

Status SwinAttentionScoreFusionPass::Fusion(ge::ComputeGraph &graph,
                                            Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kNameFusionPass, "Start SwinAttentionScoreFusionPass::Fusion.");
  ge::NodePtr mul_node = GetNodeFromMapping(PATTERN_MUL, mapping);
  FUSION_PASS_CHECK(mul_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get mul_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr transposed_node = GetNodeFromMapping(PATTERN_TRANSPOSED, mapping);
  FUSION_PASS_CHECK(transposed_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get transposed_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr bmm_node = GetNodeFromMapping(PATTERN_BATCHMATMUL, mapping);
  FUSION_PASS_CHECK(bmm_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get bmm_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr add_node = GetNodeFromMapping(PATTERN_ADD, mapping);
  FUSION_PASS_CHECK(add_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get add_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr reshape_node = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  ge::NodePtr add2_node = GetNodeFromMapping(PATTERN_ADD2, mapping);
  ge::NodePtr reshape2_node = GetNodeFromMapping(PATTERN_RESHAPE2, mapping);
  ge::NodePtr softmaxv2_node = GetNodeFromMapping(PATTERN_SOFTMAXV2, mapping);
  FUSION_PASS_CHECK(softmaxv2_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get softmaxv2_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr bmm2_node = GetNodeFromMapping(PATTERN_BATCHMATMUL2, mapping);
  FUSION_PASS_CHECK(bmm2_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get bmm2_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr transposed2_node = GetNodeFromMapping(PATTERN_TRANSPOSED2, mapping);
  FUSION_PASS_CHECK(transposed2_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get transposed2_node not success."),
                    return NOT_CHANGED);
  ge::NodePtr reshape3_node = GetNodeFromMapping(PATTERN_RESHAPE3, mapping);
  FUSION_PASS_CHECK(reshape3_node == nullptr,
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "Get reshape3_node not success."),
                    return NOT_CHANGED);
  is_inference_plateform = IsTargetPlateform(SUPPORT_PLATFORM_PATTERN[0]);
  FUSION_PASS_CHECK(!is_inference_plateform, OP_LOGW(kNameFusionPass, "Only support 310p series platform"),
                    return NOT_CHANGED);
  std::shared_ptr<ge::OpDesc> bsb_desc = nullptr;
  FUSION_PASS_MAKE_SHARED(
    bsb_desc = std::make_shared<ge::OpDesc>(softmaxv2_node->GetName() + "/SwinAttentionScore", "SwinAttentionScore"),
                                            return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("query",
                                           *(mul_node->GetOpDesc()->MutableInputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input0 of SwinAttentionScore failed."), return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("key",
                                           *(transposed_node->GetOpDesc()->MutableInputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input1 of SwinAttentionScore failed."), return FAILED);

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("value",
                                           *(bmm2_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input2 of SwinAttentionScore failed."), return FAILED);
  
  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("padding_mask1",
                                           *(add_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input3 of SwinAttentionScore failed."), return FAILED);

  if (add2_node != nullptr) {
    FUSION_PASS_CHECK(bsb_desc->AddInputDesc("padding_mask2",
                                             *(add2_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input4 of SwinAttentionScore failed."), return FAILED);
    FUSION_PASS_CHECK(bsb_desc->AddInputDesc("drop_mask",
                                             *(add2_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                      OP_LOGE(kNameFusionPass, "Add input6 of SwinAttentionScore failed."), return FAILED);
  }

  FUSION_PASS_CHECK(bsb_desc->AddInputDesc("scale",
                                           *(mul_node->GetOpDesc()->MutableInputDesc(1))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add input5 of SwinAttentionScore failed."), return FAILED);
  FUSION_PASS_CHECK(bsb_desc->AddOutputDesc("attention_score",
                                           *(reshape3_node->GetOpDesc()->MutableOutputDesc(0))) != SUCCESS,
                    OP_LOGE(kNameFusionPass, "Add output0 of SwinAttentionScore failed."), return FAILED);
  bool first_transpose_a = false;
  bool first_transpose_b = false;
  bool second_transpose_a = false;
  bool second_transpose_b = false;
  float keep_prob = 0;
  vector<int64_t> axes = {};
  ge::AttrUtils::SetFloat(bsb_desc, "keep_prob", keep_prob);
  ge::AttrUtils::SetBool(bsb_desc, "query_transpose", first_transpose_a);
  ge::AttrUtils::SetBool(bsb_desc, "key_transpose", first_transpose_b);
  ge::AttrUtils::SetBool(bsb_desc, "bmm_score_transpose_a", second_transpose_a);
  ge::AttrUtils::SetBool(bsb_desc, "bmm_score_transpose_b", second_transpose_b);
  ge::AttrUtils::SetListInt(bsb_desc, "softmax_axes", axes);

  ge::NodePtr bsb_node = graph.AddNode(bsb_desc);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transposed_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(1)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(bmm2_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(2)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                            bsb_node->GetInDataAnchor(3)) != SUCCESS,
                    OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
  if (add2_node != nullptr) {
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add2_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                              bsb_node->GetInDataAnchor(4)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                              bsb_node->GetInDataAnchor(5)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(add2_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                              bsb_node->GetInDataAnchor(6)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
  } else { 
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(mul_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
                                              bsb_node->GetInDataAnchor(4)) != SUCCESS,
                      OP_LOGW(kNameFusionPass, "Failed to get nodes."), return FAILED);
  } 
  AddOutputEdgeForNode(reshape3_node, bsb_node, 0, 0);
  
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(mul_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         mul_node->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(transposed_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         transposed_node->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(bmm_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         bmm_node->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(add_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         add_node->GetName().c_str()), return FAILED);
  if (reshape_node != nullptr) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(reshape_node),
                      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                           reshape_node->GetName().c_str()), return FAILED);
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(add2_node),
                      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                           add2_node->GetName().c_str()), return FAILED);
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(reshape2_node),
                      CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                           reshape2_node->GetName().c_str()), return FAILED);
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(softmaxv2_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         softmaxv2_node->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(bmm2_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         bmm2_node->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(transposed2_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         transposed2_node->GetName().c_str()), return FAILED);
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(reshape3_node),
                    CUBE_CALL_ERR_REPORT(kNameFusionPass, "remove fusedNode node[%s] failed",
                                         reshape3_node->GetName().c_str()), return FAILED);
                    
  return SUCCESS;
}

Status SwinAttentionScoreFusionPass::AddOutputEdgeForNode(ge::NodePtr ori_node, ge::NodePtr new_node,
                                                          int unlinkIndex, int new_node_index) const {
  OP_LOGD(kNameFusionPass, "Start SwinAttentionScoreFusionPass::AddOutputEdgeForNode.");
  if (ori_node->GetOutDataAnchor(unlinkIndex)->GetPeerInDataAnchors().size() > 0) {
    for (InDataAnchorPtr inAnchorPtr : ori_node->GetOutDataAnchor(unlinkIndex)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(new_node_index), inAnchorPtr),
                        CUBE_CALL_ERR_REPORT(
                          kNameFusionPass,
                          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                          new_node->GetName().c_str(), 0, new_node->GetName().c_str(), 0), return FAILED);
    }
  }
  return SUCCESS;
}


REGISTER_PASS("AAAAAASwinAttentionScoreFusionPass",
              BUILT_IN_GRAPH_PASS,
              SwinAttentionScoreFusionPass);
}  // namespace fe