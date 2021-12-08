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
 * \file matmul_transdata_fusion_pass.cpp
 * \brief matmu transdata fusion pass
 *     input0            data          input0
 *          \            /                \
 *        matmul     transdata          matmul
 *           |          \                  /
 *           |           \                /
 *          mul        mul            transdata     data
 *           |         /                 \         /
 *           |      /        =>           \       /
 *          add                          mul    mul
 *           |                            \     /
 *           |                             \  /
 *       transdata                        add
 *           |                             \
 *           |                              \
 *         out                              out
 *
 */
#include "matmul_transdata_fusion_pass.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "anchor_util.h"
#include "common/util/platform_info.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"

namespace fe {
const string MatMulTransdataFusionPass::FUSED_OP_TYPE = "GEMM";

namespace {
static const char OP_TYPE_MATMUL[] = "MatMulV2";
static const char OP_TYPE_TRANDATA[] = "TransData";
static const char OP_TYPE_MUL[] = "Mul";
static const char OP_TYPE_ADD[] = "Add";
static const char OP_TYPE_CAST[] = "Cast";

static const char PATTERN_MATMUL[] = "Pattern_Matmul";
static const char PATTERN_TRANDATA[] = "Pattern_Transdata";
static const char PATTERN_MUL[] = "Pattern_Mul";
static const char PATTERN_ADD[] = "Pattern_Add";
static const char PATTERN_CAST[] = "Pattern_Cast";

}  // namespace

vector<FusionPattern*> MatMulTransdataFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter MatmulTransdata::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern0 = new (std::nothrow) FusionPattern("MatmulTransdataFusionPass");
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("MatmulTransdataFusionPass");
  FUSION_PASS_CHECK(
    pattern0 == nullptr || pattern1 == nullptr,
    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
    return patterns
  );
  pattern0->AddOpDesc(PATTERN_MATMUL, {OP_TYPE_MATMUL})
           .AddOpDesc(PATTERN_TRANDATA, {OP_TYPE_TRANDATA})
           .AddOpDesc(PATTERN_MUL, {OP_TYPE_MUL})
           .AddOpDesc(PATTERN_ADD, {OP_TYPE_ADD})
           .AddOpDesc(PATTERN_CAST, {OP_TYPE_CAST})
           .SetInputs(PATTERN_CAST, {PATTERN_MATMUL})
           .SetInputs(PATTERN_MUL, {PATTERN_CAST})
           .SetInputs(PATTERN_ADD, {PATTERN_MUL})
           .SetInputs(PATTERN_TRANDATA, {PATTERN_ADD})
           .SetOutput(PATTERN_TRANDATA);
  patterns.push_back(pattern0);

  pattern1->AddOpDesc(PATTERN_MATMUL, {OP_TYPE_MATMUL})
           .AddOpDesc(PATTERN_TRANDATA, {OP_TYPE_TRANDATA})
           .AddOpDesc(PATTERN_MUL, {OP_TYPE_MUL})
           .AddOpDesc(PATTERN_ADD, {OP_TYPE_ADD})
           .SetInputs(PATTERN_MUL, {PATTERN_MATMUL})
           .SetInputs(PATTERN_ADD, {PATTERN_MUL})
           .SetInputs(PATTERN_TRANDATA, {PATTERN_ADD})
           .SetOutput(PATTERN_TRANDATA);
  patterns.push_back(pattern1);
  return patterns;
}


Status MatMulTransdataFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                         vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter MatMulTransdataFusionPass.");
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(
    PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "Get platform_info failed."),
    return FAILED
  );
  map<string, vector<string>> intrinsic_map = platform_info.ai_core_intrinsic_dtype_map;
  if (intrinsic_map.size() == 0 || intrinsic_map.find("Intrinsic_fix_pipe_l0c2out") == intrinsic_map.end()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "this version no need change gemm to matmul");
    return NOT_CHANGED;
  }

  ge::NodePtr matmul_node = GetNodeFromMapping(PATTERN_MATMUL, mapping);
  ge::NodePtr left_transdata_node = GetNodeFromMapping(PATTERN_TRANDATA, mapping);
  ge::NodePtr left_mul_node = GetNodeFromMapping(PATTERN_MUL, mapping);
  ge::NodePtr add_node = GetNodeFromMapping(PATTERN_ADD, mapping);
  ge::NodePtr cast_node = GetNodeFromMapping(PATTERN_CAST, mapping);
  FUSION_PASS_CHECK(
    matmul_node == nullptr || left_transdata_node == nullptr || left_mul_node == nullptr || add_node == nullptr,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "some node is null, fusion failed."),
    return PARAM_INVALID
  );

  ge::NodePtr right_mul_node = GetPeerOutNodeWithInDataAnchor(add_node, 1);
  FUSION_PASS_CHECK(
    right_mul_node == nullptr,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "right_mul_node is null, fusion failed."),
    return NOT_CHANGED
  );
  ge::NodePtr right_transdata_node = GetPeerOutNodeWithInDataAnchor(right_mul_node, 0);
  FUSION_PASS_CHECK(
    right_transdata_node == nullptr || right_transdata_node->GetType() != OP_TYPE_TRANDATA,
    OP_LOGI(FUSED_OP_TYPE.c_str(), "right_transdata_node is null or not transdata, fusion failed."),
    return NOT_CHANGED
  );

  // get the input and output node
  ge::NodePtr c_node = GetPeerOutNodeWithInDataAnchor(right_transdata_node, 0);
  auto left_transdata_out_ptr = left_transdata_node->GetOutDataAnchor(0);
  vector<InDataAnchorPtr> y_anchors_ptr;
  for (auto in_data_anchor : left_transdata_out_ptr->GetPeerInDataAnchors()) {
    y_anchors_ptr.push_back(in_data_anchor);
  }
  auto y_node = GetPeerInNodeByOutDataAnchor(left_transdata_out_ptr, 0);

  bool with_cast_flag = (cast_node != nullptr);
  // remove some edge of left side
  if (with_cast_flag) {
    FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(matmul_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between matmul_node and cast_node"),
      return FAILED
    );
  } else {
    FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(matmul_node->GetOutDataAnchor(0), left_mul_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between matmul_node and left_mul_node"),
      return FAILED
    );
  }
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(add_node->GetOutDataAnchor(0), left_transdata_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between add_node and left_transdata_node"),
    return FAILED
  );
  for (auto input_data_anchor : y_anchors_ptr) {
    FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(left_transdata_node->GetOutDataAnchor(0), input_data_anchor) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between transdata_node and y_nodes"),
      return FAILED
    );
  }

  // remove some edge of right side
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(c_node->GetOutDataAnchor(0), right_transdata_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between c_node and right_transdata_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    ge::GraphUtils::RemoveEdge(right_transdata_node->GetOutDataAnchor(0),
                               right_mul_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                            "fail to remove edge between right_transdata_node and right_mul_node"),
    return FAILED
  );

  // add some edge
  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), left_transdata_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between matmul_node and left_transdata_node"),
    return FAILED
  );

  if (with_cast_flag) {
    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(left_transdata_node->GetOutDataAnchor(0), cast_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between left_transdata_node and cast_node"),
      return FAILED
    );
  } else {
    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(left_transdata_node->GetOutDataAnchor(0), left_mul_node->GetInDataAnchor(0)) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                              "fail to add edge between left_transdata_node and left_mul_node"),
      return FAILED
    );
  }

  FUSION_PASS_CHECK(
    ge::GraphUtils::AddEdge(c_node->GetOutDataAnchor(0), right_mul_node->GetInDataAnchor(0)) != SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to add edge between c_node and right_mul_node"),
    return FAILED
  );

  for (auto input_data_anchor : y_anchors_ptr) {
    FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), input_data_anchor) != SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove edge between add_node and y_nodes"),
      return FAILED
    );
  }

  // updata the node desc
  auto left_transdata_node_outdesc = left_transdata_node->GetOpDesc()->GetOutputDesc(0);
  auto left_mul_node_desc = left_mul_node->GetOpDesc();
  auto right_mul_node_desc = right_mul_node->GetOpDesc();
  auto add_node_desc = add_node->GetOpDesc();
  FUSION_PASS_CHECK(
    left_mul_node_desc->UpdateInputDesc(0, left_transdata_node_outdesc) !=  SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update input description of left_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    left_mul_node_desc->UpdateOutputDesc(0, left_transdata_node_outdesc) !=  SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update output description of left_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    add_node_desc->UpdateInputDesc(0, left_transdata_node_outdesc) !=  SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update input description of add_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    add_node_desc->UpdateOutputDesc(0, left_transdata_node_outdesc) !=  SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update output description of add_node"),
    return FAILED
  );
  if (with_cast_flag) {
    auto cast_node_desc = cast_node->GetOpDesc();
    auto left_transdata_node_desc = left_transdata_node->GetOpDesc();
    auto matmul_node_outdesc = matmul_node->GetOpDesc()->GetOutputDesc(0);
    FUSION_PASS_CHECK(
      cast_node_desc->UpdateOutputDesc(0, left_transdata_node_outdesc) !=  SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update input description of left_mul_node"),
      return FAILED
    );
    left_transdata_node_outdesc.SetDataType(matmul_node_outdesc.GetDataType());
    left_transdata_node_outdesc.SetOriginDataType(matmul_node_outdesc.GetOriginDataType());
    FUSION_PASS_CHECK(
      cast_node_desc->UpdateInputDesc(0, left_transdata_node_outdesc) !=  SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update input description of cast_node_outdesc"),
      return FAILED
    );
    FUSION_PASS_CHECK(
      left_transdata_node_desc->UpdateInputDesc(0, matmul_node_outdesc) !=  SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                              "fail to update input description of left_transdata_node_outdesc"),
      return FAILED
    );
    FUSION_PASS_CHECK(
      left_transdata_node_desc->UpdateOutputDesc(0, left_transdata_node_outdesc) !=  SUCCESS,
      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
                              "fail to update output description of left_transdata_node_outdesc"),
      return FAILED
    );
  }
  auto c_node_outdesc = c_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(
    right_mul_node_desc->UpdateInputDesc(0, c_node_outdesc) !=  SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update input description of right_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    right_mul_node_desc->UpdateOutputDesc(0, c_node_outdesc) !=  SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update output description of right_mul_node"),
    return FAILED
  );
  FUSION_PASS_CHECK(
    add_node_desc->UpdateInputDesc(1, c_node_outdesc) !=  SUCCESS,
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to update input description of add_node"),
    return FAILED
  );

  // RemoveNode(right_transdata_node)
  FUSION_PASS_CHECK(
    graph.RemoveNode(right_transdata_node),
    ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to remove right_transdata_node"),
    return FAILED
  );
  OP_LOGI(FUSED_OP_TYPE.c_str(), "remove right_transdata_node");
  return SUCCESS;
}

REGISTER_PASS("MatMulTransdataFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, MatMulTransdataFusionPass);
}  // namespace fe
