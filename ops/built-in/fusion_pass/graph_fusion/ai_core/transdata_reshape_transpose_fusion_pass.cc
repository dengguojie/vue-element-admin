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
 * \file reshape_transpose_fusion_pass.cpp
 * \brief confusionTranspose fusion pass(Transpose-Reshape --> confusionTranspose)
 *     input0
 *          \
 *        transdata
 *           |
 *           |
 *       (reformat)
 *           |
 *           |                    input0
 *        reshape     ----------->    \
 *           |                       transpose
 *           |
 *       (reformat)
 *           |
 *           |
 *        transdata
 *
 */
#include "transdata_reshape_transpose_fusion_pass.h"

#include "op_log.h"
#include "pattern_fusion_util.h"
#include "../../../op_proto/util/error_util.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const std::string PATTERN_RESHAPE = "Pattern_Reshape";
static const std::string PATTERN_TRANSDATA1 = "Pattern_Transdata1";
static const std::string PATTERN_TRANSDATA2 = "Pattern_Transdata2";
static const std::string PATTERN_REFORMAT1 = "Pattern_Reformat1";
static const std::string PATTERN_REFORMAT2 = "Pattern_Reformat2";

static const std::string OP_TYPE_TRANSDATA = "TransData";
static const std::string OP_TYPE_RESHAPE = "Reshape";
static const std::string OP_TYPE_REFORMAT = "ReFormat";
static const std::string OP_TYPE_TRANSPOSE = "TransposeD";

vector<FusionPattern*> TransdataReshapeTransposeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("TransdataReshapeTransdataPattern");
  FUSION_PASS_CHECK(pattern1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern1->AddOpDesc(PATTERN_TRANSDATA1, {OP_TYPE_TRANSDATA})
           .AddOpDesc(PATTERN_TRANSDATA2, {OP_TYPE_TRANSDATA})
           .AddOpDesc(PATTERN_RESHAPE, {OP_TYPE_RESHAPE})
           .SetInputs(PATTERN_RESHAPE, {PATTERN_TRANSDATA1})
           .SetInputs(PATTERN_TRANSDATA2, {PATTERN_RESHAPE})
           .SetOutput(PATTERN_TRANSDATA2);
  patterns.push_back(pattern1);

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("TransdataReshapeReformatTransdataPattern");
  FUSION_PASS_CHECK(pattern2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern2->AddOpDesc(PATTERN_TRANSDATA1, {OP_TYPE_TRANSDATA})
           .AddOpDesc(PATTERN_TRANSDATA2, {OP_TYPE_TRANSDATA})
           .AddOpDesc(PATTERN_RESHAPE, {OP_TYPE_RESHAPE})
           .AddOpDesc(PATTERN_REFORMAT1, {OP_TYPE_REFORMAT})
           .AddOpDesc(PATTERN_REFORMAT2, {OP_TYPE_REFORMAT})
           .SetInputs(PATTERN_REFORMAT1, {PATTERN_TRANSDATA1})
           .SetInputs(PATTERN_RESHAPE, {PATTERN_REFORMAT1})
           .SetInputs(PATTERN_REFORMAT2, {PATTERN_RESHAPE})
           .SetInputs(PATTERN_TRANSDATA2, {PATTERN_REFORMAT2})
           .SetOutput(PATTERN_TRANSDATA2);
  patterns.push_back(pattern2);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Finish the fusion pattern defination.");
  return patterns;
}

Status TransdataReshapeTransposeFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                                   vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do fusion of TransdataReshapeTransposeFusionPass.");
  ge::NodePtr transdata_node1 = GetNodeFromMapping(PATTERN_TRANSDATA1, mapping);
  FUSION_PASS_CHECK(transdata_node1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Transdata1 node is null."),
                    return PARAM_INVALID);
  ge::NodePtr reshape_node = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  FUSION_PASS_CHECK(reshape_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reshape node is null."),
                    return PARAM_INVALID);
  ge::NodePtr transdata_node2 = GetNodeFromMapping(PATTERN_TRANSDATA2, mapping);
  FUSION_PASS_CHECK(transdata_node2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Transdata2 node is null."),
                    return PARAM_INVALID);

  if (!VerifyFusedNode(transdata_node1, reshape_node, transdata_node2)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Failed to verify the fused nodes.");
    return NOT_CHANGED;
  }

  ge::NodePtr reformat_node1 = GetNodeFromMapping(PATTERN_REFORMAT1, mapping);
  ge::NodePtr reformat_node2 = GetNodeFromMapping(PATTERN_REFORMAT2, mapping);

  ge::NodePtr transpose_node = CreateTransposeNode(transdata_node1, transdata_node2, graph);
  if (transpose_node == nullptr) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to create transpose node.");
    return FAILED;
  }

  if (!RelinkEdges(transdata_node1, reformat_node1, reshape_node, reformat_node2, transdata_node2,
                   transpose_node, graph)) {
    VECTOR_FUSION_INNER_ERR_REPORT(transpose_node->GetName().c_str(), "Fail to relink edge for transpose node.");
    return FAILED;
  }

  if (graph.RemoveNode(transdata_node1) != ge::SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(transdata_node1->GetName().c_str(), "Fail to remove the first transdata node.");
    return FAILED;
  }

  if (reformat_node1 != nullptr && graph.RemoveNode(reformat_node1) != ge::SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(reformat_node1->GetName().c_str(), "Fail to remove the first reformat node.");
    return FAILED;
  }

  if (graph.RemoveNode(reshape_node) != ge::SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(reshape_node->GetName().c_str(), "Fail to remove the reshape node.");
    return FAILED;
  }

  if (reformat_node2 != nullptr && graph.RemoveNode(reformat_node2) != ge::SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(reformat_node2->GetName().c_str(), "Fail to remove the second reformat node.");
    return FAILED;
  }

  if (graph.RemoveNode(transdata_node2) != ge::SUCCESS) {
    VECTOR_FUSION_INNER_ERR_REPORT(transdata_node2->GetName().c_str(), "Fail to remove the second transdata node.");
    return FAILED;
  }

  OP_LOGD(transpose_node->GetName().c_str(), "Finish the fusion of TransdataReshapeTransposeFusionPass.");
  return SUCCESS;
}

bool TransdataReshapeTransposeFusionPass::VerifyFusedNode(const ge::NodePtr &transdata_node1,
                                                          const ge::NodePtr &reshape_node,
                                                          const ge::NodePtr &transdata_node2) const {
  if (transdata_node1->GetOutDataNodesSize() > 1) {
    OP_LOGD(transdata_node1->GetName().c_str(), "This transdata node must have one output nodes.");
    return false;
  }
  if (reshape_node->GetOutDataNodesSize() > 1) {
    OP_LOGD(reshape_node->GetName().c_str(), "This reshape node must have one output nodes.");
    return false;
  }
  ge::OpDescPtr transdata1_op = transdata_node1->GetOpDesc();
  ge::OpDescPtr transdata2_op = transdata_node2->GetOpDesc();
  ge::OpDescPtr reshape_op = reshape_node->GetOpDesc();
  if (transdata1_op->GetInputDescPtr(0)->GetFormat() != ge::FORMAT_FRACTAL_NZ ||
          transdata1_op->GetOutputDescPtr(0)->GetFormat() != ge::FORMAT_ND) {
    OP_LOGD(transdata1_op->GetName().c_str(), "The transdata node1 must transfer from FRACTAL_NZ to ND.");
    return false;
  }
  if (transdata2_op->GetInputDescPtr(0)->GetFormat() != ge::FORMAT_ND ||
          transdata2_op->GetOutputDescPtr(0)->GetFormat() != ge::FORMAT_FRACTAL_NZ) {
    OP_LOGD(transdata2_op->GetName().c_str(), "The transdata node2 must transfer from ND to FRACTAL_NZ.");
    return false;
  }
  if (transdata1_op->GetInputDescPtr(0)->GetShape().GetDimNum() < 4) {
    OP_LOGD(transdata1_op->GetName().c_str(),
            "The input dim size of transdata node1 must be great or equal than 4.");
    return false;
  }
  if (transdata2_op->GetOutputDescPtr(0)->GetShape().GetDimNum() < 4) {
    OP_LOGD(transdata2_op->GetName().c_str(),
            "The output dim size of transdata node2 must be great or equal than 4.");
    return false;
  }
  ge::ConstGeTensorDescPtr reshape_input_tensor = reshape_op->GetInputDescPtr(0);
  ge::ConstGeTensorDescPtr reshape_output_tensor = reshape_op->GetOutputDescPtr(0);
  if (reshape_input_tensor->GetShape().GetDimNum() != reshape_output_tensor->GetShape().GetDimNum()) {
    OP_LOGD(reshape_op->GetName().c_str(), "The dim size of reshape node's input and output must be the same.");
    return false;
  }
  size_t reshape_dim_size = reshape_input_tensor->GetShape().GetDimNum();
  if (reshape_dim_size < 2) {
    OP_LOGD(reshape_op->GetName().c_str(),
            "The dim size of reshape node's input and output must not be small than two, but [%zu].", reshape_dim_size);
    return false;
  }

  int64_t input_dim1 = reshape_input_tensor->GetShape().GetDim(reshape_dim_size - 1);
  int64_t input_dim2 = reshape_input_tensor->GetShape().GetDim(reshape_dim_size - 2);
  int64_t output_dim1 = reshape_output_tensor->GetShape().GetDim(reshape_dim_size - 1);
  int64_t output_dim2 = reshape_output_tensor->GetShape().GetDim(reshape_dim_size - 2);
  if (input_dim1 != output_dim2 || input_dim2 != output_dim1) {
    OP_LOGD(reshape_op->GetName().c_str(),
            "The last two dim of reshape node's input and output should be exchangeable.");
    return false;
  }
  if (input_dim1 % 16 != 0 || input_dim2 % 16 != 0) {
    OP_LOGD(reshape_op->GetName().c_str(),
            "The last two dim of reshape node's input should be aligned with 16, actually is [%ld, %ld].",
            input_dim1, input_dim2);
    return false;
  }

  return true;
}

ge::NodePtr TransdataReshapeTransposeFusionPass::CreateTransposeNode(const ge::NodePtr &transdata_node1,
                                                                     const ge::NodePtr &transdata_node2,
                                                                     ge::ComputeGraph &graph) {
  ge::OpDescPtr transpose_op_desc;
  FUSION_PASS_MAKE_SHARED(
    transpose_op_desc = std::make_shared<ge::OpDesc>(transdata_node1->GetName() + "_transpose", OP_TYPE_TRANSPOSE),
    return nullptr);
  transpose_op_desc->AddInputDesc("x", transdata_node1->GetOpDesc()->GetInputDesc(0));
  transpose_op_desc->AddOutputDesc("y", transdata_node2->GetOpDesc()->GetOutputDesc(0));

  size_t dim_size = transdata_node1->GetOpDesc()->GetInputDescPtr(0)->GetShape().GetDimNum();
  vector<int64_t> perm;
  for (size_t i = 0; i < dim_size; i++) {
    perm.push_back(static_cast<int64_t>(i));
  }
  int64_t temp_val = perm[dim_size - 2];
  perm[dim_size - 2] = perm[dim_size - 1];
  perm[dim_size - 1] = temp_val;
  temp_val = perm[dim_size - 4];
  perm[dim_size - 4] = perm[dim_size - 3];
  perm[dim_size - 3] = temp_val;

  (void)ge::AttrUtils::SetListInt(transpose_op_desc, "perm", perm);
  ge::NodePtr transpose_node = graph.AddNode(transpose_op_desc);
  OP_LOGD(transpose_node->GetName().c_str(), "Transpose node has been created.");
  return transpose_node;
}

bool TransdataReshapeTransposeFusionPass::UnLinkDataEdges(const ge::NodePtr &transdata_node1,
                                                          ge::NodePtr &reformat_node1,
                                                          const ge::NodePtr &reshape_node, ge::NodePtr &reformat_node2,
                                                          const ge::NodePtr &transdata_node2,
                                                          ge::ComputeGraph &graph) {
  if (transdata_node1->GetOutDataAnchor(0) != nullptr) {
    transdata_node1->GetOutDataAnchor(0)->UnlinkAll();
  }

  if (reformat_node1 != nullptr && reformat_node1->GetOutDataAnchor(0) != nullptr) {
    reformat_node1->GetOutDataAnchor(0)->UnlinkAll();
  }

  if (reshape_node->GetOutDataAnchor(0) != nullptr) {
    reshape_node->GetOutDataAnchor(0)->UnlinkAll();
  }

  ge::InDataAnchorPtr reshape_in_anchor = reshape_node->GetInDataAnchor(1);
  if (reshape_in_anchor != nullptr) {
    if (reshape_in_anchor->GetPeerOutAnchor() != nullptr &&
        reshape_in_anchor->GetPeerOutAnchor()->GetOwnerNode() != nullptr) {
      ge::NodePtr reshape_peer_out_node = reshape_in_anchor->GetPeerOutAnchor()->GetOwnerNode();
      if (reshape_peer_out_node->GetInNodes().size() == 0 && reshape_peer_out_node->GetOutNodes().size() == 1) {
        if (graph.RemoveNode(reshape_peer_out_node) != ge::SUCCESS) {
          VECTOR_FUSION_INNER_ERR_REPORT(reshape_peer_out_node->GetName().c_str(),
                                         "Fail to remove peer out node of reshape node.");
          return false;
        }
      }
    }
    reshape_in_anchor->UnlinkAll();
  }

  if (reformat_node2 != nullptr && reformat_node2->GetOutDataAnchor(0) != nullptr) {
    reformat_node2->GetOutDataAnchor(0)->UnlinkAll();
  }
  return true;
}

bool TransdataReshapeTransposeFusionPass::RelinkEdges(ge::NodePtr &transdata_node1, ge::NodePtr &reformat_node1,
                                                      ge::NodePtr &reshape_node, ge::NodePtr &reformat_node2,
                                                      ge::NodePtr &transdata_node2, ge::NodePtr &transpose_node,
                                                      ge::ComputeGraph &graph) {
  ge::InDataAnchorPtr in_data_anchor = transdata_node1->GetInDataAnchor(0);
  if (in_data_anchor != nullptr && in_data_anchor->GetPeerOutAnchor() != nullptr) {
    if (ge::GraphUtils::AddEdge(in_data_anchor->GetPeerOutAnchor(), transpose_node->GetInDataAnchor(0))
          != ge::SUCCESS) {
      return false;
    }
    in_data_anchor->UnlinkAll();
  }

  ge::OutDataAnchorPtr out_data_anchor = transdata_node2->GetOutDataAnchor(0);
  if (out_data_anchor != nullptr && out_data_anchor->GetPeerInDataAnchors().size() > 0) {
    for (ge::InDataAnchorPtr peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      if (ge::GraphUtils::RemoveEdge(out_data_anchor, peer_in_data_anchor) != ge::SUCCESS) {
        return false;
      }
      if (ge::GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), peer_in_data_anchor) != ge::SUCCESS) {
        return false;
      }
    }
  }

  if (!UnLinkDataEdges(transdata_node1, reformat_node1, reshape_node, reformat_node2, transdata_node2, graph)) {
    return false;
  }

  if (!RelinkControlEdges(transdata_node1, transpose_node)) {
    OP_LOGE(transdata_node1->GetName().c_str(), "Fail to relink control edge for first transdata node.");
    return false;
  }

  if (!RelinkControlEdges(reshape_node, transpose_node)) {
    OP_LOGE(reshape_node->GetName().c_str(), "Fail to relink control edge for reshape node.");
    return false;
  }

  if (!RelinkControlEdges(transdata_node2, transpose_node)) {
    OP_LOGE(transdata_node2->GetName().c_str(), "Fail to relink control edge for second transdata node.");
    return false;
  }

  return true;
}

bool TransdataReshapeTransposeFusionPass::RelinkControlEdges(ge::NodePtr &src_node, ge::NodePtr &dst_node) const {
  ge::InControlAnchorPtr src_in_ctrl_anchor = src_node->GetInControlAnchor();
  ge::InControlAnchorPtr dts_in_ctrl_anchor = dst_node->GetInControlAnchor();
  if (src_in_ctrl_anchor != nullptr && dts_in_ctrl_anchor != nullptr) {
    if (src_in_ctrl_anchor->GetPeerOutControlAnchors().size() > 0) {
      for (ge::OutControlAnchorPtr peer_out_ctrl_anchor : src_in_ctrl_anchor->GetPeerOutControlAnchors()) {
        if (ge::GraphUtils::RemoveEdge(peer_out_ctrl_anchor, src_in_ctrl_anchor) != ge::SUCCESS) {
          return false;
        }
        if (ge::GraphUtils::AddEdge(peer_out_ctrl_anchor, dts_in_ctrl_anchor) != ge::SUCCESS) {
          return false;
        }
      }
    }
  }

  ge::OutControlAnchorPtr src_out_ctrl_anchor = src_node->GetOutControlAnchor();
  ge::OutControlAnchorPtr dst_out_ctrl_anchor = dst_node->GetOutControlAnchor();
  if (src_out_ctrl_anchor != nullptr && dst_out_ctrl_anchor != nullptr) {
    if (src_out_ctrl_anchor->GetPeerInControlAnchors().size() > 0) {
      for (ge::InControlAnchorPtr peer_in_ctrl_anchor : src_out_ctrl_anchor->GetPeerInControlAnchors()) {
        if (ge::GraphUtils::RemoveEdge(src_out_ctrl_anchor, peer_in_ctrl_anchor) != ge::SUCCESS) {
          return false;
        }
        if (ge::GraphUtils::AddEdge(dst_out_ctrl_anchor, peer_in_ctrl_anchor) != ge::SUCCESS) {
          return false;
        }
      }
    }
  }
  return true;
}

REGISTER_PASS("TransdataZReshapeTransposeFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS,
              TransdataReshapeTransposeFusionPass);
}  // namespace fe
