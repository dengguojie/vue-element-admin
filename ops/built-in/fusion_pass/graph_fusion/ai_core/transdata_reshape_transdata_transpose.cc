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
 *        reformat
 *           |
 *           |                    input0
 *        reshape     ----------->    \
 *           |                       transpose
 *           |
 *        reformat
 *           |
 *           |
 *        transdata
 *
 */
#include "transdata_reshape_transdata_transpose.h"

#include <iostream>
#include <map>
#include <set>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "../../../op_proto/util/error_util.h"

namespace fe {
static const char PATTERN_TRANSPOSE[] = "FusedNodeTranspose";
static const char PATTERN_RESHAPE[] = "FusedNodeReshape";
static const char PATTERN_TRANSDATA1[] = "FusedNodeTransdata1";
static const char PATTERN_TRANSDATA2[] = "FusedNodeTransdata2";
static const char PATTERN_REFORMAT1[] = "FusedNodeReformat1";
static const char PATTERN_REFORMAT2[] = "FusedNodeReformat2";
constexpr int32_t LAST_SECOND_INDEX = 2;
constexpr int32_t DIM_INDEX_TWO = 2;
constexpr int32_t DIM_INDEX_THREE = 3;
constexpr int32_t DIM_INDEX_FOUR = 4;
constexpr int32_t TRANSOPSE_OUT_LEN = 5;
constexpr int32_t TRANSDATA_OUT_LEN = 4;
constexpr int32_t BLOCK_ALLIGN = 16;

vector<FusionPattern*> TransdataTransposeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TransdataReshapeTransdata");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_TRANSDATA1, {"TransData"})
      .AddOpDesc(PATTERN_TRANSDATA2, {"TransData"})
      .AddOpDesc(PATTERN_RESHAPE, {"Reshape"})
      .AddOpDesc(PATTERN_REFORMAT1, {"ReFormat"})
      .AddOpDesc(PATTERN_REFORMAT2, {"ReFormat"})
      .SetInputs(PATTERN_REFORMAT1, {PATTERN_TRANSDATA1})
      .SetInputs(PATTERN_RESHAPE, {PATTERN_REFORMAT1})
      .SetInputs(PATTERN_REFORMAT2, {PATTERN_RESHAPE})
      .SetInputs(PATTERN_TRANSDATA2, {PATTERN_REFORMAT2})
      .SetOutput(PATTERN_TRANSDATA2);
  patterns.push_back(pattern);
  return patterns;
}

Status GenerateTransposeNode(ge::ComputeGraph* graph, ge::GeTensorDesc& transdata_input_tensor, ge::DataType& dtype,
                             vector<int64_t>& permvalue, vector<int64_t>& transpose_out_shape,
                             ge::NodePtr* transpose_node, ge::NodePtr transdata_nodel1) {
  ge::OpDescPtr transpose_desc;
  FUSION_PASS_MAKE_SHARED(
      (transpose_desc = std::make_shared<ge::OpDesc>(transdata_nodel1->GetName() + "_transposeD", "TransposeD")),
      return FAILED);
  transdata_input_tensor.SetFormat(ge::FORMAT_ND);
  transpose_desc->AddInputDesc("x", transdata_input_tensor);
  ge::GeTensorDesc transpose_out_desc;
  transpose_out_desc.SetDataType(dtype);
  transpose_out_desc.SetShape(ge::GeShape(transpose_out_shape));
  transpose_out_desc.SetFormat(ge::FORMAT_FRACTAL_NZ);
  transpose_desc->AddOutputDesc("y", transpose_out_desc);
  ge::AttrUtils::SetListInt(transpose_desc, "perm", permvalue);
  *transpose_node = graph->AddNode(transpose_desc);
  return SUCCESS;
}

Status TransdataTransposeFusionPass::UnlinkFusedNodeEdge(ge::NodePtr transdata1_node, ge::NodePtr reshape_node,
                                                         ge::NodePtr transdata2_node, ge::NodePtr transpose_node) {
  if (transdata1_node->GetInControlAnchor() != nullptr) {
    if (!transdata1_node->GetInControlAnchor()->GetPeerOutControlAnchors().empty() &&
        transpose_node->GetInControlAnchor() != nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The PeerOutControlAnchors of fused node[%s] input control anchor is empty.",
              transdata1_node->GetName().c_str());
      for (OutControlAnchorPtr& outCtrlAnchorPtr : transdata1_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(outCtrlAnchorPtr, transpose_node->GetInControlAnchor()),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add input control edge for fusion node:%s.",
                                           transpose_node->GetName().c_str()),
            return FAILED);
      }
    }
    transdata1_node->GetInControlAnchor()->UnlinkAll();
  }
  // connect out control anchopr
  // transdata2 output -> transpose output
  if (transdata2_node->GetOutControlAnchor() != nullptr) {
    if (!transdata2_node->GetOutControlAnchor()->GetPeerInControlAnchors().empty() &&
        transpose_node->GetOutControlAnchor() != nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The PeeroutControlAnchors of fused node[%s] input control anchor is empty.",
              transdata2_node->GetName().c_str());
      for (InControlAnchorPtr& inCtrlAnchorPtr : transdata2_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(
            SUCCESS != ge::GraphUtils::AddEdge(inCtrlAnchorPtr, transpose_node->GetOutControlAnchor()),
            VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Fail to add output control edge for fusion node:%s.",
                                           transpose_node->GetName().c_str()),
            return FAILED);
      }
    }
    transdata2_node->GetOutControlAnchor()->UnlinkAll();
  }
  // transdata1 input equal 1
  if (transdata1_node->GetAllInDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:[%s] only should have one input, but actually have %d.",
            transdata1_node->GetName().c_str(), transdata1_node->GetAllInDataAnchors().size());
    return FAILED;
  }
  // transdata1 input->transpose input
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(transdata1_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                                       transpose_node->GetInDataAnchor(0)),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add transdata node in data edge failed."),
                    return FAILED);
  for (auto& inAnchorPtr : transdata1_node->GetAllInDataAnchors()) {
    if (inAnchorPtr != nullptr) {
      inAnchorPtr->UnlinkAll();
    }
  }
  // transdata2 output ->transpose output
  auto transpose_out = transpose_node->GetOutDataAnchor(0);
  auto transdata2_out = transdata2_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  if (transdata2_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    for (auto inDataAnchor : transdata2_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inDataAnchor->UnlinkAll();
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(transpose_out, inDataAnchor) != ge::GRAPH_SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add transpose to transdata2  error"),
                        return FAILED);
    }
    OP_LOGD(FUSED_OP_TYPE, "Add edge from fused node:%s's 1st index to fusion node:%s's 1st index.",
            transpose_node->GetName().c_str(), transdata2_node->GetName().c_str());
  }
  return SUCCESS;
}

Status TransdataTransposeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                            vector<ge::NodePtr>& fusionNodes) {
  // get transdata1 node info
  ge::NodePtr transdata_node1 = GetNodeFromMapping(PATTERN_TRANSDATA1, mapping);
  FUSION_PASS_CHECK(transdata_node1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransdataNode1 is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr transdata_desc1 = transdata_node1->GetOpDesc();
  FUSION_PASS_CHECK(transdata_desc1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransdataNode1 is null, fusion failed."),
                    return PARAM_INVALID);

  // get transdata2 node info
  ge::NodePtr transdata_node2 = GetNodeFromMapping(PATTERN_TRANSDATA2, mapping);
  FUSION_PASS_CHECK(transdata_node2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransdataNode2 is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr transdata_desc2 = transdata_node2->GetOpDesc();
  FUSION_PASS_CHECK(transdata_desc2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "TransdataNode2 is null, fusion failed."),
                    return PARAM_INVALID);

  // get reformat1 node info
  ge::NodePtr reformat_node1 = GetNodeFromMapping(PATTERN_REFORMAT1, mapping);
  FUSION_PASS_CHECK(reformat_node1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReformatNode1 is null, fusion failed."),
                    return PARAM_INVALID);

  // get reformat2 node info
  ge::NodePtr reformat_node2 = GetNodeFromMapping(PATTERN_REFORMAT2, mapping);
  FUSION_PASS_CHECK(reformat_node2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ReformatNode2 is null, fusion failed."),
                    return PARAM_INVALID);

  // get reshape node info
  ge::NodePtr reshape_node = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  FUSION_PASS_CHECK(reshape_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape_node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr reshape_desc = reshape_node->GetOpDesc();
  FUSION_PASS_CHECK(reshape_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reshape_desc is null, fusion failed."),
                    return PARAM_INVALID);

  // input of transdata1 must be NZ
  ge::GeTensorDesc transdatainputtensor1 = transdata_desc1->GetInputDesc(0);
  ge::GeTensorDesc transdataoutputtensor1 = transdata_desc1->GetOutputDesc(0);
  if (!(transdataoutputtensor1.GetFormat() == ge::FORMAT_ND &&
        ge::GetPrimaryFormat(transdatainputtensor1.GetFormat()) == ge::FORMAT_FRACTAL_NZ)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "node[TransData1]'s input format is not FRACTAL_NZ , not support fusion,"
                                   "TransDataConfusionTransposeFusionPass fusion end");
    return NOT_CHANGED;
  }

  // input of transdata2 must be ND
  ge::GeTensorDesc transdatainputtensor2 = transdata_desc2->GetInputDesc(0);
  ge::GeTensorDesc transdataoutputtensor2 = transdata_desc2->GetOutputDesc(0);
  if (!(transdataoutputtensor2.GetFormat() == ge::FORMAT_FRACTAL_NZ &&
        ge::GetPrimaryFormat(transdatainputtensor2.GetFormat()) == ge::FORMAT_ND)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                   "node[TransData2]'s input format is not FRACTAL_ND , not support fusion,"
                                   "TransDataConfusionTransposeFusionPass fusion end");
    return NOT_CHANGED;
  }
  // perm vlaue
  // (a,c/16,b/16,16,16)->(c/16,a,b/16,16,16)=[1,0,2,3,4]
  // (a,c/16,b/16,16,16)->(b/16,c/16,a,16,16)=[2,1,0,3,4]
  // (ab,d/16,c/16,16,16)->(ab,d/16,c/16,16,16)=[0,1,2,3,4]
  // transdata1 info
  vector<int64_t> transdata1_origin_shape = transdata_desc1->GetInputDesc(0).GetOriginShape().GetDims();
  vector<int64_t> transdata_diminfo1_shape = transdata_desc1->GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> transdata_diminfo1_out_shape = transdata_desc1->GetOutputDesc(0).GetShape().GetDims();
  DataType datatype1 = transdatainputtensor1.GetDataType();
  DataType datatype2 = transdataoutputtensor2.GetDataType();

  // reshape output dim info
  vector<int64_t> reshapediminfo = reshape_desc->GetOutputDesc(0).GetShape().GetDims();

  // transdata1 input length must be 5
  if (transdata_diminfo1_shape.size() != 5) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], cannot be applied to fusion pass.",
            transdata_node1->GetName().c_str(), transdata_diminfo1_shape.size());
    return NOT_CHANGED;
  }
  if (transdata1_origin_shape[transdata1_origin_shape.size() - 1] % BLOCK_ALLIGN != 0 ||
      transdata1_origin_shape[transdata1_origin_shape.size() - 2] % BLOCK_ALLIGN != 0 ||
      reshapediminfo[reshapediminfo.size() - 1] % BLOCK_ALLIGN != 0 ||
      reshapediminfo[reshapediminfo.size() - LAST_SECOND_INDEX] % BLOCK_ALLIGN != 0) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "last two dimension should be divisible by 16, but actually is [%ld] and [%ld].",
            transdata1_origin_shape[transdata1_origin_shape.size() - LAST_SECOND_INDEX],
            transdata1_origin_shape[transdata1_origin_shape.size() - 1]);
    return NOT_CHANGED;
  }
  vector<int64_t> permValue_cond1({1, 0, 2, 3, 4});
  vector<int64_t> trans_after_reshape_cond1(
      {transdata_diminfo1_shape[0] * transdata_diminfo1_shape[DIM_INDEX_TWO] * 16, transdata_diminfo1_shape[1] * 16});
  vector<int64_t> transpose_out_shape1(
      {transdata_diminfo1_shape[1], transdata_diminfo1_shape[0] * transdata_diminfo1_shape[DIM_INDEX_TWO], 16, 16});
  vector<int64_t> permValue_cond2({2, 1, 0, 3, 4});
  vector<int64_t> trans_after_reshape_cond2(
      {transdata_diminfo1_shape[0], transdata_diminfo1_shape[1] * transdata_diminfo1_shape[DIM_INDEX_TWO] * 16 * 16});
  vector<int64_t> transpose_out_shape2({transdata_diminfo1_shape[1] * transdata_diminfo1_shape[DIM_INDEX_TWO] * BLOCK_ALLIGN,
                                        transdata_diminfo1_shape[0] / BLOCK_ALLIGN, 16, 16});
  vector<int64_t> permValue_cond3({0, 1, 2, 3, 4});
  vector<int64_t> transpose_out_shape3(TRANSOPSE_OUT_LEN);
  if (transdata_diminfo1_out_shape.size() == TRANSDATA_OUT_LEN) {
    transpose_out_shape3[0] = transdata_diminfo1_out_shape[0];
    transpose_out_shape3[1] = transdata_diminfo1_out_shape[DIM_INDEX_THREE] / BLOCK_ALLIGN;
    transpose_out_shape3[DIM_INDEX_TWO] =
        transdata_diminfo1_out_shape[1] * transdata_diminfo1_out_shape[DIM_INDEX_TWO] / BLOCK_ALLIGN;
    transpose_out_shape3[DIM_INDEX_THREE] = 16;
    transpose_out_shape3[DIM_INDEX_FOUR] = 16;
  }
  std::set<DataType> supported_perm_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32,
                                              ge::DT_INT16, ge::DT_UINT16,  ge::DT_UINT32};
  if (supported_perm_dtypes.count(datatype1) == 0) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Dtype of input is not supported in transpose.");
    return NOT_CHANGED;
  }
  ge::NodePtr transpose_node = nullptr;
  if (reshapediminfo == trans_after_reshape_cond1) {
    FUSION_PASS_CHECK(GenerateTransposeNode(&graph, transdatainputtensor1, datatype2, permValue_cond1,
                                            transpose_out_shape1, &transpose_node, transdata_node1) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE, "fail to generate transpose node B"), return FAILED);
    FUSION_PASS_CHECK(UnlinkFusedNodeEdge(transdata_node1, reshape_node, transdata_node2, transpose_node) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE, "fail to unlink edge"), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(transdata_node1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transdata1 node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(transdata_node2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transdata2 node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reshape_node) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reformat_node1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reformat_node2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);

    fusionNodes.push_back(transpose_node);
    OP_LOGD("TransdataTransposeFusionPass success.");
    return SUCCESS;
  } else if (reshapediminfo == trans_after_reshape_cond2) {
    FUSION_PASS_CHECK(GenerateTransposeNode(&graph, transdatainputtensor1, datatype2, permValue_cond2,
                                            transpose_out_shape2, &transpose_node, transdata_node1) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate transpose node B"),
                      return FAILED);
    FUSION_PASS_CHECK(UnlinkFusedNodeEdge(transdata_node1, reshape_node, transdata_node2, transpose_node) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE, "fail to unlink edge"), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(transdata_node1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transdata1 node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(transdata_node2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transdata2 node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reshape_node) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reformat_node1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reformat_node2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    OP_LOGD("TransdataTransposeFusionPass success.");
    fusionNodes.push_back(transpose_node);
    return SUCCESS;
  } else {
    FUSION_PASS_CHECK(GenerateTransposeNode(&graph, transdatainputtensor1, datatype2, permValue_cond3,
                                            transpose_out_shape3, &transpose_node, transdata_node1) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate transpose node B"),
                      return FAILED);
    FUSION_PASS_CHECK(UnlinkFusedNodeEdge(transdata_node1, reshape_node, transdata_node2, transpose_node) != SUCCESS,
                      ge::CommonRuntimeErrLog(FUSED_OP_TYPE, "fail to unlink edge"), return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(transdata_node1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transdata1 node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(transdata_node2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove transdata2 node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reshape_node) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reformat_node1) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    FUSION_PASS_CHECK(graph.RemoveNode(reformat_node2) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove reshape node failed."),
                      return FAILED);
    fusionNodes.push_back(transpose_node);
    OP_LOGD("TransdataTransposeFusionPass success.");
    return SUCCESS;
  }
  return SUCCESS;
}

REGISTER_PASS("TransdataTransposeFusionPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, TransdataTransposeFusionPass);
}  // namespace fe
