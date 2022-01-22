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
 * \file batch_matmul_fusion_pass.h
 * \brief matmul reshape fusion (batchmatmul->matmul)
 */
#include "batch_matmul_fusion_pass.h"

#include <vector>

#include "anchor_util.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
static const std::string PATTEN_MATMUL = "batmatmul";
static const string kTranspose = "Transpose";
static const string kTransposeD = "TransposeD";
static const int32_t kDimLimit = 2;
vector<FusionPattern*> BatchMatMulFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("BatchMatMulFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    CUBE_CALL_ERR_REPORT("BatchMatMulFusionPass", "new pattern error"), return patterns);
  pattern->AddOpDesc(PATTEN_MATMUL, {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"}).SetOutput(PATTEN_MATMUL);
  patterns.push_back(pattern);
  return patterns;
}

Status BatchMatMulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE, "Enter BatchMatMulFusionPass.");
  ge::NodePtr fused_node = GetNodeFromMapping(PATTEN_MATMUL, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr,
                    CUBE_CALL_ERR_REPORT("BatchMatMulFusionPass", "Fusion GetNode Error"),
                    return PARAM_INVALID);
  // Do Transpose + MatMul Fusion
  bool is_transpose_fusion = CheckAndDoTransposeFusion(graph, fused_node);

  // Do BatchMatMul -> MatMul Fusion
  if (CheckIsNeedFusion(fused_node)) {
    ge::NodePtr matmul_node = nullptr;
    auto ret = CreateMatMulNode(graph, fused_node, matmul_node);
    FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT("MatmulFusionPass", "CreateMatMulNode FAIL"), return FAILED);
    ret = AddEdgeForMatMulNode(fused_node, matmul_node);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      CUBE_INNER_ERR_REPORT("MatmulFusionPass", "AddEdgeForMatMulNode FAIL"), return FAILED);
    ret = RemoveFusedNode(graph, fused_node);
    FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT("MatmulFusionPass", "RemoveFusedNode FAIL"), return FAILED);
    fusionNodes.push_back(matmul_node);
  } else if (!is_transpose_fusion) {
    return NOT_CHANGED;
  }

  OP_LOGD(FUSED_OP_TYPE, "BatchMatMulFusionPass success.");
  return SUCCESS;
}

bool BatchMatMulFusionPass::CheckIsNeedFusion(const ge::NodePtr& fused_node) const {
  if (fused_node->GetType() != "BatchMatMul" && fused_node->GetType() != "BatchMatMulV2") {
    OP_LOGD(FUSED_OP_TYPE, "BatchMatMul type is %s, skip BatchMatMul fusion.", fused_node->GetType().c_str());
    return false;
  }
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc1 = op.GetInputDesc(0);
  auto input_desc2 = op.GetInputDesc(1);
  int32_t dim_num1 = input_desc1.GetShape().GetDimNum();
  int32_t dim_num2 = input_desc2.GetShape().GetDimNum();
  if (dim_num1 > kDimLimit || dim_num2 > kDimLimit) {
    OP_LOGI("BatchMatMulFusionPass", "not need to BatchMatMulFusionPass");
    return false;
  }
  return true;
}

bool BatchMatMulFusionPass::CheckAndDoTransposeFusion(ge::ComputeGraph &graph, const ge::NodePtr &fused_node) const {
  ge::DataType data_dtype = fused_node->GetOpDesc()->GetInputDesc(1).GetDataType();
  if (data_dtype != ge::DT_FLOAT16 && data_dtype != ge::DT_FLOAT) {
    OP_LOGD(FUSED_OP_TYPE, "Transpose MatMul Fusion only support float16 and float32.");
    return false;
  }
  bool is_transpose_fusion = false;
  OP_LOGD(FUSED_OP_TYPE, "MatMul type is %s.", fused_node->GetType().c_str());
  string attr_name = "adj_x";
  if (fused_node->GetType() == "MatMul" or fused_node->GetType() == "MatMulV2") {
    attr_name = "transpose_x";
  }
  // due left transpose
  auto peer0_anchor = GetPeerOutAnchorWithInDataAnchor(fused_node, 0);
  if (peer0_anchor != nullptr) {
    OP_LOGD(FUSED_OP_TYPE, "Start check Transpose left.");
    ge::NodePtr transpose0_node = peer0_anchor->GetOwnerNode();
    auto check_transpose0 = CheckTransposeFusion(transpose0_node);
    if (check_transpose0) {
      is_transpose_fusion = true;
      FUSION_PASS_CHECK(DoTransposeFusion(transpose0_node, fused_node, 0, attr_name) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE, "DoTransposeFusion left failed."), return FAILED);
      FUSION_PASS_CHECK(RemoveFusedNode(graph, transpose0_node) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE, "RemoveFusedNode left failed."), return FAILED);
    }
  }
  // due right transpose
  auto peer1_anchor = GetPeerOutAnchorWithInDataAnchor(fused_node, 1);
  if (peer1_anchor != nullptr) {
    OP_LOGD(FUSED_OP_TYPE, "Start check Transpose right.");
    ge::NodePtr transpose1_node = peer1_anchor->GetOwnerNode();
    auto check_transpose1 = CheckTransposeFusion(transpose1_node);
    if (check_transpose1) {
      is_transpose_fusion = true;
      FUSION_PASS_CHECK(DoTransposeFusion(transpose1_node, fused_node, 1, attr_name) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE, "DoTransposeFusion right failed."), return FAILED);
      FUSION_PASS_CHECK(RemoveFusedNode(graph, transpose1_node) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE, "RemoveFusedNode right failed."), return FAILED);
    }
  }
  return is_transpose_fusion;
}

// check transpose node
bool BatchMatMulFusionPass::CheckTransposeFusion(const ge::NodePtr &transpose_node) const {
  if (transpose_node == nullptr) {
    OP_LOGE(FUSED_OP_TYPE, "Transpose node is null.");
    return false;
  }
  auto transpose_type = transpose_node->GetType();
  if (transpose_type != kTransposeD && transpose_type != kTranspose) {
    OP_LOGD(FUSED_OP_TYPE, "Transpose type is %s.", transpose_type.c_str());
    return false;
  }
  if (transpose_node->GetOutDataNodes().size() != 1) {
    OP_LOGD(FUSED_OP_TYPE, "Transpose output is not 1.");
    return false;
  }
  vector<int32_t> perm_value;
  ge::OpDescPtr transpose_desc = transpose_node->GetOpDesc();
  OP_LOGD(FUSED_OP_TYPE, "Transpose type is %s.", transpose_node->GetType().c_str());
  if (transpose_node->GetType() == kTransposeD) {
    if (!ge::AttrUtils::GetListInt(transpose_desc, "perm", perm_value)) {
      OP_LOGD(FUSED_OP_TYPE, "Get attr perm failed.");
      return false;
    }
  } else {
    // check perm const
    ge::Tensor perm_const;
    ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(transpose_node);
    if (GRAPH_SUCCESS != op.GetInputConstData("perm", perm_const)) {
      OP_LOGD(FUSED_OP_TYPE, "Get perm const data failed.");
      return false;
    }
    auto perm_desc = transpose_desc->GetInputDesc(1);
    vector<int64_t> perm_shape = perm_desc.GetShape().GetDims();
    if (perm_shape.size() != 1) {
      OP_LOGD(FUSED_OP_TYPE, "The perm dim size must be 1, now is %d.", perm_shape.size());
      return false;
    }
    // get const data
    DataType perm_dtype = perm_desc.GetDataType();
    const uint8_t *perm_data = perm_const.GetData();
    if (perm_data == nullptr) {
      OP_LOGD(FUSED_OP_TYPE, "Get perm data failed.");
      return false;
    }
    size_t size;
    if (perm_dtype == ge::DT_INT32) {
      size = perm_const.GetSize() / sizeof(int32_t);
      for (size_t i = 0; i < size; ++i) {
        perm_value.push_back(*((int32_t *)perm_data + i));
      }
    } else if (perm_dtype == ge::DT_INT64) {
      size = perm_const.GetSize() / sizeof(int64_t);
      for (size_t i = 0; i < size; ++i) {
        perm_value.push_back(*((int64_t *)perm_data + i));
      }
    } else {
      OP_LOGD(FUSED_OP_TYPE, "Perm dtype is not int32 or int64.");
      return false;
    }
  }
  int perm_len = perm_value.size();
  vector<int64_t> input_shape = transpose_desc->GetInputDesc(0).GetShape().GetDims();
  if (perm_len != input_shape.size() or perm_len < kDimLimit) {
    OP_LOGD(FUSED_OP_TYPE,
            "perm value dim should be equal to input dim and must be >= 2, now perm value len is %d, "
            "input shape len is %d.",
            perm_len, input_shape.size());
    return false;
  }
  for (int i = 0; i < perm_len - kDimLimit; i++) {
    if (perm_value[i] != i) {
      OP_LOGD(FUSED_OP_TYPE, "batch dim is transposed.");
      return false;
    }
  }
  return perm_value[perm_len - 1] == perm_len - 2 && perm_value[perm_len - 2] == perm_len - 1;
}

Status BatchMatMulFusionPass::DoTransposeFusion(const ge::NodePtr& transpose_node, const ge::NodePtr& fused_node,
                                                int data_index, const string& attr_name) const {
  OP_LOGD(FUSED_OP_TYPE, "Start due transpose %d.", data_index);
  ge::OpDescPtr matmul_desc = fused_node->GetOpDesc();
  string real_attr_name = attr_name + std::to_string(data_index + 1);
  bool adj;
  if (!ge::AttrUtils::GetBool(matmul_desc, real_attr_name, adj)) {
    OP_LOGE(FUSED_OP_TYPE, "Get matmul attr %s failed.", real_attr_name.c_str());
    return FAILED;
  }
  if (!ge::AttrUtils::SetBool(matmul_desc, real_attr_name, !adj)) {
    OP_LOGE(FUSED_OP_TYPE, "Set matmul attr %s failed.", real_attr_name.c_str());
    return FAILED;
  }

  auto matmul_input_desc = matmul_desc->MutableInputDesc(data_index);
  vector<int64_t> x_shape = matmul_input_desc->GetShape().GetDims();
  vector<std::pair<int64_t, int64_t>> x_range;
  matmul_input_desc->GetShapeRange(x_range);
  int dims = x_shape.size();
  if (dims < kDimLimit) {
    OP_LOGE(FUSED_OP_TYPE, "MatMul input %d shape dim must be >= 2.", data_index);
    return false;
  }
  if (x_range.empty()) {
    OP_LOGD(FUSED_OP_TYPE, "MatMul input %d range is null.", data_index);
  } else if (x_range.size() != dims) {
    OP_LOGE(FUSED_OP_TYPE, "MatMul input %d shape dim and range dim are not equal.", data_index);
    return false;
  } else {
    swap(x_range[dims - 1], x_range[dims - 2]);
    matmul_input_desc->SetShapeRange(x_range);
    matmul_input_desc->SetOriginShapeRange(x_range);
  }
  swap(x_shape[dims - 1], x_shape[dims - 2]);
  matmul_input_desc->SetShape(ge::GeShape(x_shape));
  matmul_input_desc->SetOriginShape(ge::GeShape(x_shape));

  return LinkEdge(transpose_node, fused_node, data_index);
}

Status BatchMatMulFusionPass::LinkEdge(const ge::NodePtr &transpose_node, const ge::NodePtr &fused_node,
                                       int data_index) const {
  OP_LOGD(FUSED_OP_TYPE, "Find node[%s]->matmul[%s].", transpose_node->GetName().c_str(), fused_node->GetName().c_str());
  auto matmul_in_anchor = fused_node->GetInDataAnchor(data_index);
  auto ret = ge::GraphUtils::RemoveEdge(transpose_node->GetOutDataAnchor(0), matmul_in_anchor);
  if (ret != SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE, "RemoveEdge transpose-->matmul failed.");
    return FAILED;
  }
  auto peer_out_anchor = GetPeerOutAnchorWithInDataAnchor(transpose_node, 0);
  if (peer_out_anchor == nullptr) {
    OP_LOGE(FUSED_OP_TYPE, "Transpose peer out anchor is null.");
    return FAILED;
  }
  ret = ge::GraphUtils::RemoveEdge(peer_out_anchor, transpose_node->GetInDataAnchor(0));
  if (ret != SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE, "RemoveEdge peer-->transpose failed.");
    return FAILED;
  }
  ret = ge::GraphUtils::AddEdge(peer_out_anchor, matmul_in_anchor);
  if (ret != SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE, "AddEdge peer-->matmul failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status BatchMatMulFusionPass::CreateMatMulNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node,
                                               ge::NodePtr& new_node) const {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(fused_node->GetName() + "_MatMul", "MatMul")),
                          return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);

  auto input_desc = op.GetInputDesc(0);
  ge::GeShape input_shape(input_desc.GetShape().GetDims());
  ge::GeShape origin_input_shape(input_desc.GetOriginShape().GetDims());
  ge::Format data_format = input_desc.GetFormat();
  ge::DataType data_type = input_desc.GetDataType();
  auto ret = new_desc->AddInputDesc("x1", GeTensorDesc(input_shape, data_format, data_type));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "CreateMulNode AddInputDesc one fail."), return FAILED);
  auto new_input_desc1 = new_desc->GetInputDesc(0);
  new_input_desc1.SetOriginShape(origin_input_shape);
  new_input_desc1.SetOriginDataType(data_type);
  new_input_desc1.SetOriginFormat(input_desc.GetOriginFormat());
  FUSION_PASS_CHECK(new_desc->UpdateInputDesc(0, new_input_desc1) != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "CreateMulNode UpdateInputDesc zero fail."),
                    return FAILED);

  auto input_desc1 = op.GetInputDesc(1);
  ge::GeShape input_shape1(input_desc1.GetShape().GetDims());
  ge::GeShape origin_input_shape1(input_desc1.GetOriginShape().GetDims());
  ge::Format data_format1 = input_desc1.GetFormat();
  ge::DataType data_type1 = input_desc1.GetDataType();
  ret = new_desc->AddInputDesc("x2", GeTensorDesc(input_shape1, data_format1, data_type1));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "CreateMulNode AddinputDesc two fail."), return FAILED);
  auto new_input_desc2 = new_desc->GetInputDesc(1);
  new_input_desc2.SetOriginShape(origin_input_shape1);
  new_input_desc2.SetOriginDataType(data_type1);
  new_input_desc2.SetOriginFormat(input_desc1.GetOriginFormat());
  FUSION_PASS_CHECK(new_desc->UpdateInputDesc(1, new_input_desc2) != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "CreateMulNode UpdateInputDesc one fail."),
                    return FAILED);

  auto output_desc = op.GetOutputDesc(0);
  ge::GeShape output_shape(output_desc.GetShape().GetDims());
  ge::GeShape origin_output_shape(output_desc.GetOriginShape().GetDims());
  ge::Format output_format = output_desc.GetFormat();
  ge::DataType output_dtype = output_desc.GetDataType();
  ret = new_desc->AddOutputDesc("y", GeTensorDesc(output_shape, output_format, output_dtype));
  FUSION_PASS_CHECK(ret != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "CreateMulNode AddoutputDesc fail."), return FAILED);
  auto new_output_desc = new_desc->GetOutputDesc(0);
  new_output_desc.SetOriginShape(origin_output_shape);
  new_output_desc.SetOriginDataType(output_dtype);
  new_output_desc.SetOriginFormat(output_desc.GetOriginFormat());
  FUSION_PASS_CHECK(new_desc->UpdateOutputDesc(0, new_output_desc) != ge::GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "Update output desc fail."), return FAILED);
  new_node = graph.AddNode(new_desc);
  FUSION_PASS_CHECK(new_node == nullptr, CUBE_INNER_ERR_REPORT("MatmulFusionPass", "Failed to add matmul to graph"),
                    return FAILED);
  Operator new_op = ge::OpDescUtils::CreateOperatorFromNode(new_node);
  bool adj_x1 = false;
  bool adj_x2 = false;
  op.GetAttr("adj_x1", adj_x1);
  op.GetAttr("adj_x2", adj_x2);
  new_op.SetAttr("transpose_x1", adj_x1);
  new_op.SetAttr("transpose_x2", adj_x2);
  return SUCCESS;
}

Status BatchMatMulFusionPass::AddEdgeForMatMulNode(const ge::NodePtr &fused_node,
                                                   const ge::NodePtr &matmul_node) const {
  auto peer_out_anchor = GetPeerOutAnchorWithInDataAnchor(fused_node, 0);
  FUSION_PASS_CHECK(peer_out_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "Failed to get peer out anchor of input 0"),
                    return FAILED);
  auto ret = ge::GraphUtils::AddEdge(peer_out_anchor, matmul_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "AddEdg to MatMulNode fail"), return FAILED);

  auto peer_out_anchor1 = GetPeerOutAnchorWithInDataAnchor(fused_node, 1);
  FUSION_PASS_CHECK(peer_out_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "Failed to get peer out anchor of input 1"),
                    return FAILED);
  ret = ge::GraphUtils::AddEdge(peer_out_anchor1, matmul_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "AddEdg to MatMulNode fail"), return FAILED);

  auto out_data_anchor = fused_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(out_data_anchor == nullptr,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "GetOutDataAnchor fail"), return FAILED);
  auto peer_indata_anchors = out_data_anchor->GetPeerInDataAnchors();
  for (auto in_data_anchor : peer_indata_anchors) {
    ret = ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), in_data_anchor);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      CUBE_INNER_ERR_REPORT("MatmulFusionPass", "AddEdg to MatMulNode removeEdge fail"),
                      return FAILED);
    ret = ge::GraphUtils::AddEdge(matmul_node->GetOutDataAnchor(0), in_data_anchor);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      CUBE_INNER_ERR_REPORT("MatmulFusionPass", "AddEdg to MatMulNode addEdge fail"), return FAILED);
  }
  return SUCCESS;
}

Status BatchMatMulFusionPass::RemoveFusedNode(ge::ComputeGraph& graph, const ge::NodePtr& fused_node) const {
  for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
    if (in_anchor != nullptr) {
      in_anchor->UnlinkAll();
    }
  }

  for (auto out_anchor : fused_node->GetAllOutDataAnchors()) {
    if (out_anchor != nullptr) {
      out_anchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(fused_node) != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "RemoveFusedNode error"),
                    return FAILED);
  return SUCCESS;
}

REGISTER_PASS("BatchMatMulFusionPass", BUILT_IN_GRAPH_PASS, BatchMatMulFusionPass);
}  // namespace fe