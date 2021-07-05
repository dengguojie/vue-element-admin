/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
static const int32_t DIM_LIMIT = 2;
vector<FusionPattern*> BatchMatmulFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("batchMatmulFusion");
  FUSION_PASS_CHECK(pattern == nullptr,
                    CUBE_CALL_ERR_REPORT("BatchMatmulFusionPass", "new pattern error"), return patterns);
  pattern->AddOpDesc(PATTEN_MATMUL, {"BatchMatMul"}).SetOutput(PATTEN_MATMUL);
  patterns.push_back(pattern);
  return patterns;
}

Status BatchMatmulFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTEN_MATMUL, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr,
                    CUBE_CALL_ERR_REPORT("BatchMatmulFusionPass", "Fusion GetNode Error"),
                    return PARAM_INVALID);
  if (!CheckIsNeedFusion(fused_node)) {
    return SUCCESS;
  }

  ge::NodePtr matmul_node = nullptr;
  auto ret = CreateMatMulNode(graph, fused_node, matmul_node);
  FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT("MatmulFusionPass", "CreateMatMulNode FAIL"), return FAILED);
  ret = AddEdgeForMatMulNode(fused_node, matmul_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    CUBE_INNER_ERR_REPORT("MatmulFusionPass", "AddEdgeForMatMulNode FAIL"), return FAILED);
  ret = RemoveFusedNode(graph, fused_node);
  FUSION_PASS_CHECK(ret != SUCCESS, CUBE_INNER_ERR_REPORT("MatmulFusionPass", "RemoveFusedNode FAIL"), return FAILED);
  fusionNodes.push_back(matmul_node);
  return SUCCESS;
}

bool BatchMatmulFusionPass::CheckIsNeedFusion(ge::NodePtr& fused_node) const {
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc1 = op.GetInputDesc(0);
  auto input_desc2 = op.GetInputDesc(1);
  int32_t dim_num1 = input_desc1.GetShape().GetDimNum();
  int32_t dim_num2 = input_desc2.GetShape().GetDimNum();
  if (dim_num1 > DIM_LIMIT || dim_num2 > DIM_LIMIT) {
    OP_LOGI("BatchMatmulFusionPass", "not need to BatchMatmulFusionPass");
    return false;
  }
  return true;
}

Status BatchMatmulFusionPass::CreateMatMulNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node,
                                               ge::NodePtr& new_node) {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(fused_node->GetName() + "_MatMul", "MatMul")),
                          return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);

  auto input_desc = op.GetInputDesc(0);
  ge::GeShape input_shape(input_desc.GetShape().GetDims());
  ge::GeShape origin_input_shape(input_desc.GetOriginShape().GetDims());
  ge::Format data_format = input_desc.GetFormat();
  ge::DataType data_type = input_desc.GetDataType();
  auto ret = new_desc->AddInputDesc(GeTensorDesc(input_shape, data_format, data_type));
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
  ret = new_desc->AddInputDesc(GeTensorDesc(input_shape1, data_format1, data_type1));
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
  ret = new_desc->AddOutputDesc(GeTensorDesc(output_shape, output_format, output_dtype));
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

Status BatchMatmulFusionPass::AddEdgeForMatMulNode(ge::NodePtr& fused_node, ge::NodePtr& matmul_node) {
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

Status BatchMatmulFusionPass::RemoveFusedNode(ge::ComputeGraph& graph, ge::NodePtr& fused_node) {
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

REGISTER_PASS("BatchMatmulFusionPass", BUILT_IN_GRAPH_PASS, BatchMatmulFusionPass);
}  // namespace fe