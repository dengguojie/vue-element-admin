/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "multinomial_fusion_pass.h"

#include <memory>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
const std::string MultinomialPass::PATTERN_FUSEDNODE = "MultinomialFuss";
const int MultinomialPass::DATA_TYPE_INT64_ONNX = 7;
vector<FusionPattern *> MultinomialPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      (new (std::nothrow) FusionPattern("MultinomialPass"));
  FUSION_PASS_CHECK(pattern == nullptr,
                    OP_LOGE("MultinomialPass", "new pattern error"),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {"MultinomialFuss"})
      .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status MultinomialPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                               vector<ge::NodePtr> &fusion_nodes) {
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr,
                    OP_LOGE("MultinomialPass", "Fusion GetNode Error"),
                    return PARAM_INVALID);
  ge::NodePtr multinomial_node = nullptr;
  auto ret = CreateMultinomialNode(graph, fused_node, multinomial_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("MultinomialPass", "CreateMultinomialNode fail"),
                    return FAILED);
  fusion_nodes.push_back(multinomial_node);
  ge::NodePtr const_node = nullptr;
  ret = CreateConstNode(graph, fused_node, const_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("MultinomialPass", "CreateConstNode fail"),
                    return FAILED);
  fusion_nodes.push_back(const_node);
  auto in_data_anchor = fused_node->GetInDataAnchor(0);
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  ret = ge::GraphUtils::AddEdge(peer_out_anchor,
                                multinomial_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("MultinomialPass", "AddEdge multinomial fail"),
                    return FAILED);
  ret = ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0),
                                multinomial_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("MultinomialPass", "AddEdge constnode fail"),
                    return FAILED);
  ret = AddEdgeForOut(fused_node, multinomial_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("MultinomialPass", "AddEdgeForOut fail"),
                    return FAILED);
  ret = RemoveFusedNode(graph, fused_node);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    OP_LOGE("MultinomialPass", "RemoveFusedNode fail"),
                    return FAILED);
  return SUCCESS;
}

Status MultinomialPass::CreateMultinomialNode(ge::ComputeGraph &graph,
                                              ge::NodePtr &fused_node,
                                              ge::NodePtr &mul_node) const {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(
                               "Multinomial_Fuss", "Multinomial")),
                          return INTERNAL_ERROR);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetInputDesc(0);
  ge::GeShape shape(input_desc.GetShape().GetDims());
  ge::DataType data_type = input_desc.GetDataType();
  ge::Format format = input_desc.GetFormat();
  new_desc->AddInputDesc(GeTensorDesc(shape, format, data_type));
  new_desc->AddInputDesc(GeTensorDesc(GeShape(), format, DT_INT32));
  auto output_desc = op.GetOutputDesc(0);
  ge::GeShape output_shape(output_desc.GetShape().GetDims());
  ge::DataType output_type = output_desc.GetDataType();
  ge::Format output_format = output_desc.GetFormat();
  new_desc->AddOutputDesc(
      GeTensorDesc(output_shape, output_format, output_type));
  mul_node = graph.AddNode(new_desc);
  Operator op_mul = ge::OpDescUtils::CreateOperatorFromNode(mul_node);
  int dtype = 0;
  float seed = 0;
  op.GetAttr("dtype", dtype);
  op.GetAttr("seed", seed);
  ge::DataType dtype_om = DT_INT32;
  if (dtype == DATA_TYPE_INT64_ONNX) {
    dtype_om = DT_INT64;
  }
  op_mul.SetAttr("dtype", dtype_om);
  int seed_om = static_cast<int>(seed);
  op_mul.SetAttr("seed", seed_om);
  op_mul.SetAttr("seed2", 0);
  return SUCCESS;
}

Status MultinomialPass::CreateConstNode(ge::ComputeGraph &graph,
                                        ge::NodePtr &fused_node,
                                        ge::NodePtr &const_node) const {
  ge::OpDescPtr new_desc = nullptr;
  FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>(
                               "Multinomial_Const_Fuss", "Const")),
                          return INTERNAL_ERROR);
  ge::GeTensorDesc value_desc;
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  auto input_desc = op.GetInputDesc(0);
  ge::Format format = input_desc.GetFormat();
  value_desc.SetFormat(format);
  value_desc.SetDataType(DT_INT32);
  int sample_size = 1;
  op.GetAttr("sample_size", sample_size);
  std::vector<int> val(1, sample_size);
  ge::GeTensorPtr val_tensor_ptr = std::make_shared<ge::GeTensor>(
      value_desc, reinterpret_cast<uint8_t *>(val.data()), sizeof(int));
  AttrUtils::SetTensor(new_desc, ATTR_NAME_WEIGHTS, val_tensor_ptr);
  new_desc->AddOutputDesc(value_desc);
  const_node = graph.AddNode(new_desc);
  return SUCCESS;
}

Status MultinomialPass::AddEdgeForOut(ge::NodePtr &fused_node,
                                      ge::NodePtr &multinomial_node) const {
  Status ret = SUCCESS;
  auto out_data_anchor = fused_node->GetOutDataAnchor(0);
  auto peer_indata_anchors = out_data_anchor->GetPeerInDataAnchors();
  for (auto in_data_anchor : peer_indata_anchors) {
    ret = ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0),
                                     in_data_anchor);
    FUSION_PASS_CHECK(
        ret != SUCCESS,
        OP_LOGE("MultinomialPass", "AddEdgeToMulForOut removeEdge fail"),
        return FAILED);
    ret = ge::GraphUtils::AddEdge(multinomial_node->GetOutDataAnchor(0),
                                  in_data_anchor);
    FUSION_PASS_CHECK(
        ret != SUCCESS,
        OP_LOGE("MultinomialPass", "AddEdgeToMulForOut addEdge fail"),
        return FAILED);
  }
  return SUCCESS;
}

Status MultinomialPass::RemoveFusedNode(ge::ComputeGraph &graph,
                                        ge::NodePtr &fused_node) const {
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
                    OP_LOGE("MultinomialPass", "RemoveFusedNode error"),
                    return FAILED);
  return SUCCESS;
}
REGISTER_PASS("MultinomialPass", BUILT_IN_GRAPH_PASS, MultinomialPass);
}  //  namespace fe
