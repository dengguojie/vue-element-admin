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
 * \file reduce_min_d_fusion_pass.cpp
 * \brief fusedbatchnormgrad fusion pass(min --> mind)
 */
#include "reduce_min_d_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char PATTERN_MIND[] = "mind";
static const char MIND[] = "ReduceMinD";
static const char MINDLAST[] = "ReduceMinD";
static const char KEEPDIMS[] = "keep_dims";

/*MinD*/
static const std::string AXIS = "axes";
vector<FusionPattern*> ReduceMinDFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ReduceMinDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_MIND, {MIND}).SetOutput(PATTERN_MIND);
  patterns.push_back(pattern);
  return patterns;
}

Status ReduceMinDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ReduceMinDFusionPass fusion begin");
  ge::NodePtr minDNode = GetNodeFromMapping(PATTERN_MIND, mapping);
  FUSION_PASS_CHECK(minDNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minDNode is null, fusion failed."),
                    return PARAM_INVALID);
  // validation
  ge::GeTensorDesc tensor_input = minDNode->GetOpDesc()->GetInputDesc(0);

  ge::DataType input_type = tensor_input.GetDataType();
  ge::Format input_format = minDNode->GetOpDesc()->GetInputDesc(0).GetFormat();
  if (input_type != DT_FLOAT) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "dtype is not float32,no need change");
    return NOT_CHANGED;
  }
  std::vector<int64_t> axis;
  if (!ge::AttrUtils::GetListInt(minDNode->GetOpDesc(), AXIS, axis)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "get attr axis failed");
    return NOT_CHANGED;
  }

  vector<int64_t> dim_info = tensor_input.GetShape().GetDims();

  int64_t dims_size = dim_info.size();
  FUSION_PASS_CHECK(((dims_size == 0) || (dims_size == 1)),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "minD shape dims in one %d,not change", dims_size),
                    return NOT_CHANGED);

  if (axis.size() == 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "min_d axis is only one,no need change");
    return NOT_CHANGED;
  }
  vector<int64_t>::iterator it;
  vector<int64_t>::iterator it1;
  int64_t value = -1;
  int64_t value1 = dim_info.size() - 1;

  it = find(axis.begin(), axis.end(), value);
  it1 = find(axis.begin(), axis.end(), value1);
  if ((it == axis.end()) && (it1 == axis.end())) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "axis is not the last, not match");
    return NOT_CHANGED;
  }
  // validation ends
  std::vector<int32_t> axis_xin;
  for (size_t i = 0; i < axis.size(); ++i) {
    if ((axis[i] != -1) && (axis[i] != value1)) {
      axis_xin.push_back(axis[i]);
    }
  }

  std::vector<int32_t> axis_last;
  axis_last.push_back(-1);

  // copy Opdesc
  // Calculate the last axis of fp32 of min_d operator
  std::shared_ptr<ge::OpDesc> minNewOpdesc = nullptr;
  minNewOpdesc = std::make_shared<ge::OpDesc>(minDNode->GetName(), MIND);
  FUSION_PASS_CHECK(minNewOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "minNewOpdesc is null,"
                            "fusion failed."),
                    return PARAM_INVALID);

  // Create a new min_d_last node description
  std::shared_ptr<ge::OpDesc> minLastOpdesc = nullptr;
  minLastOpdesc = std::make_shared<ge::OpDesc>(minDNode->GetName() + "_Tik", MINDLAST);
  FUSION_PASS_CHECK(minLastOpdesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "minLastOpdesc is null,"
                            "fusion failed."),
                    return PARAM_INVALID);

  // add input for minnew
  ge::GeTensorDesc input_tensor1_new = minDNode->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(minNewOpdesc->AddInputDesc(input_tensor1_new) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minNew add input failed."), return FAILED);

  // add input for minlast
  ge::GeTensorDesc input_tensor1 = minDNode->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(minLastOpdesc->AddInputDesc(input_tensor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minLast add input failed."), return FAILED);

  // add output for minnew
  bool keepdims;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(minDNode->GetOpDesc(), KEEPDIMS, keepdims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Get keepdims attr failed."), return FAILED);
  int32_t dimNum = tensor_input.GetShape().GetDimNum();
  if (axis_xin.empty()) {
    for (int64_t i = 0; i < (dims_size - 1); ++i) {
      axis_xin.push_back(i);
    }
  }

  for (size_t i = 0; i < axis_xin.size(); ++i) {
    if (axis_xin[i] < -dimNum || axis_xin[i] > (dimNum - 1)) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "the axis of reduce verify failed.");
      return false;
    }
    if (axis_xin[i] < 0) {
      axis_xin[i] = dimNum + axis_xin[i];
    }
  }

  std::vector<int64_t> oShapeVector;
  std::vector<int32_t>::iterator tmp;
  for (int32_t item = 0; item < dimNum; ++item) {
    tmp = std::find(axis_xin.begin(), axis_xin.end(), item);
    if (tmp != axis_xin.end()) {
      if (keepdims) {
        // If keepDims is true, current dimesion set to 1
        oShapeVector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oShapeVector.push_back(dim_info[item]);
    }
  }

  ge::GeTensorDesc out_tensor1_new = minDNode->GetOpDesc()->GetOutputDesc(0);

  out_tensor1_new.SetShape(ge::GeShape(oShapeVector));
  out_tensor1_new.SetDataType(input_type);
  out_tensor1_new.SetFormat(input_format);
  out_tensor1_new.SetOriginFormat(input_format);
  out_tensor1_new.SetOriginShape(ge::GeShape(oShapeVector));
  FUSION_PASS_CHECK(minNewOpdesc->AddOutputDesc(out_tensor1_new) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minNew add output failed."), return FAILED);
  minLastOpdesc->UpdateInputDesc(0, out_tensor1_new);
  // add output for minlast
  ge::GeTensorDesc out_tensor1 = minDNode->GetOpDesc()->GetOutputDesc(0);
  std::vector<int64_t> oShapeVector1;
  for (size_t item1 = 0; item1 < (oShapeVector.size() - 1); ++item1) {
    oShapeVector1.push_back(oShapeVector[item1]);
  }

  if (keepdims) {
    // If keepDims is true, current dimesion set to 1
    oShapeVector1.push_back(1);
  }

  out_tensor1.SetShape(ge::GeShape(oShapeVector1));
  out_tensor1.SetDataType(input_type);
  out_tensor1.SetFormat(input_format);
  out_tensor1.SetOriginFormat(input_format);
  out_tensor1.SetOriginShape(ge::GeShape(oShapeVector1));
  FUSION_PASS_CHECK(minLastOpdesc->AddOutputDesc(out_tensor1) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minLast add output failed."), return FAILED);

  ge::NodePtr minNewNode = graph.AddNode(minNewOpdesc);
  ge::NodePtr minLastNode = graph.AddNode(minLastOpdesc);
  newNodes.push_back(minNewNode);
  newNodes.push_back(minLastNode);

  // copy attr
  // float keepdims;
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(minNewNode->GetOpDesc(), KEEPDIMS, keepdims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[name=%s]: Set attr %s failed", minNewNode->GetName().c_str(),
                            KEEPDIMS),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(minLastNode->GetOpDesc(), KEEPDIMS, keepdims),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Op[name=%s]: Set attr %s failed", minLastNode->GetName().c_str(),
                            KEEPDIMS),
                    return FAILED);

  // set axis
  // Give a list of -1 to mindnew
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(minNewNode->GetOpDesc(), AXIS, axis_xin),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minNew Set axis attr failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(minLastNode->GetOpDesc(), AXIS, axis_last),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minLast Set axis attr failed"), return FAILED);
  // connect output edge for minnew
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(minNewNode->GetOutDataAnchor(0), minLastNode->GetInDataAnchor(0)) != SUCCESS,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);

  // copy output edge for minlast
  for (auto inDataAnchor : minDNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(minDNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minDNode Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(minLastNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minLastNode Add out data edge failed."), return FAILED);
  }

  if (minDNode->GetOutControlAnchor()) {
    for (auto inControlAnchor : minDNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(minDNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minDNode Remove out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(minLastNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minLastNode Add out control edge failed."), return FAILED);
    }
  }

  // connect input for minnew
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(minDNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            minNewNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "minNewNode Add edge between node %s. and node %s failed.",
                            minDNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            minNewNode->GetName().c_str()),
                    return FAILED);

  // set grad op type to BNInferGrad
  minNewNode->GetOpDesc()->SetType(MIND);
  minLastNode->GetOpDesc()->SetType(MINDLAST);

  FUSION_PASS_CHECK(graph.RemoveNode(minDNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove minDNode node failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ReduceMinDFusionPass fusion end");

  return SUCCESS;
}

REGISTER_PASS("ReduceMinDFusionPass", BUILT_IN_GRAPH_PASS, ReduceMinDFusionPass);
}  // namespace fe
