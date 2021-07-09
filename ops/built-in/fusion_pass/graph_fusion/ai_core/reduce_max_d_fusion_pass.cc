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
 * \file reduce_max_d_fusion_pass.cpp
 * \brief fusedbatchnormgrad fusion pass(max --> max_d)
 */
#include "reduce_max_d_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <algorithm>
#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {
static const char PATTERN_MAXD[] = "maxd";
static const char MAXD[] = "ReduceMaxD";
static const char MAXDLAST[] = "ReduceMaxD";
static const char KEEPDIMS[] = "keep_dims";
static const char AXIS[] = "axes";
vector<FusionPattern*> ReduceMaxDFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ReduceMaxDFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_MAXD, {MAXD}).SetOutput(PATTERN_MAXD);
  patterns.push_back(pattern);
  return patterns;
}

Status ReduceMaxDFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ReduceMaxDFusionPass fusion begin");
  ge::NodePtr max_node = GetNodeFromMapping(PATTERN_MAXD, mapping);
  FUSION_PASS_CHECK(max_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "max_node is null, fusion failed."),
                    return PARAM_INVALID);
  // validation
  ge::GeTensorDesc tensor_input = max_node->GetOpDesc()->GetInputDesc(0);

  // get shape
  ge::GeShape input_shape = tensor_input.GetShape();
  vector<int64_t> dimInfo = input_shape.GetDims();
  if (dimInfo.size() != 0) {
    for (size_t i = 0; i < dimInfo.size(); ++i) {
      if (dimInfo[i] < 0) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "dynamic shape is %d", dimInfo[i]);
        return NOT_CHANGED;
      }
    }
  }

  ge::DataType input_type = tensor_input.GetDataType();
  ge::Format input_format = max_node->GetOpDesc()->GetInputDesc(0).GetFormat();
  if ((input_type != DT_FLOAT) && (input_type != DT_INT32)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "dtype is not in (float32,int32),no need change");
    return NOT_CHANGED;
  }
  std::vector<int64_t> axis;
  if (!ge::AttrUtils::GetListInt(max_node->GetOpDesc(), AXIS, axis)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "get attr axis failed");
    return NOT_CHANGED;
  }

  int64_t dims_size = dimInfo.size();
  FUSION_PASS_CHECK(((dims_size == 0) || (dims_size == 1)),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "minD shape dims in one %d,not change", dims_size),
                    return NOT_CHANGED);

  if (axis.size() == 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "minD axis is one,not change");
    return NOT_CHANGED;
  }
  vector<int64_t>::iterator it;
  vector<int64_t>::iterator it1;
  int64_t value = -1;
  int64_t value1 = dimInfo.size() - 1;
  it = find(axis.begin(), axis.end(), value);
  it1 = find(axis.begin(), axis.end(), value1);
  if ((it == axis.end()) && (it1 == axis.end())) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "axis not last,so not match");
    return NOT_CHANGED;
  }
  std::vector<int32_t> axis_xin;
  for (size_t i = 0; i < axis.size(); ++i) {
    if ((axis[i] != -1) && (axis[i] != value1)) {
      axis_xin.push_back(axis[i]);
    }
  }

  std::vector<int32_t> axis_last;
  axis_last.push_back(-1);

  // Create a new minD node description
  std::shared_ptr<ge::OpDesc> minNewOpdesc = nullptr;
  minNewOpdesc = std::make_shared<ge::OpDesc>(max_node->GetName(), MAXD);
  FUSION_PASS_CHECK(minNewOpdesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "minNewOpdesc is null,"
                            "fusion failed."),
                    return PARAM_INVALID);

  // Create a new minlast node description
  std::shared_ptr<ge::OpDesc> minLastOpdesc = nullptr;
  minLastOpdesc = std::make_shared<ge::OpDesc>(max_node->GetName() + "_Tik", MAXDLAST);
  FUSION_PASS_CHECK(minLastOpdesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "minLastOpdesc is null,"
                            "fusion failed."),
                    return PARAM_INVALID);

  // add input for minnew
  ge::GeTensorDesc input_tensor1_new = max_node->GetOpDesc()->GetInputDesc(0);
  FUSION_PASS_CHECK(minNewOpdesc->AddInputDesc(input_tensor1_new) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "minNew add input failed."), return FAILED);

  // add input for minlast
  ge::GeTensorDesc input_tensor1 = max_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(minLastOpdesc->AddInputDesc(input_tensor1) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "minLast add input failed."), return FAILED);

  // add output for minnew
  // The output tensor description needs to be calculated
  bool keepdims;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(max_node->GetOpDesc(), KEEPDIMS, keepdims),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Get keepdims attr failed."), return FAILED);
  int32_t dimNum = tensor_input.GetShape().GetDimNum();
  if (axis_xin.empty()) {
    for (int64_t i = 0; i < (dims_size - 1); ++i) {
      axis_xin.push_back(i);
    }
  }

  for (size_t i = 0; i < axis_xin.size(); ++i) {
    if (axis_xin[i] < -dimNum || axis_xin[i] > (dimNum - 1)) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "the axis of reduce verify failed.");
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
      // item in axis
      if (keepdims) {
        // If keepDims is true, current dimesion set to 1
        oShapeVector.push_back(1);
      }
    } else {
      // item is not in ConstValueAxis
      oShapeVector.push_back(dimInfo[item]);
    }
  }

  ge::GeTensorDesc out_tensor1_new = max_node->GetOpDesc()->GetOutputDesc(0);

  out_tensor1_new.SetShape(ge::GeShape(oShapeVector));
  out_tensor1_new.SetDataType(input_type);
  out_tensor1_new.SetFormat(input_format);
  out_tensor1_new.SetOriginFormat(input_format);
  out_tensor1_new.SetOriginShape(ge::GeShape(oShapeVector));
  FUSION_PASS_CHECK(minNewOpdesc->AddOutputDesc(out_tensor1_new) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "minNew add output failed."), return FAILED);

  // add output for minlast
  minLastOpdesc->UpdateInputDesc(0, out_tensor1_new);
  ge::GeTensorDesc out_tensor1 = max_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<int64_t> oShapeVector1;
  for (size_t item1 = 0; item1 < (oShapeVector.size() - 1); ++item1) {
    oShapeVector1.push_back(oShapeVector[item1]);
  }
  // item in axis
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
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "minLast add output failed."), return FAILED);

  ge::NodePtr minNewNode = graph.AddNode(minNewOpdesc);
  ge::NodePtr minLastNode = graph.AddNode(minLastOpdesc);
  newNodes.push_back(minNewNode);
  newNodes.push_back(minLastNode);
  // copy attr
  // float keepdims;

  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(minNewNode->GetOpDesc(), KEEPDIMS, keepdims),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set keepdims attr failed"), return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(minLastNode->GetOpDesc(), KEEPDIMS, keepdims),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Set keepdims attr failed"), return FAILED);

  // set axis
  // Assign -1 to the axis property of mindnew
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(minNewNode->GetOpDesc(), AXIS, axis_xin),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "minNew Set axis attr failed"), return FAILED);

  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(minLastNode->GetOpDesc(), AXIS, axis_last),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "minLast Set axis attr failed"), return FAILED);

  // connect output edge for minnew
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(minNewNode->GetOutDataAnchor(0), minLastNode->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);

  // copy output edge for minlast
  for (auto inDataAnchor : max_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(max_node->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "max_node Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(minLastNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "minLastNode Add out data edge failed."), return FAILED);
  }

  if (max_node->GetOutControlAnchor()) {
    for (auto inControlAnchor : max_node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(max_node->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "max_node Remove out control edge failed."), return FAILED);
      FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(minLastNode->GetOutControlAnchor(), inControlAnchor) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "minLastNode Add out control edge failed."), return FAILED);
    }
  }

  // connect input for minnew
  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(max_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            minNewNode->GetInDataAnchor(0)) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "minNewNode Add edge between node %s. and node %s failed.",
                            max_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                            minNewNode->GetName().c_str()),
                    return FAILED);

  // set grad op type to BNInferGrad
  minNewNode->GetOpDesc()->SetType(MAXD);
  minLastNode->GetOpDesc()->SetType(MAXDLAST);

  FUSION_PASS_CHECK(graph.RemoveNode(max_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove max_node node failed."), return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define ReduceMaxDFusionPass fusion end");
  return SUCCESS;
}

REGISTER_PASS("ReduceMaxDFusionPass", BUILT_IN_GRAPH_PASS, ReduceMaxDFusionPass);
}  // namespace fe
