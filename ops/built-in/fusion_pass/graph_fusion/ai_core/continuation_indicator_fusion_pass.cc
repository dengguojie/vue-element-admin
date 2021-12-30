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
 * \file continuation_indicator_fusion_pass.cc
 * \brief for caffe (ContinuationIndicator)
 */
#include <string>
#include <vector>
#include "continuation_indicator_fusion_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include <string>
#include <vector>

namespace fe {
  static const int64_t INVALID_NUM = 0;
  static const string PATTERN_CONTINUATIONINDICATOR = "ContinuationIndicator";
  /*

     ContinuationIndicator         ----->            const

  */
 vector<FusionPattern*> ContinuationIndicatorFusionPass::DefinePatterns() {
    vector<FusionPattern*> patterns;
    FusionPattern* pattern = new (std::nothrow) FusionPattern("ContinuationIndicatorFusion");
    if (pattern == nullptr) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "pattern is nullptr, Create pattern not success!");
      return patterns;
    }
    pattern->AddOpDesc(PATTERN_CONTINUATIONINDICATOR, {FUSED_OP_TYPE}).SetOutput(PATTERN_CONTINUATIONINDICATOR);
    patterns.push_back(pattern);
    return patterns;
}

Status ContinuationIndicatorFusionPass::CreatConstNode(ge::ComputeGraph& graph, ge::NodePtr &const_node,
                                                       int64_t time_step, int64_t batch_size) {
  if (time_step <= INVALID_NUM || batch_size <=  INVALID_NUM) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Time step and batch_size must be greater than 0,"
            "cur time step is %lld, batch_size is %lld.", time_step, batch_size);
    return FAILED;
  }
  // compute core
  int64_t tensor_size = time_step * batch_size;
  unique_ptr<float[]> output_res(new (std::nothrow) float[tensor_size]());
  FUSION_PASS_CHECK(output_res.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "create output data failed"),
                    return PARAM_INVALID);
  float *output_res_ptr = output_res.get();
  if (output_res_ptr == nullptr) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Get output data ptr failed.");
    return FAILED;
  }
  for (int64_t t = 0; t < time_step; ++t) {
    for (int64_t b = 0; b < batch_size; ++b) {
      output_res_ptr[t * batch_size + b] = (t == 0 ? 0 : 1);
    }
  }
  // set const node
  ge::GeTensorDesc tensor_desc;
  ge::GeShape shape(vector<int64_t>{time_step, batch_size});
  tensor_desc.SetShape(shape);
  tensor_desc.SetDataType(ge::DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  auto tensor_ptr = std::make_shared<ge::GeTensor>(tensor_desc, reinterpret_cast<uint8_t*>(output_res.get()),
      time_step*batch_size*sizeof(float));
  ge::OpDescPtr const_desc = ge::OpDescUtils::CreateConstOp(tensor_ptr);
  const_node = graph.AddNode(const_desc);
  return SUCCESS;
}

Status ContinuationIndicatorFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                               vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr fusion_node = GetNodeFromMapping(PATTERN_CONTINUATIONINDICATOR, mapping);
  FUSION_PASS_CHECK(fusion_node == nullptr, OP_LOGW(FUSED_OP_TYPE.c_str(), "Fusion node is null"), return FAILED);

  ge::OpDescPtr node_desc = fusion_node->GetOpDesc();
  FUSION_PASS_CHECK(node_desc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "ContinuationIndicator[%s] is not supported by FE, fusion abort.",
                            node_desc->GetName().c_str()),
                    return PARAM_INVALID);

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusion_node);
  int64_t time_step;
  if (op.GetAttr("time_step", time_step) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "GetOpAttr time_step failed!");
    return PARAM_INVALID;
  }
  int64_t batch_size;
  if (op.GetAttr("batch_size", batch_size) != ge::GRAPH_SUCCESS) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "GetOpAttr batch_size failed!");
    return PARAM_INVALID;
  }
  ge::NodePtr const_node = nullptr;
  if (CreatConstNode(graph, const_node, time_step, batch_size) != SUCCESS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Create const node failed.");
    return FAILED;
  }
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Attr time_step = %lld, batch_size = %lld ", time_step, batch_size);
  // delete continuation_indicator node, add const node
  for (auto in_data_anchor : fusion_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusion_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), in_data_anchor) != SUCCESS,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add out data edge failed."), return FAILED);
  }
  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(fusion_node) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove fusion_node failed."),
                    return FAILED);
  return SUCCESS;
}
REGISTER_PASS("ContinuationIndicatorFusionPass", BUILT_IN_GRAPH_PASS, ContinuationIndicatorFusionPass);
}  // namespace fe
