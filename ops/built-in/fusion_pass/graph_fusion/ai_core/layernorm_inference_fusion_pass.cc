/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file layernorm_inference_fusion_pass.cc
 * \brief fuse matched scope to target LayerNorm
 */
#include "layernorm_pass_libs.h"
#include "layernorm_inference_fusion_pass.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "op_const.h"
#include "op_attr.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const std::string ADD = "Add";
static const std::string MUL = "Mul";
static const std::string SUB = "Sub";
static const std::string RSQRT = "Rsqrt";
static const std::string REDUCE_MEAN = "ReduceMean";
static const std::string SQUARED_DIFFERENCE = "SquaredDifference";
static const std::string RSQRT_GRAD = "RsqrtGrad";
static const char* PATTERN_ADD_1 = "add_1";
static const char* PATTERN_ADD_2 = "add_2";
static const char* PATTERN_MUL_1 = "mul_1";
static const char* PATTERN_MUL_2 = "mul_2";
static const char* PATTERN_MUL_3 = "mul_3";
static const char* PATTERN_SUB = "sub";
static const char* PATTERN_RSQRT = "rsqrt";
static const char* PATTERN_REDUCE_MEAN_1 = "reduce_mean_1";
static const char* PATTERN_REDUCE_MEAN_2 = "reduce_mean_2";
static const char* PATTERN_SQUARED_DIFFERENCE = "squared_difference";

vector<FusionPattern*> LayerNormInferenceFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("LayerNormInferenceFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new an object failed."), return patterns);
  pattern->AddOpDesc(PATTERN_REDUCE_MEAN_1, {REDUCE_MEAN})
          .AddOpDesc(PATTERN_SQUARED_DIFFERENCE, {SQUARED_DIFFERENCE})
          .AddOpDesc(PATTERN_REDUCE_MEAN_2, {REDUCE_MEAN})
          .AddOpDesc(PATTERN_ADD_1, {ADD})
          .AddOpDesc(PATTERN_RSQRT, {RSQRT})
          .AddOpDesc(PATTERN_MUL_1, {MUL})
          .AddOpDesc(PATTERN_MUL_2, {MUL})
          .AddOpDesc(PATTERN_MUL_3, {MUL})
          .AddOpDesc(PATTERN_SUB, {SUB})
          .AddOpDesc(PATTERN_ADD_2, {ADD})
          .SetInputs(PATTERN_SQUARED_DIFFERENCE, {PATTERN_REDUCE_MEAN_1})
          .SetInputs(PATTERN_REDUCE_MEAN_2, {PATTERN_SQUARED_DIFFERENCE})
          .SetInputs(PATTERN_ADD_1, {PATTERN_REDUCE_MEAN_2})
          .SetInputs(PATTERN_RSQRT, {PATTERN_ADD_1})
          .SetInputs(PATTERN_MUL_1, {PATTERN_RSQRT})
          .SetInputs(PATTERN_MUL_2, {PATTERN_REDUCE_MEAN_1, PATTERN_MUL_1})
          .SetInputs(PATTERN_MUL_3, {PATTERN_MUL_1})
          .SetInputs(PATTERN_SUB, {PATTERN_MUL_2})
          .SetInputs(PATTERN_ADD_2, {PATTERN_MUL_3, PATTERN_SUB})
          .SetOutput(PATTERN_ADD_2);
  patterns.emplace_back(pattern);

  return patterns;
}

bool LayerNormInferenceFusionPass::IsTrainingFlow(const ge::NodePtr& rsqrt_node) {
  bool is_training = false;
  for(const auto& node : FusionTurbo::GetPeerInNodes(rsqrt_node, 0)) {
    if (node->GetType() == RSQRT_GRAD) {
      is_training = true;
      break;
    }
  }

  return is_training;
}

Status LayerNormInferenceFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                            vector<ge::NodePtr>& new_nodes) {
  ge::NodePtr mean_node_1 = GetNodeFromMapping(PATTERN_REDUCE_MEAN_1, mapping);
  ge::NodePtr squared_difference_node = GetNodeFromMapping(PATTERN_SQUARED_DIFFERENCE, mapping);
  ge::NodePtr mean_node_2 = GetNodeFromMapping(PATTERN_REDUCE_MEAN_2, mapping);
  ge::NodePtr add_node_1 = GetNodeFromMapping(PATTERN_ADD_1, mapping);
  ge::NodePtr rsqrt_node = GetNodeFromMapping(PATTERN_RSQRT, mapping);
  ge::NodePtr mul_node_1 = GetNodeFromMapping(PATTERN_MUL_1, mapping);
  ge::NodePtr mul_node_2 = GetNodeFromMapping(PATTERN_MUL_2, mapping);
  ge::NodePtr mul_node_3 = GetNodeFromMapping(PATTERN_MUL_3, mapping);
  ge::NodePtr sub_node = GetNodeFromMapping(PATTERN_SUB, mapping);
  ge::NodePtr add_node_2 = GetNodeFromMapping(PATTERN_ADD_2, mapping);

  std::vector<ge::NodePtr> pattern_nodes = {mean_node_1, squared_difference_node, mean_node_2, add_node_1,
                                            rsqrt_node, mul_node_1, mul_node_2, mul_node_3, sub_node, add_node_2};
  FUSION_PASS_CHECK(CheckNullPtr(pattern_nodes) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "node has nullptr, please check."), return NOT_CHANGED);

  FUSION_PASS_CHECK(IsTrainingFlow(rsqrt_node) == true,
                    OP_LOGD(FUSED_OP_TYPE, "for training, do nothing."), return NOT_CHANGED);

  std::vector<int64_t> axes_1 = {-1};
  bool keep_dims_1 = false;
  FUSION_PASS_CHECK(GetReduceOpAttr(axes_1, keep_dims_1, mean_node_1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to get attribute of mean1."), return NOT_CHANGED);
  FUSION_PASS_CHECK(CheckReduceOpAttr(FUSED_OP_TYPE, axes_1, keep_dims_1, mean_node_1, mean_node_2) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to match attribute of mean1."), return NOT_CHANGED);

  // add layernorm into graph
  FusionTurbo turbo_instance(graph);
  std::string layer_norm_name = mean_node_1->GetOpDesc()->GetName() + "/" + FUSED_OP_TYPE;
  ge::NodePtr layer_norm_node = turbo_instance.AddNodeOnly(layer_norm_name, FUSED_OP_TYPE);
  FUSION_PASS_CHECK(layer_norm_node == nullptr,
                    OP_LOGD(FUSED_OP_TYPE, "failed to add LayerNorm node."), return NOT_CHANGED);
  FUSION_PASS_CHECK(SetLayerNormAttr(FUSED_OP_TYPE, layer_norm_node, axes_1, add_node_1) != SUCCESS,
                    OP_LOGD(FUSED_OP_TYPE, "failed to set LayerNorm's attributes."), return NOT_CHANGED);

  // to be LayerNorm
  std::pair<ge::NodePtr, ge::NodePtr> input_pair_0(squared_difference_node, mean_node_1);
  std::pair<ge::NodePtr, ge::NodePtr> input_pair_1(mul_node_1, rsqrt_node);
  std::pair<ge::NodePtr, ge::NodePtr> input_pair_2(sub_node, mul_node_2);
  std::vector<std::pair<ge::NodePtr, ge::NodePtr>> input_node_pairs = {input_pair_0, input_pair_1, input_pair_2};
  std::vector<ge::NodePtr> output_nodes = {add_node_2};
  Relations input_relations;
  Relations output_relations;
  GetInputRelations(input_relations, input_node_pairs);
  GetOutputRelations(output_relations, output_nodes);
  Status ret = turbo_instance.MultiInOne(layer_norm_node, input_relations, output_relations, pattern_nodes);
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to fuse LayerNorm."), return FAILED);

  new_nodes.emplace_back(layer_norm_node);

  return SUCCESS;
}
REGISTER_PASS("LayerNormInferenceFusionPass", BUILT_IN_GRAPH_PASS, LayerNormInferenceFusionPass);
}  // namespace fe
