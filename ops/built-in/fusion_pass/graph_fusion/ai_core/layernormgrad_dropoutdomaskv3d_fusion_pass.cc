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
 * \file layernormgrad_dropoutdomaskv3d_fusion_pass.cpp
 * \brief LayerNormGradDropOutDoMaskV3D fusion pass
 *   (LayerNormGrad & DropOutDoMaskV3D --> LNDropoutGrad)
 */
#include "layernormgrad_dropoutdomaskv3d_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/util/platform_info.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/fusion_common/fusion_turbo.h"

using namespace ge;
namespace fe {
static const std::string LAYER_NORM_GRAD = "LayerNormGrad";
static const std::string DROP_OUT_DO_MASK_V3D = "DropOutDoMaskV3D";
static const std::string TYPE_CAST = "Cast";
static const std::string PATTERN_LAYER_NORAM_GRAD = "FusedLayerNormGrad";
static const std::string PATTERN_DROP_OUT_DO_MASK_V3D = "FusedDropOutDoMaskV3D";
static const std::string PATTERN_TYPE_CAST = "FusedCast";
static const std::string KEEP_PROB = "keep_prob";
static const int64_t C0 = 16;
static const int64_t MAX_REDUCE_SIZE = 1536;
static const int64_t MIN_SHAPE_SIZE = 2;

vector<FusionPattern *> LayerNormGradDropOutDoMaskV3DFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern_with_cast = new(std::nothrow) FusionPattern("LNDropOutGradWithCastFusionPass");
  FusionPattern *pattern_without_cast = new(std::nothrow) FusionPattern("LNDropOutGradWithoutCastFusionPass");
  FUSION_PASS_CHECK(pattern_with_cast == nullptr || pattern_without_cast == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "new a pattern object failed."),
                    return patterns);

  pattern_with_cast->AddOpDesc(PATTERN_LAYER_NORAM_GRAD, {LAYER_NORM_GRAD})
      .AddOpDesc(PATTERN_DROP_OUT_DO_MASK_V3D, {DROP_OUT_DO_MASK_V3D})
      .AddOpDesc(PATTERN_TYPE_CAST, {TYPE_CAST})
      .SetInputs(PATTERN_TYPE_CAST, {PATTERN_LAYER_NORAM_GRAD})
      .SetInputs(PATTERN_DROP_OUT_DO_MASK_V3D, {PATTERN_TYPE_CAST})
      .SetOutput(PATTERN_DROP_OUT_DO_MASK_V3D);
  patterns.push_back(pattern_with_cast);

  pattern_without_cast->AddOpDesc(PATTERN_LAYER_NORAM_GRAD, {LAYER_NORM_GRAD})
      .AddOpDesc(PATTERN_DROP_OUT_DO_MASK_V3D, {DROP_OUT_DO_MASK_V3D})
      .SetInputs(PATTERN_DROP_OUT_DO_MASK_V3D, {PATTERN_LAYER_NORAM_GRAD})
      .SetOutput(PATTERN_DROP_OUT_DO_MASK_V3D);
  patterns.push_back(pattern_without_cast);

  return patterns;
}

bool LayerNormGradDropOutDoMaskV3DFusionPass::IsMatch(const ge::NodePtr& ln_grad_node,
                                                      const ge::OpDescPtr& layer_norm_grad_desc) {
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
    OP_LOGW(ln_grad_node, "Fail to get platform info.");
    return false;
  }

  if (platform_info.str_info.short_soc_version != "Ascend910") {
    OP_LOGI(ln_grad_node, "currently only supports Ascend910, which is %s.",
            platform_info.str_info.short_soc_version);
    return false;
  }

  size_t gamma_dims_num = layer_norm_grad_desc->GetInputDesc(4).GetOriginShape().GetDimNum();
  if (gamma_dims_num != 1) {
    OP_LOGI(ln_grad_node, "length of the shape for input[4] is %zu,  not equal to 1.", gamma_dims_num);
    return false;
  }

  vector<int64_t> ln_grad_input_shape = layer_norm_grad_desc->GetInputDesc(0).GetOriginShape().GetDims();
  if (ln_grad_input_shape.size() < MIN_SHAPE_SIZE) {
    OP_LOGI(ln_grad_node,
            "length of the shape for input[0] is less than [%ld], which is [%ld].",
            MIN_SHAPE_SIZE, ln_grad_input_shape.size());
    return false;
  }
  // check shape 16 aligned
  bool shape_not_aligned = ln_grad_input_shape[ln_grad_input_shape.size() - 1] % C0 != 0;
  if (shape_not_aligned) {
    OP_LOGI(ln_grad_node,
            "size of the last axis of the shape for input[0]: [%ld] is not aligned.",
            ln_grad_input_shape[ln_grad_input_shape.size() - 1]);
    return false;
  }
  if (ln_grad_input_shape[ln_grad_input_shape.size() - 1] > MAX_REDUCE_SIZE) {
    OP_LOGI(ln_grad_node,
            "size of the last axis of the shape for input[0]: [%ld] should be less than or equal to [%ld].",
            ln_grad_input_shape[ln_grad_input_shape.size() - 1], MAX_REDUCE_SIZE);
    return false;
  }

  return true;
}

Status LayerNormGradDropOutDoMaskV3DFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                                       vector<ge::NodePtr>& fusion_nodes) {
  ge::NodePtr layer_norm_grad_node = GetNodeFromMapping(PATTERN_LAYER_NORAM_GRAD, mapping);
  FUSION_PASS_CHECK(layer_norm_grad_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "LayerNormGrad is null, fusion isn't supported."),
                    return NOT_CHANGED);
  ge::NodePtr drop_out_do_mask_v3d_node = GetNodeFromMapping(PATTERN_DROP_OUT_DO_MASK_V3D, mapping);
  FUSION_PASS_CHECK(drop_out_do_mask_v3d_node == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                                   "DropOutDoMaskV3D is null, fusion isn't supported."),
                    return NOT_CHANGED);

  ge::OpDescPtr layer_norm_grad_desc = layer_norm_grad_node->GetOpDesc();
  FUSION_PASS_CHECK(layer_norm_grad_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                                   "LayerNormGrad OpDesc is null, fusion isn't supported."),
                    return NOT_CHANGED);
  ge::OpDescPtr drop_out_do_mask_v3d_desc = drop_out_do_mask_v3d_node->GetOpDesc();
  FUSION_PASS_CHECK(drop_out_do_mask_v3d_desc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE,
                                                   "DropOutDoMaskV3D OpDesc is null, fusion isn't supported."),
                    return NOT_CHANGED);

  if (!IsMatch(layer_norm_grad_node, layer_norm_grad_desc)) {
    OP_LOGD(layer_norm_grad_node, "LayerNormGradDropOutDoMaskV3DFusionPass is not matched.");
    return NOT_CHANGED;
  }

  // init FusionTurbo
  FusionTurbo lnd_turbo(graph);
  // new a target node and add it into graph
  std::string lnd_dropout_grad_name = layer_norm_grad_desc->GetName() + "/" + FUSED_OP_TYPE;
  auto lnd_dropout_grad_node = lnd_turbo.AddNodeOnly(lnd_dropout_grad_name, FUSED_OP_TYPE);
  FUSION_PASS_CHECK(lnd_dropout_grad_node == nullptr,
                    OP_LOGW(FUSED_OP_TYPE, "failed to create node LNDropoutGrad."),
                    return NOT_CHANGED);
  ge::OpDescPtr lnd_dropout_grad_desc = lnd_dropout_grad_node->GetOpDesc();
  FUSION_PASS_CHECK(lnd_dropout_grad_desc == nullptr,
                    OP_LOGW(FUSED_OP_TYPE, "LNDropoutGrad's OpDesc is null."),
                    return NOT_CHANGED);

  float keep_prob;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(drop_out_do_mask_v3d_desc, KEEP_PROB, keep_prob),
                    OP_LOGW(FUSED_OP_TYPE, "get attr keep_prob failed."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetFloat(lnd_dropout_grad_desc, KEEP_PROB, keep_prob),
                    OP_LOGW(FUSED_OP_TYPE, "set attr keep_dims failed"),
                    return NOT_CHANGED);

  // set input relations, format is {new_node_input_index, {old_node, input_index, peer_output}}
  Relations input_relations = {{0, {layer_norm_grad_node, 0, PEER}},
                               {1, {layer_norm_grad_node, 1, PEER}},
                               {2, {layer_norm_grad_node, 2, PEER}},
                               {3, {layer_norm_grad_node, 3, PEER}},
                               {4, {layer_norm_grad_node, 4, PEER}},
                               {5, {drop_out_do_mask_v3d_node, 1, PEER}}};
  // set output relations, format is {new_node_output_index, {old_node, output_index, peer_inputs}}
  Relations output_relations;
  if (layer_norm_grad_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 1) {
    Relations tmp_relations({{0, {layer_norm_grad_node, 0, PEER}},
                             {1, {drop_out_do_mask_v3d_node, 0, PEER}},
                             {2, {layer_norm_grad_node, 1, PEER}},
                             {3, {layer_norm_grad_node, 2, PEER}}});
    output_relations.UpdatePeerIndex(tmp_relations.GetRelations());
  } else {
    ge::NodePtr cast_node = GetNodeFromMapping(PATTERN_TYPE_CAST, mapping);
    FUSION_PASS_CHECK(cast_node == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Cast is null, fusion failed."),
                      return NOT_CHANGED);
    ge::OpDescPtr cast_desc = cast_node->GetOpDesc();
    FUSION_PASS_CHECK(cast_desc == nullptr,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "Cast OpDesc is null, fusion failed."),
                      return NOT_CHANGED);
    Relations tmp_relations({{0, {layer_norm_grad_node, 0, PEER}},
                             {1, {drop_out_do_mask_v3d_node, 0, PEER}},
                             {2, {layer_norm_grad_node, 1, PEER}},
                             {3, {layer_norm_grad_node, 2, PEER}}});
    output_relations.UpdatePeerIndex(tmp_relations.GetRelations());
  }

  // transform control links and delete old nodes
  Status ret = lnd_turbo.MultiInOne(lnd_dropout_grad_node,
                                    input_relations, output_relations,
                                    {layer_norm_grad_node, drop_out_do_mask_v3d_node});
  FUSION_PASS_CHECK(ret != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "failed to fuse LNDropoutGrad."),
                    return FAILED);
  fusion_nodes.push_back(lnd_dropout_grad_node);

  return SUCCESS;
}

REGISTER_PASS("LayerNormGradDropOutDoMaskV3DFusionPass", BUILT_IN_GRAPH_PASS, LayerNormGradDropOutDoMaskV3DFusionPass);
}  // namespace fe
