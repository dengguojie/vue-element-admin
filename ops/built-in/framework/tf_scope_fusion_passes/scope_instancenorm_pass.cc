/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file scope_instancenorm_pass.cc
 * \brief
 */
#include "scope_instancenorm_pass.h"

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
namespace {
const char* const kScopeTypeInstancenorm = "instancenorm";
const char* const kScopeTypeMoments = "moments";
const char* const kScopeRltType = "InstanceNorm";
const char* const kOpType = "InstanceNorm";
}  // namespace

std::vector<ScopeFusionPatterns> ScopeInstanceNormPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns_instancenorm;
  ScopeFusionPatterns patterns_moments;
  GenInstancenormScopePatterns(patterns_instancenorm);  // match instancenorm Scope
  GenMomentsScopePatterns(patterns_moments);            // match moments Scope
  patterns_list.push_back(patterns_instancenorm);
  patterns_list.push_back(patterns_moments);
  return patterns_list;
}

std::string ScopeInstanceNormPass::PassName() {
  return std::string("ScopeInstanceNormPass");
}

/**
 * @brief LastMatch for multiple scopes
 */
Status ScopeInstanceNormPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
                                                    std::vector<ScopesResult>& results) {
  if (scope_graph == nullptr) {
    OP_LOGE(kOpType, "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }
  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  if (scope_tree == nullptr) {
    OP_LOGE(kOpType, "Scope tree is nullptr.");
    return domi::PARAM_INVALID;
  }
  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();
  std::vector<Scope*> fusion_scopes_in;
  std::vector<Scope*> fusion_scopes_m;
  for (auto& scope : scopes) {
    // uwc can not do scope fusion
    if (scope->Name().find("sequence_encoder_loop/while") != std::string::npos) {
      return FAILED;
    }
    // tf2.x can not do scope fusion
    if (scope->Name().find("instancenorm/mul") != std::string::npos) {
      const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
      for (auto& it : nodes_map) {
        OP_LOGI(kOpType, "instancenorm/mul Scope has node %s.", it.second->GetName().c_str());
        if (it.second->GetName().find("mul/ReadVariableOp") != std::string::npos) {
          return FAILED;
        }
      }
    }
    // Class ScopeTree guarantees scope is not empty.
    if ((scope->SubType() == kScopeTypeInstancenorm)) {
      fusion_scopes_in.push_back(scope);
    } else if (scope->SubType() == kScopeTypeMoments) {
      fusion_scopes_m.push_back(scope);
    }
  }

  if (fusion_scopes_in.size() == fusion_scopes_m.size()) {
    // the two scope instancenorm and moments in the same instancenorm
    for (size_t i = 0; i < fusion_scopes_in.size(); i++) {
      auto scope_bn = fusion_scopes_in[i];
      for (size_t j = 0; j < fusion_scopes_m.size(); j++) {
        auto scope_m = fusion_scopes_m[j];
        std::string scope_bn_name = scope_bn->Name();
        std::string scope_m_name = scope_m->Name();
        int pos_bn = scope_bn_name.find("instancenorm");
        int pos_m = scope_m_name.find("moments");
        int is_biggan_bn = scope_bn_name.find("resblock");
        int is_biggan_m = scope_m_name.find("resblock");
        if (is_biggan_bn != -1 || is_biggan_m != -1) {
          return FAILED;
        }
        if (pos_bn != -1 && pos_m != -1 && scope_bn_name.substr(0, pos_bn) == scope_m_name.substr(0, pos_m)) {
          // scope result
          ScopesResult result;
          std::vector<Scope*> result_scopes;
          result_scopes.push_back(scope_bn);
          result_scopes.push_back(scope_m);
          result.SetScopes(result_scopes);
          results.push_back(result);
          OP_LOGI(kOpType, "scope:%s, and scope:%s is connect.", scope_bn->Name().c_str(), scope_m->Name().c_str());
          break;
        }
      }
    }
  }
  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeInstanceNormPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }

  int index1 = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, index1, "instancenorm/mul_1", "instancenorm");
  }
  if (index1 == 0) {
    fusion_rlt->InsertInputs("instancenorm/mul_1", {0, kFusionDisableIndex});
  } else if (index1 == 1) {
    fusion_rlt->InsertInputs("instancenorm/mul_1", {kFusionDisableIndex, 0});  // input x
  } else {
    OP_LOGE(kOpType, "Only support input xIndex(%d) is 0 or 1", index1);
    return;
  }

  int index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, index, "instancenorm/mul", "instancenorm");
  }
  if (index == 0) {
    fusion_rlt->InsertInputs("instancenorm/mul", {1, kFusionDisableIndex});
  } else if (index == 1) {
    fusion_rlt->InsertInputs("instancenorm/mul", {kFusionDisableIndex, 1});  // input gamma
  } else {
    OP_LOGE(kOpType, "Only support input gammaIndex(%d) is 0 or 1", index);
    return;
  }

  fusion_rlt->InsertInputs("instancenorm/sub", {2, kFusionDisableIndex});  // input beta
  fusion_rlt->InsertOutputs("instancenorm/add_1", {0});                    // output y
  fusion_rlt->InsertOutputs("moments/mean", {1});                          // output mean
  fusion_rlt->InsertOutputs("instancenorm/Rsqrt", {2});                    // output variance

  for (auto& scope : scopes) {
    // The upper call guarantees that the scope is not empty.
    if (scope->SubType() == kScopeTypeInstancenorm) {
      std::string scope_bn_name = scope->Name();
      int pos_bn = scope_bn_name.find("instancenorm");
      if (pos_bn != -1) {
        fusion_rlt->SetName(scope_bn_name.substr(0, pos_bn));
        break;
      }
    }
  }

  fusion_rlt->SetType(kScopeRltType);  // op Type InstanceNorm
  fusion_rlt->SetDescription("");

  OP_LOGI(kOpType, "InstanceNorm Scope fusion success.");
  return;
}

void ScopeInstanceNormPass::GenInstancenormScopePatterns(ScopeFusionPatterns& patterns) const {
  // match instancenorm
  std::vector<ScopePattern*> instance;
  ScopePattern* instance_norm_cell = new (std::nothrow) ScopePattern();
  if (instance_norm_cell == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  instance_norm_cell->SetSubType(kScopeTypeInstancenorm);

  instance_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Sub", 1));        // Sub num is 1
  instance_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Rsqrt", 1));      // Rsqrt num is 1
  instance_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 3));        // Mul num is 3
  instance_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Identity", -1));  // Identity num is -1
  instance_norm_cell->AddScopeFeature(ScopeFeature("", -1, "instancenorm"));

  instance.push_back(instance_norm_cell);
  patterns.push_back(instance);
}

void ScopeInstanceNormPass::GenMomentsScopePatterns(ScopeFusionPatterns& patterns) const {
  // match moments
  std::vector<ScopePattern*> instance;
  ScopePattern* moments_cell = new (std::nothrow) ScopePattern();
  if (moments_cell == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  moments_cell->SetSubType(kScopeTypeMoments);
  moments_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Mean", 2));               // Mean num is 2
  moments_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("SquaredDifference", 1));  // SquaredDifference num is 1
  moments_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Squeeze", -1));           // Squeeze num is 0
  moments_cell->AddScopeFeature(ScopeFeature("", -1, "moments"));

  instance.push_back(moments_cell);
  patterns.push_back(instance);
}

void ScopeInstanceNormPass::FindInputIndex(const Scope* scope, int& index, const std::string& name,
                                           const std::string& base_name) const {
  const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
  std::vector<std::string> mul_input_names;
  for (auto& it : nodes_map) {
    auto node_def = it.second;
    std::string sub_name = (node_def->GetName().length() > name.length())
                               ? node_def->GetName().substr(node_def->GetName().length() - name.length())
                               : node_def->GetName();
    if (sub_name == name) {
      for (size_t i = 0; i < node_def->GetInputsSize(); i++) {
        auto input_desc = node_def->GetInputDesc(i);
        std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
        mul_input_names.push_back(input_name);
      }
    }
  }
  for (unsigned int i = 0; i < mul_input_names.size(); i++) {
    std::string mul_input_name = mul_input_names[i];
    OP_LOGI(kOpType, "The %s of inputname is %s", name.c_str(), mul_input_name.c_str());
    if (mul_input_name.find(base_name) == mul_input_name.npos) {
      index = i;
      OP_LOGI(kOpType, "The %s is not found, the index is %d", base_name.c_str(), i);
      return;
    }
    if (mul_input_name.find("instancenorm/mul/Enter") != string::npos) {
      index = i;
      OP_LOGI(kOpType, "found instancenorm/mul/Enter, the index is %d", i);
      return;
    }
    if (mul_input_name.find("instancenorm/add_1") != string::npos) {
      index = i;
      OP_LOGI(kOpType, "found instancenorm/add_1, the index is %d", i);
      return;
    }
  }
}
}  // namespace ge
