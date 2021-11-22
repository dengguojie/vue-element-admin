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
 * \file scope_instancenorm_grad_pass.cc
 * \brief
 */
#include "scope_instancenorm_grad_pass.h"

#include <set>

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
namespace {
const char* const kScopeType = "instance_norm_grad";
const char* const kScopeRltType = "InstanceNormGrad";
const char* const kOpType = "InstanceNormGrad";
}  // namespace

std::vector<ScopeFusionPatterns> ScopeInstanceNormGradPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns;
  GenScopePatterns(patterns);  // recognize instancenorm grad Scope
  patterns_list.push_back(patterns);

  return patterns_list;
}

std::string ScopeInstanceNormGradPass::PassName() {
  return std::string("ScopeInstanceNormGradPass");
}

void const ScopeInstanceNormGradPass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  // recognize moments
  std::vector<ScopePattern*> batch1;
  ScopePattern* moments_grad = new (std::nothrow) ScopePattern();
  if (moments_grad == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  moments_grad->SetSubType("moments_grad");
  moments_grad->AddScopeFeature(ScopeFeature("", -1, "moments", "SquaredDifference_grad"));  // moments grad
  batch1.push_back(moments_grad);

  // recognize instancenorm
  ScopePattern* instancenorm_grad = new (std::nothrow) ScopePattern();
  if (instancenorm_grad == nullptr) {
    delete moments_grad;
    moments_grad = nullptr;
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  instancenorm_grad->SetSubType("instancenorm_grad");
  instancenorm_grad->AddScopeFeature(ScopeFeature("", -1, "instancenorm", "Rsqrt_grad"));  // instancenorm grad
  batch1.push_back(instancenorm_grad);
  patterns.push_back(batch1);

  // recognize InstanceNorm grad
  std::vector<ScopePattern*> batch2;
  ScopePattern* instance_norm_grad = new (std::nothrow) ScopePattern();
  if (instance_norm_grad == nullptr) {
    delete moments_grad;
    moments_grad = nullptr;
    delete instance_norm_grad;
    instance_norm_grad = nullptr;
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  instance_norm_grad->SetSubType(kScopeType);
  instance_norm_grad->AddScopeFeature(ScopeFeature("moments_grad", 1, "InstanceNorm"));
  instance_norm_grad->AddScopeFeature(ScopeFeature("instancenorm_grad", 1, "InstanceNorm"));
  batch2.push_back(instance_norm_grad);
  patterns.push_back(batch2);
}

void FindInputsShouldInScopeING(const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map,
                                const std::unordered_map<std::string, ge::OperatorPtr>& all_nodes_map,
                                std::vector<ge::OperatorPtr>& result_nodes) {
  std::set<std::string> inputs_name;
  for (auto& it : nodes_map) {
    for (size_t i = 0; i < it.second->GetInputsSize(); i++) {
      // some input name contain "^", should be removed
      auto input_desc = it.second->GetInputDesc(i);
      std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
      if (nodes_map.count(input_name) == 0) {
        inputs_name.insert(input_name);
      }
    }
  }

  for (auto& name : inputs_name) {
    auto iter = all_nodes_map.find(name);
    if (iter != all_nodes_map.end()) {
      size_t count = 0;
      auto node_op = iter->second;
      for (size_t i = 0; i < node_op->GetInputsSize(); i++) {
        auto input_desc = node_op->GetInputDesc(i);
        std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
        if (nodes_map.count(input_name) != 0) {
          count++;
        }
      }
      if (count != 0 && count == node_op->GetInputsSize()) {
        result_nodes.push_back(node_op);
      }
    }
  }
}

std::vector<ge::OperatorPtr> ScopeInstanceNormGradPass::FindOutNodesShouldInScope(
    const Scope* scope, std::shared_ptr<ScopeGraph>& scope_graph, const std::string& mask) {
  // find inputs of scope should be in fused scope
  std::vector<ge::OperatorPtr> result_nodes;
  const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
  const std::unordered_map<std::string, ge::OperatorPtr>& all_nodes_map = scope_graph->GetNodesMap();
  FindInputsShouldInScopeING(nodes_map, all_nodes_map, result_nodes);
  // find outputs of scope should be in fused scope
  for (auto& it : all_nodes_map) {
    // node name should contain mask
    if ((mask != "") && (it.second->GetName().find(mask) == std::string::npos)) {
      continue;
    }
    // find the node whose all inputs are from this scope
    bool can_find = false;
    bool all_in = true;
    OP_LOGD(kOpType, "Addn node name is %s", it.second->GetName().c_str());
    size_t input_nums = it.second->GetInputsSize();
    for (size_t i = 0; i < input_nums; i++) {
      // some input name contain "^", should be removed
      auto input_desc = it.second->GetInputDesc(i);
      std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
      if (nodes_map.count(input_name) != 0) {
        can_find = true;
      }
      if (nodes_map.count(input_name) == 0) {
        all_in = false;
      }
    }

    if ((all_in == true && input_nums == 2) || (input_nums > 2 && can_find == true)) {
      // if the node is not in result_nodes, it will be insert to result_nodes
      bool find1 = true;
      for (auto& it1 : result_nodes) {
        if (it1->GetName() == it.second->GetName()) {
          find1 = false;
          break;
        }
      }
      if (find1) {
        result_nodes.push_back(it.second);
      }
    }
  }
  return result_nodes;
}

Status ScopeInstanceNormGradPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
                                                        std::vector<ScopesResult>& results) {
  if (scope_graph == nullptr) {
    OP_LOGE(kOpType, "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }
  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();

  for (auto& scope : scopes) {
    if (scope == nullptr) {
      OP_LOGE(kOpType, "Scope is nullptr.");
      return domi::PARAM_INVALID;
    }
    if (scope->SubType() == kScopeType) {
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      std::vector<ge::OperatorPtr> in_scope_nodes = FindOutNodesShouldInScope(scope, scope_graph, "AddN");
      result.SetNodes(in_scope_nodes);
      results.push_back(result);
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void const ScopeInstanceNormGradPass::FindInputIndex(const Scope* scope, int& index, const std::string& name,
                                               const std::string& base_name) {
  if (scope == nullptr) {
    OP_LOGE(kOpType, "scope is nullptr.");
    return;
  }
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
  }
}

void const ScopeInstanceNormGradPass::FindInputXIndex(const Scope* scope, int& index) {
  if (scope == nullptr) {
    OP_LOGE(kOpType, "scope is nullptr.");
    return;
  }
  const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
  std::vector<std::string> mul_input_names;
  std::vector<std::string> mul1_input_names;
  for (auto& it : nodes_map) {
    auto node_def = it.second;
    std::string names = "instancenorm/mul_1_grad/Mul";
    std::string sub_name = (node_def->GetName().length() > names.length())
                               ? node_def->GetName().substr(node_def->GetName().length() - names.length())
                               : node_def->GetName();
    if (sub_name == names) {
      for (size_t i = 0; i < node_def->GetInputsSize(); i++) {
        // some input name contain "^", should be removed
        auto input_desc = node_def->GetInputDesc(i);
        std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
        mul_input_names.push_back(input_name);
      }
    }

    std::string names1 = "instancenorm/mul_1_grad/Mul_1";
    std::string sub_name1 = (node_def->GetName().length() > names1.length())
                                ? node_def->GetName().substr(node_def->GetName().length() - names1.length())
                                : node_def->GetName();
    if (sub_name1 == names1) {
      for (size_t i = 0; i < node_def->GetInputsSize(); i++) {
        // some input name contain "^", should be removed
        auto input_desc = node_def->GetInputDesc(i);
        std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
        mul1_input_names.push_back(input_name);
      }
    }
  }
  for (unsigned int i = 0; i < mul1_input_names.size(); i++) {
    std::string mul1_input_name = mul1_input_names[i];
    bool find = false;
    for (unsigned int j = 0; j < mul_input_names.size(); j++) {
      if (mul1_input_name == mul_input_names[j]) {
        find = true;
        break;
      }
    }
    if (!find) {
      index = i;
      return;
    }
  }
  return;
}

void ScopeInstanceNormGradPass::FindOutputdXNode(const std::vector<ge::OperatorPtr>& nodes, std::string& node_def_name,
                                                 int& index) {
  for (auto& node_def : nodes) {
    OP_LOGI(kOpType, "Result node name is %s", node_def->GetName().c_str());
    if (node_def->GetName().find("AddN") != std::string::npos) {
      bool find_mul1 = false;
      bool find_mul = false;
      bool find_scope_external_input = false;
      size_t input_size = node_def->GetInputsSize();
      for (size_t i = 0; i < input_size; i++) {
        // some input name contain "^", should be removed
        auto input_desc = node_def->GetInputDesc(i);
        std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
        if (input_name.find("instancenorm/mul_2_grad/Mul_1") != std::string::npos) {
          find_mul1 = true;
        }
        if (input_name.find("instancenorm/mul_1_grad/Mul") != std::string::npos) {
          find_mul = true;
        }
      }
      if ((!find_mul1 && find_mul && !find_scope_external_input) || (index != -1)) {
        node_def_name = node_def->GetName();
        return;
      }
    }
  }
  return;
}

void ScopeInstanceNormGradPass::ProcessInputDy(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  int dy_index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, dy_index, "instancenorm/mul_1_grad/Mul", "instancenorm");
  }
  if (dy_index == 0) {
    fusion_rlt->InsertInputs("instancenorm/mul_1_grad/Mul", {0, kFusionDisableIndex});  // input dy
  } else if (dy_index == 1) {
    fusion_rlt->InsertInputs("instancenorm/mul_1_grad/Mul", {kFusionDisableIndex, 0});  // input dy
  } else {
    OP_LOGE(kOpType, "Only support input dy_index(%d) is 0 or 1", dy_index);
    return;
  }
}

void ScopeInstanceNormGradPass::ProcessInputX(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  int x_index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputXIndex(scope, x_index);
  }
  if (x_index == 0) {
    fusion_rlt->InsertInputs("instancenorm/mul_1_grad/Mul_1", {1, kFusionDisableIndex});  // input x
  } else if (x_index == 1) {
    fusion_rlt->InsertInputs("instancenorm/mul_1_grad/Mul_1", {kFusionDisableIndex, 1});  // input x
  } else {
    OP_LOGE(kOpType, "Only support input x_index(%d) is 0 or 1", x_index);
    return;
  }
}

void ScopeInstanceNormGradPass::ProcessInputGamma(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  int gamma_index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, gamma_index, "instancenorm/mul_grad/Mul", "gradients");
  }
  if (gamma_index == 0) {
    fusion_rlt->InsertInputs("instancenorm/mul_grad/Mul", {4, kFusionDisableIndex});  // input gamma
  } else if (gamma_index == 1) {
    fusion_rlt->InsertInputs("instancenorm/mul_grad/Mul", {kFusionDisableIndex, 4});  // input gamma
  } else {
    OP_LOGE(kOpType, "Only support input gamma_index(%d) is 0 or 1", gamma_index);
    return;
  }
}

void ScopeInstanceNormGradPass::GenerateFusionResult(const std::vector<Scope*>& scopes,
                                                     FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  ProcessInputDy(scopes, fusion_rlt);
  ProcessInputX(scopes, fusion_rlt);

  fusion_rlt->InsertInputs("instancenorm/Rsqrt_grad/RsqrtGrad", {2});  // input variance

  fusion_rlt->InsertInputs("moments/SquaredDifference_grad/sub", {kFusionDisableIndex, 3});  // input mean
  ProcessInputGamma(scopes, fusion_rlt);
  fusion_rlt->InsertOutputs("instancenorm/mul_grad/Sum", {1});  // output d_gamma
  fusion_rlt->InsertOutputs("instancenorm/sub_grad/Sum", {2});  // output d_beta
  std::string output_dx;
  int add_index = -1;
  FindOutputdXNode(fusion_rlt->Nodes(), output_dx, add_index);
  if (output_dx == "") {
    OP_LOGE(kOpType, "Not find output dX node name");
    return;
  } else {
    OP_LOGI(kOpType, "Output DX name is %s.", output_dx.c_str());
    for (auto& scope : scopes) {
      // The upper call guarantees that the scope is not empty.
      if (scope->SubType() == kScopeType) {
        std::string scope_name = scope->Name();
        OP_LOGD(kOpType, "InstanceNormGrad scope name is %s", scope_name.substr(0, scope_name.length() - 1).c_str());
        fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
        break;
      }
    }
    if (add_index == -1) {
      OP_LOGD(kOpType, "single-to-single fusion case in");
      fusion_rlt->InsertOutputs(output_dx.c_str(), {0});
      fusion_rlt->SetType(kScopeRltType);
    }
  }
  fusion_rlt->SetDescription("");
  OP_LOGI(kOpType, "InstanceNormGrad Scope fusion success.");
  return;
}
}  // namespace ge
