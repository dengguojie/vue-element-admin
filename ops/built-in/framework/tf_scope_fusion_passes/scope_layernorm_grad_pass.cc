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
 * \file scope_layernorm_grad_pass.cc
 * \brief
 */
#include "scope_layernorm_grad_pass.h"

#include <set>

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
namespace {
const char * const kScopeType = "layer_norm_grad";
const char * const kScopeRltType = "LayerNormGrad";
const char * const kOpType = "LayerNormGrad";
constexpr int32_t DST_OUTSIZE = 2;
constexpr int32_t DST_INSIZE = 2;
}  // namespace

std::vector<ScopeFusionPatterns> ScopeLayerNormGradPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns;
  GenScopePatterns(patterns);  // recognize layernorm grad Scope
  patterns_list.push_back(patterns);

  return patterns_list;
}

std::string ScopeLayerNormGradPass::PassName() {
  return std::string("ScopeLayerNormGradPass");
}

void ScopeLayerNormGradPass::GenScopePatterns(ScopeFusionPatterns& patterns) const {
  OP_LOGD(kOpType, "Enter ScopeLayerNormGradPass GenScopePatterns");
  // recognize moments
  std::vector<ScopePattern*> batch1;
  ScopePattern* moments_grad = new (std::nothrow) ScopePattern();
  if (moments_grad == nullptr) {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Alloc an object failed.");
    return;
  }
  moments_grad->SetSubType("moments_grad");
  moments_grad->AddScopeFeature(ScopeFeature("", -1, "moments", "SquaredDifference_grad"));  // moments grad
  batch1.push_back(moments_grad);

  // recognize batchnorm
  ScopePattern* batchnorm_grad = new (std::nothrow) ScopePattern();
  if (batchnorm_grad == nullptr) {
    delete moments_grad;
    moments_grad = nullptr;
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Alloc an object failed.");
    return;
  }
  batchnorm_grad->SetSubType("batchnorm_grad");
  batchnorm_grad->AddScopeFeature(ScopeFeature("", -1, "batchnorm", "Rsqrt_grad"));  // batchnorm grad
  batchnorm_grad->AddNodeOpTypeFeature(NodeOpTypeFeature("StackPopV2", -1));  // Convlstm net is not support.
  batchnorm_grad->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 6, 0));
  batch1.push_back(batchnorm_grad);
  patterns.push_back(batch1);
  OP_LOGI(kOpType, "Mul of LayerNormGrad scope is 6");

  // recognize LayerNorm grad

  std::vector<ScopePattern*> batch2;
  ScopePattern* layernorm_grad = new (std::nothrow) ScopePattern();
  if (layernorm_grad == nullptr) {
    delete moments_grad;
    moments_grad = nullptr;
    delete batchnorm_grad;
    batchnorm_grad = nullptr;
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Alloc an object failed.");
    return;
  }
  layernorm_grad->SetSubType(kScopeType);
  layernorm_grad->AddScopeFeature(ScopeFeature("moments_grad", 1, "LayerNorm"));
  layernorm_grad->AddScopeFeature(ScopeFeature("batchnorm_grad", 1, "LayerNorm"));
  batch2.push_back(layernorm_grad);
  patterns.push_back(batch2);
}

void FindInputsShouldInScope(const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map,
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

std::vector<ge::OperatorPtr> ScopeLayerNormGradPass::FindOutNodesShouldInScope(const Scope* scope,
                                                                               std::shared_ptr<ScopeGraph>& scope_graph,
                                                                               const std::string& mask) {
  // find inputs of scope should be in fused scope
  std::vector<ge::OperatorPtr> result_nodes;
  const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
  const std::unordered_map<std::string, ge::OperatorPtr>& all_nodes_map = scope_graph->GetNodesMap();
  FindInputsShouldInScope(nodes_map, all_nodes_map, result_nodes);
  // find outputs of scope should be in fused scope
  for (auto& it : all_nodes_map) {
    // node name should contain mask
    if ((mask != "") && (it.second->GetName().find(mask) == std::string::npos)) {
      continue;
    }
    // find the node whose all inputs are from this scope
    bool can_find = false;
    bool all_in = true;
    bool has_ln_output = false;
    OP_LOGI(kOpType, "Addn node name is %s", it.second->GetName().c_str());
    size_t input_nums = it.second->GetInputsSize();
    for (size_t i = 0; i < input_nums; i++) {
      // some input name contain "^", should be removed
      auto input_desc = it.second->GetInputDesc(i);
      std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
      if (nodes_map.count(input_name) != 0) {
        can_find = true;
      }
      if (input_name.find("batchnorm/mul_grad/Sum_1") != std::string::npos ||
          input_name.find("batchnorm/sub_grad/Sum") != std::string::npos) {
        has_ln_output = true;
      }
      if (nodes_map.count(input_name) == 0) {
        all_in = false;
      }
    }

    if ((all_in && input_nums == DST_INSIZE) || (input_nums > DST_INSIZE && can_find && !has_ln_output)) {
      // if the node is not in result_nodes, it will be insert to result_nodes
      bool find1 = true;
      for (auto& it1 : result_nodes) {
        if (it1->GetName() == it.second->GetName()) {
          find1 = false;
          break;
        }
      }
      if (find1) {
        OP_LOGI(kOpType, " Push Back AddN noede %s into scope", it.second->GetName().c_str());
        result_nodes.push_back(it.second);
      }
    }
  }
  return result_nodes;
}

Status ScopeLayerNormGradPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
                                                     std::vector<ScopesResult>& results) {
  OP_LOGD(kOpType, "Enter ScopeLayerNormGradPass LastMatchScopesAndOPs");
  if (scope_graph == nullptr) {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Input params is nullptr.");
    return domi::PARAM_INVALID;
  }
  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  const std::vector<Scope*>& scopes = scope_tree->GetAllScopes();

  for (auto& scope : scopes) {
    if (scope == nullptr) {
      OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Scope is nullptr.");
      return domi::PARAM_INVALID;
    }
    const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map_mul = scope->AllNodesMap();
    for (auto& it : nodes_map_mul) {
        auto mode_def = it.second;
        if (mode_def->GetName().find("gradients_2/discriminator_2/d_ln2/LayerNorm/"
        "batchnorm/mul_grad/Reshape") != std::string::npos) {
            OP_LOGI(kOpType, "ScopeLayerNormGradPass Reshape name is found");
            std::vector<std::string>outputs;
            mode_def->GetAttr("_origin_graph_node_outputs", outputs);
            OP_LOGI(kOpType, "Reshape node data_out_num is %zu", outputs.size());
            if (outputs.size() == DST_OUTSIZE) {
                OP_LOGI(kOpType, "Reshape output size is 2, not fusion scope");
                return FAILED;
            }
        }
        if (mode_def->GetName().find("gradients/vit-b16/Transformer/encoderblock") != std::string::npos) {
          OP_LOGI(kOpType, "gradients/vit-b16/Transformer/encoderblock is found");
          if (mode_def->GetName().find("LayerNorm_2/batchnorm/mul_grad/Mul") != std::string::npos) {
            if (mode_def->GetOpType() == "Mul") {
              OP_LOGI(kOpType, "Mul is found,name is %s", mode_def->GetName().c_str());
              auto input_desc0 = mode_def->GetInputDesc(0);
              std::string input_name = ScopeUtil::StringReplaceAll(input_desc0.GetName(), "^", "");
              auto input_desc1 = mode_def->GetInputDesc(1);
              std::string input_name1 = ScopeUtil::StringReplaceAll(input_desc1.GetName(), "^", "");
              if ((input_name.find("gradients/AddN") != std::string::npos) ||
              (input_name1.find("gradients/AddN") != std::string::npos)) {
                OP_LOGI(kOpType, "scope result is not fusion");
                return FAILED;
              }
            }
          }
        }
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

void ScopeLayerNormGradPass::FindInputIndex(const Scope* scope, int& index, const std::string& name,
                                            const std::string& base_name) const {
  if (scope == nullptr) {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, scope is nullptr.");
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

void ScopeLayerNormGradPass::IsConnectReshape(const Scope* scope, const std::string& name,
                                              bool& is_connected_reshape) const {
  if (scope == nullptr) {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, scope is nullptr.");
    return;
  }
  const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
  is_connected_reshape = false;
  for (auto& it : nodes_map) {
    auto node_def = it.second;
    std::string sub_name = (node_def->GetName().length() > name.length())
                              ? node_def->GetName().substr(node_def->GetName().length() - name.length())
                              : node_def->GetName();
    if (sub_name == name) {
      is_connected_reshape = true;
      return;
    }
  }
return;
}

void ScopeLayerNormGradPass::OutputGammaBetaProcess(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  bool is_connect_reshape = false;
  for (auto& scope : scopes) {
    IsConnectReshape(scope, "batchnorm/mul_grad/Reshape_1", is_connect_reshape);
  }
  if (is_connect_reshape) {
    fusion_rlt->InsertOutputs("batchnorm/mul_grad/Reshape_1", {1});  // output d_gamma
  }
  else {
    fusion_rlt->InsertOutputs("batchnorm/mul_grad/Sum_1", {1});  // output d_gamma
  }
  is_connect_reshape = false;
  for (auto& scope : scopes) {
    IsConnectReshape(scope, "batchnorm/sub_grad/Reshape", is_connect_reshape);
  }
  if (is_connect_reshape) {
    fusion_rlt->InsertOutputs("batchnorm/sub_grad/Reshape", {2});  // output d_beta
  }
  else {
    fusion_rlt->InsertOutputs("batchnorm/sub_grad/Sum", {2});  // output d_beta
  }
}

void ScopeLayerNormGradPass::FindInputXIndex(const Scope* scope, int& index) const {
  if (scope == nullptr) {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, scope is nullptr.");
    return;
  }
  const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
  std::vector<std::string> mul_input_names;
  std::vector<std::string> mul1_input_names;
  for (auto& it : nodes_map) {
    auto node_def = it.second;
    std::string names = "batchnorm/mul_1_grad/Mul";
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

    std::string names1 = "batchnorm/mul_1_grad/Mul_1";
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

void ScopeLayerNormGradPass::FindOutputdXNode(const std::vector<ge::OperatorPtr>& nodes, std::string& node_def_name,
                                              int &index) {
  for (auto& node_def : nodes) {
    OP_LOGI(kOpType, "Result node name is %s", node_def->GetName().c_str());
    if (node_def->GetName().find("AddN") != std::string::npos) {
      bool find_mul1 = false;
      bool find_mul = false;
      bool find_mul2 = false;
      bool find_scope_external_input = false;
      size_t input_size = node_def->GetInputsSize();
      for (size_t i = 0; i < input_size; i++) {
        // some input name contain "^", should be removed
        auto input_desc = node_def->GetInputDesc(i);
        std::string input_name = ScopeUtil::StringReplaceAll(input_desc.GetName(), "^", "");
        if (input_name.find("/batchnorm/") == std::string::npos &&
            input_name.find("/moments/") == std::string::npos) {
          index = i;
          find_scope_external_input = true;
        }
        if (input_name.find("batchnorm/mul_1_grad/Mul_1") != std::string::npos) {
          find_mul1 = true;
        }
        if (input_name.find("batchnorm/mul_2_grad/Mul_1") != std::string::npos) {
          find_mul2 = true;
        }
        if (input_name.find("batchnorm/mul_1_grad/Mul") != std::string::npos) {
          find_mul = true;
        }
      }
      if ((!find_mul1 && !find_mul2 && find_mul && !find_scope_external_input) || (index != -1)) {
        node_def_name = node_def->GetName();
        return;
      }
    }
  }
  return;
}

void ScopeLayerNormGradPass::ProcessInputDy(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  int dy_index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, dy_index, "batchnorm/mul_1_grad/Mul", "batchnorm");
  }
  if (dy_index == 0) {
    fusion_rlt->InsertInputs("batchnorm/mul_1_grad/Mul", {0, kFusionDisableIndex});  // input dy
  } else if (dy_index == 1) {
    fusion_rlt->InsertInputs("batchnorm/mul_1_grad/Mul", {kFusionDisableIndex, 0});  // input dy
  } else {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Only support input dy_index(%d) is 0 or 1", dy_index);
    return;
  }
}

void ScopeLayerNormGradPass::ProcessInputX(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  int x_index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputXIndex(scope, x_index);
  }
  if (x_index == 0) {
    fusion_rlt->InsertInputs("batchnorm/mul_1_grad/Mul_1", {1, kFusionDisableIndex});  // input x
  } else if (x_index == 1) {
    fusion_rlt->InsertInputs("batchnorm/mul_1_grad/Mul_1", {kFusionDisableIndex, 1});  // input x
  } else {
    OP_LOGE(kOpType, "Only support input x_index(%d) is 0 or 1", x_index);
    return;
  }
}

void ScopeLayerNormGradPass::ProcessInputGamma(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  int gamma_index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, gamma_index, "batchnorm/mul_grad/Mul", "gradients");
  }
  if (gamma_index == 0) {
    fusion_rlt->InsertInputs("batchnorm/mul_grad/Mul", {4, kFusionDisableIndex});  // input gamma
  } else if (gamma_index == 1) {
    fusion_rlt->InsertInputs("batchnorm/mul_grad/Mul", {kFusionDisableIndex, 4});  // input gamma
  } else {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Only support input gamma_index(%d) is 0 or 1", gamma_index);
    return;
  }
}

void ScopeLayerNormGradPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  OP_LOGI(kOpType, "Enter ScopeLayerNormGradPass GenerateFusionResult");
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Input fusion_rlt is nullptr.");
    return;
  }
  ProcessInputDy(scopes, fusion_rlt);
  ProcessInputX(scopes, fusion_rlt);

  fusion_rlt->InsertInputs("batchnorm/Rsqrt_grad/RsqrtGrad", {2});  // input variance

  fusion_rlt->InsertInputs("moments/SquaredDifference_grad/sub", {kFusionDisableIndex, 3});  // input mean
  ProcessInputGamma(scopes, fusion_rlt);
  OutputGammaBetaProcess(scopes, fusion_rlt);
  std::string output_dx;
  int add_index = -1;
  FindOutputdXNode(fusion_rlt->Nodes(), output_dx, add_index);
  if (output_dx == "") {
    OP_LOGE(kOpType, "In ScopeLayerNormGradPass, Not find output dX node name.");
    return;
  }
  else {
    OP_LOGI(kOpType, "Output DX name is %s.", output_dx.c_str());
    for (auto& scope : scopes) {
      // The upper call guarantees that the scope is not empty.
      if (scope->SubType() == kScopeType) {
        std::string scope_name = scope->Name();
        OP_LOGI(kOpType, "LayerNormGrad scope name is %s", scope_name.substr(0, scope_name.length() - 1).c_str());
        fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
        break;
      }
    }
    if (add_index != -1) {
    OP_LOGI(kOpType, "multi-to-multi fusion case in");
    if (add_index == 0) {
      fusion_rlt->InsertInputs(output_dx.c_str(), {5, kFusionDisableIndex, kFusionDisableIndex});
    } else if (add_index == 1) {
      fusion_rlt->InsertInputs(output_dx.c_str(), {kFusionDisableIndex, 5, kFusionDisableIndex});
    }
    else {
      fusion_rlt->InsertInputs(output_dx.c_str(), {kFusionDisableIndex, kFusionDisableIndex, 5});
    }
    fusion_rlt->InsertOutputs(output_dx.c_str(), {0});
    fusion_rlt->SetType(kScopeToMultiNodes);

    auto in_layernorm_grad_node = fusion_rlt->AddInnerNode("in_layernorm_grad_node", "LayerNormGrad");
    CHECK_INNER_NODE_CONDITION(in_layernorm_grad_node != nullptr, fusion_rlt);
    Status ret = in_layernorm_grad_node->InsertInput(kInputFromFusionScope, 0)
      .InsertInput(kInputFromFusionScope, 1)
      .InsertInput(kInputFromFusionScope, 2)
      .InsertInput(kInputFromFusionScope, 3)
      .InsertInput(kInputFromFusionScope, 4)
      .InsertOutput("in_add_node", 0)
      .InsertOutput(kOutputToFusionScope, 1)
      .InsertOutput(kOutputToFusionScope, 2)
      .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

    auto in_add_node = fusion_rlt->AddInnerNode("in_add_node", "Add");
    CHECK_INNER_NODE_CONDITION(in_add_node != nullptr, fusion_rlt);
    ret = in_add_node->InsertInput("in_layernorm_grad_node", 0)
      .InsertInput(kInputFromFusionScope, 5)
      .InsertOutput(kOutputToFusionScope, 0)
      .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    }
    if (add_index == -1) {
    OP_LOGI(kOpType, "single-to-single fusion case in");
    fusion_rlt->InsertOutputs(output_dx.c_str(), {0});
    fusion_rlt->SetType(kScopeRltType);
    }
  }
  fusion_rlt->SetDescription("");
  OP_LOGI(kOpType, "LayerNormGrad Scope fusion success.");
  return;
}
}  // namespace ge
