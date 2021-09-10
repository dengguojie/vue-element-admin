/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file scope_layernorm_pass.cc
 * \brief
 */
#include "scope_layernorm_pass.h"

#include "op_log.h"
#include "register/scope/scope_fusion_pass_register.h"

namespace ge {
namespace {
const char* const kScopeTypeBatchnorm = "batchnorm";
const char* const kScopeTypeMoments = "moments";
const char* const kScopeRltType = "LayerNorm";
const char* const kOpType = "LayerNorm";
}  // namespace

std::vector<ScopeFusionPatterns> ScopeLayerNormPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns patterns_batchnorm;
  ScopeFusionPatterns patterns_moments;
  GenBatchnormScopePatterns(patterns_batchnorm);  // match batchnorm Scope
  GenMomentsScopePatterns(patterns_moments);      // match moments Scope
  patterns_list.push_back(patterns_batchnorm);
  patterns_list.push_back(patterns_moments);
  return patterns_list;
}

std::string ScopeLayerNormPass::PassName() {
  return std::string("ScopeLayerNormPass");
}

/**
 * @brief LastMatch for multiple scopes
 */
Status ScopeLayerNormPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
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
  std::vector<Scope*> fusion_scopes_bn;
  std::vector<Scope*> fusion_scopes_m;
  for (auto& scope : scopes) {
    // uwc can not do scope fusion
    if (scope->Name().find("sequence_encoder_loop/while") != std::string::npos) {
      return FAILED;
    }
    const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map_mul = scope->AllNodesMap();
    for (auto& it : nodes_map_mul) {
        auto mode_def = it.second;
        if(mode_def->GetName().find("discriminator_2/d_ln3/LayerNorm/batchnorm/mul") != std::string::npos) {
            OP_LOGI(kOpType, "discriminator_2/d_ln3/LayerNorm/batchnorm/mul name is found");
            std::vector<std::string> outputs;
            mode_def->GetAttr("_origin_graph_node_outputs", outputs);
            OP_LOGI(kOpType, "layernorm data_out_num is %zu", outputs.size());
            if (outputs.size() >= 6) {
                OP_LOGI(kOpType, "output size is 6, not scope");
                return FAILED;
            }
        }
        if (mode_def->GetName().find("moments/variance") != std::string::npos){
          if (mode_def->GetName().find("generator/ResBlock") != std::string::npos) {
            OP_LOGI(kOpType, "generator/ResBlock is found, %s", mode_def->GetName().c_str());
            return FAILED;
          }
          if (mode_def->GetName().find("generator/BN/moments/variance") != std::string::npos) {
            OP_LOGI(kOpType, "generator/BN/moments/variance is found %s", mode_def->GetName().c_str());
            return FAILED;
          }
        }
        if (mode_def->GetName().find("LayerNorm/batchnorm/Rsqrt") != std::string::npos &&
        mode_def->GetOpType() == "Rsqrt") {
          OP_LOGI(kOpType, "LayerNorm/batchnorm/Rsqrt is found, name is %s", mode_def->GetName().c_str());
          std::vector<std::string> outputs_sqrt;
          mode_def->GetAttr("_origin_graph_node_outputs", outputs_sqrt);
          bool found_grad_mul = false;
          for (size_t i = 0; i < outputs_sqrt.size(); i++) {
            std::string output_name = outputs_sqrt.at(i);
            std::string split_name = output_name.substr(1);
            OP_LOGI(kOpType, "split name is %s", split_name.c_str());
            if (split_name.find("mul_grad/Mul_1") != std::string::npos) {
              found_grad_mul = true;
              break;
            }
          }
          if (not found_grad_mul) {
            OP_LOGI(kOpType, " not find mul_grad/Mul_1");
            return FAILED;
          }
        }
    }
    // tf2.x can not do scope fusion
    if (scope->Name().find("batchnorm/mul") != std::string::npos) {
      const std::unordered_map<std::string, ge::OperatorPtr>& nodes_map = scope->AllNodesMap();
      for (auto& it : nodes_map) {
        OP_LOGI(kOpType, "batchnorm/mul Scope has node %s.", it.second->GetName().c_str());
        if (it.second->GetName().find("mul/ReadVariableOp") != std::string::npos) {
          return FAILED;
        }
      }
    }
    // Class ScopeTree guarantees scope is not empty.
    if ((scope->SubType() == kScopeTypeBatchnorm)) {
      fusion_scopes_bn.push_back(scope);
    } else if (scope->SubType() == kScopeTypeMoments) {
      fusion_scopes_m.push_back(scope);
    }
  }

  if (fusion_scopes_bn.size() == fusion_scopes_m.size()) {
    // the two scope batchnorm and moments in the same layernorm
    for (size_t i = 0; i < fusion_scopes_bn.size(); i++) {
      auto scope_bn = fusion_scopes_bn[i];
      std::string scope_bn_name = scope_bn->Name();
      int is_while = scope_bn_name.find("while");
      if (is_while != -1) {
        OP_LOGI(kOpType, "break  scope_bn_name is %s.", scope_bn_name.c_str());
        continue;
      }
      for (size_t j = 0; j < fusion_scopes_m.size(); j++) {
        auto scope_m = fusion_scopes_m[j];
        std::string scope_m_name = scope_m->Name();
        int pos_bn = scope_bn_name.find("batchnorm");
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

void ScopeLayerNormPass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }

  int index1 = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, index1, "batchnorm/mul_1", "batchnorm");
  }
  if (index1 == 0) {
    fusion_rlt->InsertInputs("batchnorm/mul_1", {0, kFusionDisableIndex});
  } else if (index1 == 1) {
    fusion_rlt->InsertInputs("batchnorm/mul_1", {kFusionDisableIndex, 0});  // input x
  } else {
    OP_LOGE(kOpType, "Only support input xIndex(%d) is 0 or 1", index1);
    return;
  }

  int index = kFusionDisableIndex;
  for (auto& scope : scopes) {
    FindInputIndex(scope, index, "batchnorm/mul", "batchnorm");
  }
  if (index == 0) {
    fusion_rlt->InsertInputs("batchnorm/mul", {1, kFusionDisableIndex});
  } else if (index == 1) {
    fusion_rlt->InsertInputs("batchnorm/mul", {kFusionDisableIndex, 1});  // input gamma
  } else {
    OP_LOGE(kOpType, "Only support input gammaIndex(%d) is 0 or 1", index);
    return;
  }

  fusion_rlt->InsertInputs("batchnorm/sub", {2, kFusionDisableIndex});  // input beta
  fusion_rlt->InsertOutputs("batchnorm/add_1", {0});                    // output y
  fusion_rlt->InsertOutputs("moments/mean", {1});                       // output mean
  fusion_rlt->InsertOutputs("batchnorm/Rsqrt", {2});                    // output variance

  for (auto& scope : scopes) {
    // The upper call guarantees that the scope is not empty.
    if (scope->SubType() == kScopeTypeBatchnorm) {
      std::string scope_bn_name = scope->Name();
      int pos_bn = scope_bn_name.find("batchnorm");
      if (pos_bn != -1) {
        fusion_rlt->SetName(scope_bn_name.substr(0, pos_bn));
        break;
      }
    }
  }

  fusion_rlt->SetType(kScopeRltType);  // op Type LayerNorm
  fusion_rlt->SetDescription("");

  OP_LOGI(kOpType, "LayerNorm Scope fusion success.");
  return;
}

void ScopeLayerNormPass::GenBatchnormScopePatterns(ScopeFusionPatterns& patterns) {
  // match batchnorm
  std::vector<ScopePattern*> batch;
  ScopePattern* batch_norm_cell = new (std::nothrow) ScopePattern();
  if (batch_norm_cell == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  batch_norm_cell->SetSubType(kScopeTypeBatchnorm);

  batch_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Sub", 1));        // Sub num is 1
  batch_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Rsqrt", 1));      // Rsqrt num is 1
  batch_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 3));        // Mul num is 3
  batch_norm_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Identity", -1));  // Identity num is -1
  batch_norm_cell->AddScopeFeature(ScopeFeature("", -1, "batchnorm"));

  batch.push_back(batch_norm_cell);
  patterns.push_back(batch);
}

void ScopeLayerNormPass::GenMomentsScopePatterns(ScopeFusionPatterns& patterns) {
  // match moments
  std::vector<ScopePattern*> batch;
  ScopePattern* moments_cell = new (std::nothrow) ScopePattern();
  if (moments_cell == nullptr) {
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }

  moments_cell->SetSubType(kScopeTypeMoments);
  moments_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Mean", 2));               // Mean num is 2
  moments_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("SquaredDifference", 1));  // SquaredDifference num is 1
  moments_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("Squeeze", -1));           // Squeeze num is 0
  moments_cell->AddNodeOpTypeFeature(NodeOpTypeFeature("StopGradient", -1));
  moments_cell->AddScopeFeature(ScopeFeature("", -1, "moments"));

  batch.push_back(moments_cell);
  patterns.push_back(batch);
}

void ScopeLayerNormPass::FindInputIndex(const Scope* scope, int& index, const std::string& name,
                                        const std::string& base_name) {
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
    if (mul_input_name.find("batchnorm/mul/Enter") != string::npos) {
      index = i;
      OP_LOGI(kOpType, "found batchnorm/mul/Enter, the index is %d", i);
      return;
    }
    if (mul_input_name.find("batchnorm/add_1") != string::npos) {
      index = i;
      OP_LOGI(kOpType, "found batchnorm/add_1, the index is %d", i);
      return;
    }
  }
}
}  // namespace ge
