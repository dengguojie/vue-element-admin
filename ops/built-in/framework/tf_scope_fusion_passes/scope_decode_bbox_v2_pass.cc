/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file scope_decode_bbox_v2_pass.cc
 * \brief
 */
#include "scope_decode_bbox_v2_pass.h"

#include "op_log.h"


namespace ge {
namespace {
const char* const kScopeType = "DecodeBboxV2";
const char* const kScopeTypeDecodeBboxV2 = "DecodeBboxV2";
const char* const kScopeTypeDecodeBboxV2SSD = "DecodeBboxV2SSD";
const char* const kOpType = "DecodeBboxV2";
}  // namespace

std::vector<ScopeFusionPatterns> ScopeDecodeBboxV2Pass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns_list;
  ScopeFusionPatterns pattern;
  GenScopePatterns(pattern);  // match decode Scope
  patterns_list.push_back(pattern);
  return patterns_list;
}

std::string ScopeDecodeBboxV2Pass::PassName() {
  return std::string("ScopeDecodeBboxV2Pass");
}

/**
 * @brief LastMatch for multiple scopes
 */
Status ScopeDecodeBboxV2Pass::LastMatchScopesAndOPs(shared_ptr<ScopeGraph>& scope_graph,
                                                    std::vector<ScopesResult>& results) {
  OP_LOGI(kOpType, "LastMatchScopesAndOPs start.");
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

  for (auto& scope : scopes) {
    // Class ScopeTree guarantees scope is not empty.
    if (scope->SubType() == kScopeTypeDecodeBboxV2) {
      OP_LOGI(kOpType, "DecodeBbox LastMatchScopesAndOPs match SubType %s.", scope->Name().c_str());
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
    if (scope->SubType() == kScopeTypeDecodeBboxV2SSD) {
      OP_LOGI(kOpType, "DecodeBboxSSD LastMatchScopesAndOPs match SubType %s.", scope->Name().c_str());
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(scope);
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeDecodeBboxV2Pass::GenScopePatterns(ScopeFusionPatterns& patterns) {
  // match decodebboxv2
  std::vector<ScopePattern*> batch;
  ScopePattern* decodeBboxV2Pattern = new (std::nothrow) ScopePattern();
  if (decodeBboxV2Pattern == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  decodeBboxV2Pattern->SetSubType(kScopeTypeDecodeBboxV2);
  decodeBboxV2Pattern->AddScopeFeature(ScopeFeature("", 1, "Decode"));
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Exp", 2, 0));        // Exp num is 2
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 4, 0));        // Mul num is 4
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Sub", 4, 0));        // Sub num is 4
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("RealDiv", 0, 2));    // RealDiv num is 2*n
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Unpack", 2, 0));     // Unpack num is 2
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Pack", 1, 0));       // Pack num is 1
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 3, 0));  // Transpose num is 3
  decodeBboxV2Pattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Softmax", -1, 0));   // don't have Softmax

  OP_LOGI(kOpType, "Add GenScopePatterns DecodeBboxV2.");
  batch.push_back(decodeBboxV2Pattern);

  ScopePattern* decodeBboxV2SSDPattern = new (std::nothrow) ScopePattern();
  if (decodeBboxV2SSDPattern == nullptr) {
    ScopeUtil::FreeScopePatterns(patterns);
    ScopeUtil::FreeOneBatchPattern(batch);
    OP_LOGE(kOpType, "Alloc an object failed.");
    return;
  }
  decodeBboxV2SSDPattern->SetSubType(kScopeTypeDecodeBboxV2SSD);
  decodeBboxV2SSDPattern->AddScopeFeature(ScopeFeature("", 1, "Decode"));
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Exp", 2, 0));        // Exp num is 2
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Mul", 4, 0));        // Mul num is 4
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Sub", 10, 0));       // Sub num is 10
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("RealDiv", 0, 2));    // RealDiv num 2*n
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Unpack", 2, 0));     // Unpack num is 2
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Pack", 1, 0));       // Pack num is 1
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Transpose", 3, 0));  // Transpose num is 3
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Rank", 3, 0));       // Transpose num is 3
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Range", 3, 0));      // Transpose num is 3
  decodeBboxV2SSDPattern->AddNodeOpTypeFeature(NodeOpTypeFeature("Sigmoid", -1, 0));   // don't have Sigmoid

  OP_LOGI(kOpType, "Add GenScopePatterns DecodeBboxV2.");
  batch.push_back(decodeBboxV2SSDPattern);

  patterns.push_back(batch);
}

void ScopeDecodeBboxV2Pass::GenerateFusionResult(const std::vector<Scope*>& scopes, FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType, "Input fusion_rlt is nullptr.");
    return;
  }
  for (auto& scope : scopes) {
    if (scope->SubType() == kScopeTypeDecodeBboxV2) {
      OP_LOGI(kOpType, "fusion type DecodeBboxV2 scope only has 3 transpose nodes.");
      fusion_rlt->InsertInputs("transpose", {0, kFusionDisableIndex});  // Input index 0 of decode_bbox
      fusion_rlt->InsertInputs("get_center_coordinates_and_sizes/transpose", {1, kFusionDisableIndex});
      fusion_rlt->InsertOutputs("transpose_1", {0});  // Output index 0 of decode_bbox
      fusion_rlt->SetType(kScopeType);
      std::string scope_name = scope->Name();
      fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
    }
    if (scope->SubType() == kScopeTypeDecodeBboxV2SSD) {
      OP_LOGI(kOpType, "fusion type DecodeBboxV2 scope  has 3 transpose scopes.");
      fusion_rlt->InsertInputs("transpose/Rank", {0, kFusionDisableIndex});  // Input index 0 of decode_bbox
      fusion_rlt->InsertInputs("get_center_coordinates_and_sizes/transpose/Rank", {1, kFusionDisableIndex});
      fusion_rlt->InsertOutputs("transpose_1", {0});  // Output index 0 of decode_bbox
      fusion_rlt->SetType(kScopeType);
      std::string scope_name = scope->Name();
      fusion_rlt->SetName(scope_name.substr(0, scope_name.length() - 1));
    }
  }
  return;
}
}  // namespace ge
