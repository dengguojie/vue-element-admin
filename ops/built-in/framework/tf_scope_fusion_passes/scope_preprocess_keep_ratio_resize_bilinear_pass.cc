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
 * \file scope_preprocess_keep_ratio_resize_bilinear_pass.cc
 * \brief scope fusion of KeepRatioResizeBilinear
 *
 * function info:
 * Preprocessor/sub    second_output(batch, 3)   3 = [h, w, c]
 *            \        /
 *             \      /
 *         Preprocessor/map
 *             |
 *             |
 *           input
 * =========================================================================
 *          to :
 *  Preprocessor/sub     second_output(batch, 3)   3 = [h, w, c]
 *      |                    |
 *      |                    |
 *      |                  Tile
 *      |               /         \
 *      |              /           \
 *      |       [1,3]  |           |[2]
 *      |        ExpandDims      ConcatV2
 *       \           \[3]        /[1]   \[1]
 *        \         Slice     Slice     Const(1)
 *         \           \  [4] /
 *          \            Shape
 *           \           /
 *            \         /
 *             \       /
 *              \     /
 *      KeepRatioResizeBilinear
 *                 |
 *                 |
 *               input
 */
#include "scope_preprocess_keep_ratio_resize_bilinear_pass.h"

#include <cmath>

#include "op_log.h"
#include "register/register.h"

namespace ge {

std::vector<ScopeFusionPatterns> ScopeKeepRatioResizeBilinearPass::DefinePatterns() {
  std::vector<ScopeFusionPatterns> patterns;
  patterns.push_back(GenWhileScopePatterns());
  return patterns;
}

Status ScopeKeepRatioResizeBilinearPass::LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph>& scope_graph,
                                                               std::vector<ScopesResult>& results) {
  if (!scope_graph) {
    OP_LOGE(kOpType.c_str(), "Input params is nullptr.");
    return domi::PARAM_INVALID;
  }

  const ScopeTree* scope_tree = scope_graph->GetScopeTree();
  if (scope_tree == nullptr) {
    OP_LOGE(kOpType.c_str(), "Scope tree is nullptr.");
    return domi::PARAM_INVALID;
  }

  std::vector<Scope*> scopes = scope_tree->GetAllScopes();
  for (auto& scope : scopes) {
    if (scope->Name() == "rootmap/") {
      continue;
    }
    if (scope->LastName() == kScopeMap) {
      const Scope* fatherScope = scope;
      if (fatherScope == nullptr) {
        continue;
      }

      if (!MatchedSubScopes(fatherScope, {"while/", "ResizeToRange/"})) {
        continue;
      }
      OP_LOGI(kOpType.c_str(), "find scope name: %s", fatherScope->Name().c_str());
      ScopesResult result;
      std::vector<Scope*> result_scopes;
      result_scopes.push_back(const_cast<Scope*>(fatherScope));
      result.SetScopes(result_scopes);
      results.push_back(result);
    }
  }

  return (!(results.empty())) ? SUCCESS : FAILED;
}

void ScopeKeepRatioResizeBilinearPass::GenerateFusionResult(const std::vector<Scope*>& scopes,
                                                            FusionScopesResult* fusion_rlt) {
  if (fusion_rlt == nullptr) {
    OP_LOGE(kOpType.c_str(), "Input fusion_rlt is nullptr.");
    return;
  }

  OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass begin do GenerateFusionResult");
  std::string minDivNodeName = "while/ResizeToRange/Const";
  std::string maxDivNodeName = "while/ResizeToRange/Const_1";
  std::string resizeType = "ResizeBilinear";
  for (auto& scope : scopes) {
    OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass do scope for %s", scope->Name().c_str());
    std::string minConstNode = "";
    std::string maxConstNode = "";
    int32_t minDims = 0;
    int32_t maxDims = 0;
    bool alignCorners = false;
    bool halfPixelCenters = false;
    // do insert input 0 boxes like map/TensorArrayUnstack/Shape
    std::string scopeInputName0 = scope->Name() + "Shape";
    fusion_rlt->InsertInputs(scopeInputName0, {0});
    // do insert outputs
    if (MatchedSubScopes(scope, {"TensorArrayStack/"})) {
      std::string scopeOutputName0 = scope->Name() + "TensorArrayStack/TensorArrayGatherV3";
      fusion_rlt->InsertOutputs(scopeOutputName0, {0});
    }
    if (MatchedSubScopes(scope, {"TensorArrayStack_1/"})) {
      std::string scopeOutputName0 = scope->Name() + "TensorArrayStack_1/TensorArrayGatherV3";
      fusion_rlt->InsertOutputs(scopeOutputName0, {1});
    }

    // do parser for scope
    auto nodesMap = scope->AllNodesMap();
    for (auto& it : nodesMap) {
      auto nodeDef = it.second;
      if (nodeDef == nullptr) {
        continue;
      }
      std::string nodeName = nodeDef->GetName().c_str();
      if ((minConstNode == "") && (nodeName.find(minDivNodeName) != std::string::npos) &&
          (nodeName.find(maxDivNodeName) == std::string::npos)) {
        minConstNode = nodeName;
        Tensor data;
        nodeDef->GetAttr("value", data);
        float minDimsFloat = *reinterpret_cast<float*>(data.GetData());
        minDims = round(minDimsFloat);
        OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass get min_dimension = %d", minDims);
      }
      if ((maxConstNode == "") && (nodeName.find(maxDivNodeName) != std::string::npos)) {
        maxConstNode = nodeName;
        Tensor data;
        nodeDef->GetAttr("value", data);
        float maxDimsFloat = *reinterpret_cast<float*>(data.GetData());
        maxDims = round(maxDimsFloat);
        OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass get max_dimension = %d", maxDims);
      }
      if (nodeDef->GetOpType() == resizeType) {
        nodeDef->GetAttr("align_corners", alignCorners);
      }
    }

    fusion_rlt->SetType(kScopeToMultiNodes);
    fusion_rlt->SetDescription("");
    std::string scopeName = scope->Name();
    fusion_rlt->SetName(scopeName.substr(0, scopeName.length() - 1));

    // add inner node
    OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass add KeepRatioResizeBilinear begin");
    auto resizeNode = fusion_rlt->AddInnerNode("KeepRatioResizeBilinear_node", "KeepRatioResizeBilinear");
    CHECK_INNER_NODE_CONDITION(resizeNode != nullptr, fusion_rlt);
    auto ret = resizeNode->InsertInput(kInputFromFusionScope, 0)
                   .InsertOutput(kOutputToFusionScope, 0)
                   .InsertOutput("shape_node", 0)
                   .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    CHECK_INNER_NODE_CONDITION(resizeNode->MutableOperator() != nullptr, fusion_rlt);
    resizeNode->SetInputFormat("images", "NHWC");
    resizeNode->SetOutputFormat("y", "NHWC");
    resizeNode->MutableOperator()->SetAttr("align_corners", alignCorners);
    resizeNode->MutableOperator()->SetAttr("half_pixel_centers", halfPixelCenters);
    resizeNode->MutableOperator()->SetAttr("min_dimension", minDims);
    resizeNode->MutableOperator()->SetAttr("max_dimension", maxDims);

    OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass add KeepRatioResizeBilinear end");

    OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass add sencond edge begin");
    auto shapeNode = fusion_rlt->AddInnerNode("shape_node", "Shape");
    CHECK_INNER_NODE_CONDITION(shapeNode != nullptr, fusion_rlt);
    ret = shapeNode->InsertInput("KeepRatioResizeBilinear_node", 0)
              .InsertOutput("split_node", 0)
              .InsertOutput("batch_slice_node", 0)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

    auto sliceNode = fusion_rlt->AddInnerNode("slice_node", "Slice");
    CHECK_INNER_NODE_CONDITION(sliceNode != nullptr, fusion_rlt);
    ret = sliceNode->InsertInput("shape_node", 0)
              .InsertInput("const_begin_node", 0)
              .InsertInput("const_size_node", 0)
              .InsertOutput("expand_dims_1", 0)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

    auto constBeginNode = fusion_rlt->AddInnerNode("const_begin_node", "Const");
    CHECK_INNER_NODE_CONDITION(constBeginNode != nullptr, fusion_rlt);
    ret = constBeginNode->InsertOutput("slice_node", 1)
              .InsertOutput("batch_slice_node", 2)
              .InsertOutput("concat_batch_node", 0)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    int32_t* beginData = nullptr;
    beginData = new(std::nothrow) int32_t[1];
    if (beginData == nullptr) {
      OP_LOGE(kOpType.c_str(), "Scope apply beginData is nullptr.");
      return;
    }
    *(beginData) = 1;
    TensorDesc beginDesc(ge::Shape({1}), FORMAT_ND, DT_INT32);
    Tensor beginTensor(beginDesc, (uint8_t*)beginData, sizeof(int32_t));
    constBeginNode->MutableOperator()->SetAttr("value", beginTensor);
    delete[] beginData;
    beginData = nullptr;

    auto constSizeNode = fusion_rlt->AddInnerNode("const_size_node", "Const");
    CHECK_INNER_NODE_CONDITION(constSizeNode != nullptr, fusion_rlt);
    ret = constSizeNode->InsertOutput("slice_node", 2).BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    int32_t* sizeData = nullptr;
    sizeData = new(std::nothrow) int32_t[1];
    if (sizeData == nullptr) {
      OP_LOGE(kOpType.c_str(), "Scope apply sizeData is nullptr.");
      return;
    }
    *(sizeData) = 3;
    TensorDesc sizeDesc(ge::Shape({1}), FORMAT_ND, DT_INT32);
    Tensor sizeTensor(sizeDesc, (uint8_t*)sizeData, sizeof(int32_t));
    constSizeNode->MutableOperator()->SetAttr("value", sizeTensor);
    delete[] sizeData;
    sizeData = nullptr;

    auto sliceBatchNode = fusion_rlt->AddInnerNode("batch_slice_node", "Slice");
    CHECK_INNER_NODE_CONDITION(sliceBatchNode != nullptr, fusion_rlt);
    ret = sliceBatchNode->InsertInput("shape_node", 0)
              .InsertInput("const_batch_begin_node", 0)
              .InsertInput("const_begin_node", 0)
              .InsertOutput("concat_batch_node", 0)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

    auto constBatchNode = fusion_rlt->AddInnerNode("const_batch_begin_node", "Const");
    CHECK_INNER_NODE_CONDITION(constBatchNode != nullptr, fusion_rlt);
    ret = constBatchNode->InsertOutput("batch_slice_node", 1)
              .InsertOutput("expand_dims_1", 1)
              .InsertOutput("concat_batch_node", 1)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    TensorDesc batchDesc(ge::Shape({1}), FORMAT_ND, DT_INT32);
    int32_t* beginBatchData = nullptr;
    beginBatchData = new(std::nothrow) int32_t[1];
    if (beginBatchData == nullptr) {
      OP_LOGE(kOpType.c_str(), "Scope apply beginBatchData is nullptr.");
      return;
    }
    *(beginBatchData) = 0;
    Tensor batchTensor(batchDesc, (uint8_t*)beginBatchData, sizeof(int32_t));
    auto constBatchNodeOp = constBatchNode->MutableOperator();
    if (constBatchNodeOp != nullptr) {
        constBatchNodeOp->SetAttr("value", batchTensor);
    }
    delete[] beginBatchData;
    beginBatchData = nullptr;

    // expand_dims (3) to (1,3)
    auto expandDimsNode1 = fusion_rlt->AddInnerNode("expand_dims_1", "ExpandDims");
    ret = expandDimsNode1->InsertInput("slice_node", 0)
              .InsertInput("const_batch_begin_node", 0)
              .InsertOutput("tile_hw_node", 0)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

    // concat (1) to (2)  value = [batch_size, 1]
    auto concatNode = fusion_rlt->AddInnerNode("concat_batch_node", "ConcatV2");
    ret = concatNode->InsertInput("batch_slice_node", 0)
              .InsertInput("const_begin_node", 0)
              .InsertInput("const_batch_begin_node", 0)
              .InsertOutput("tile_hw_node", 1)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);

    std::vector<domi::DynamicInputOutputInfo> dynamic_name_attr_value;
    domi::DynamicInputOutputInfo dyn_info;
    dyn_info.type = domi::kInput;
    dyn_info.port_name = "x";
    dyn_info.port_name_len = 1;
    dyn_info.attr_name = "N";
    dyn_info.attr_name_len = 1;
    dynamic_name_attr_value.emplace_back(dyn_info);
    Operator op_src(concatNode->GetName(), concatNode->GetType());
    int dyn_num = 2;
    op_src.SetAttr("N", dyn_num);
    CHECK_INNER_NODE_CONDITION(concatNode->MutableOperator() != nullptr, fusion_rlt);
    AutoMappingByOpFnDynamic(op_src, *(concatNode->MutableOperator()), dynamic_name_attr_value);
    op_src.BreakConnect();

    // tile to (batch,3)
    auto tileNode = fusion_rlt->AddInnerNode("tile_hw_node", "Tile");
    ret = tileNode->InsertInput("expand_dims_1", 0)
              .InsertInput("concat_batch_node", 0)
              .InsertOutput(kOutputToFusionScope, 1)
              .BuildInnerNode();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
    OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass add second edge end");

    ret = fusion_rlt->CheckInnerNodesInfo();
    CHECK_INNER_NODE_CONDITION(ret == ge::GRAPH_SUCCESS, fusion_rlt);
  }
  OP_LOGI(kOpType.c_str(), "ScopeKeepRatioResizeBilinearPass end do GenerateFusionResult");
  return;
}

ScopeFusionPatterns ScopeKeepRatioResizeBilinearPass::GenWhileScopePatterns() {
  ScopePattern* while_cell = new (std::nothrow) ScopePattern();
  if (while_cell == nullptr) {
    OP_LOGE(kOpType.c_str(), "Alloc an object failed.");
    return ScopeFusionPatterns();
  }

  while_cell->SetSubType(kScopeResultType);
  while_cell->AddScopeFeature(ScopeFeature("", 1, "", "map"));
  while_cell->AddScopeFeature(ScopeFeature("", 1, "", "while"));
  while_cell->AddScopeFeature(ScopeFeature("", 1, "", "ResizeToRange"));

  ScopeFusionPatterns while_scope_pattern = {{while_cell}};
  return while_scope_pattern;
}

string ScopeKeepRatioResizeBilinearPass::to_string(const std::vector<Scope*>& scopes) const {
  string result;
  for (auto& scope : scopes) {
    result += scope->Name();
    result += " ";
  }
  return result;
}

bool ScopeKeepRatioResizeBilinearPass::MatchedSubScopes(const Scope* root_scope,
                                                        const std::vector<string>& scopes2check) const {
  string full_name;
  auto root = root_scope;
  for (auto& scope_name : scopes2check) {
    full_name = root->Name();
    full_name += scope_name;
    auto sub_scope = root->GetSubScope(full_name);
    if (sub_scope == nullptr) {
      OP_LOGI(kOpType.c_str(), "Get sub scope:%s failed, %s's sub scopes:%s", full_name.c_str(), root->Name().c_str(),
              to_string(root->GetAllSubScopes()).c_str());
      return false;
    }
    root = sub_scope;
  }

  OP_LOGI(kOpType.c_str(), "MatchedSubScopes:%s success.", root_scope->Name().c_str());
  return true;
}

}  // namespace ge
