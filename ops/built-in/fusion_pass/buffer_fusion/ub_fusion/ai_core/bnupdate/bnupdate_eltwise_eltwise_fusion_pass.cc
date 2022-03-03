/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#include "bnupdate_eltwise_eltwise_fusion_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;
namespace {
static const char kPatternBnupdate[] = "bnupdate";
static const char kPatternEltwise0[] = "eltwise0";
static const char kPatternEltwise1[] = "eltwise1";
static const char kPatternOutput0[] = "Output0";
static const char kPatternOutput1[] = "Output1";
static const char kPatternOutput2[] = "Output2";
static const char kPatternOutput3[] = "Output3";
static const char kPatternOutput4[] = "Output4";
static const char kPatternOutput5[] = "Output5";
static const char kPatternOtherInput[] = "otherInput";
const int kNodeOutputSize = 2;
const int kNodeInputSize = 2;
static const string kFusedOpType = "FusedOp";
}

/*
 * @brief:  define_bn_update and Elemwise and Elemwiseinput op fusion pattern
 *
 *   BnUpdate and Elemwise and Elemwise
 *
 * fusion node: BnUpdate and Elemwise and Elemwise
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> BnupdateEltwiseEltwiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "BNUpdateEltwiseEltwiseFusionPass";

  int max_count = 7;
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name, TBE_FUSION_OP_NUM_MAX + max_count);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(kFusedOpType.c_str(), "New an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());

  pattern->AddOpDesc(kPatternBnupdate, {OP_PATTERN_BNUPDATE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternEltwise0, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternEltwise1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput0, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput3, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput4, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput5, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternBnupdate})
      .SetOutputs(kPatternBnupdate, {kPatternEltwise0, kPatternOutput0, kPatternOutput1, kPatternOutput2,
                                     kPatternOutput3, kPatternOutput4, kPatternOutput5},
                  TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternEltwise0, {kPatternEltwise1})
      .SetOutputs(kPatternOtherInput, {kPatternEltwise0});

  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}

std::vector<ge::NodePtr> BnupdateEltwiseEltwiseFusionPass::GetMatchedNodes(
  const BufferFusionMapping &mapping, ge::NodePtr &eltwise_0, ge::NodePtr &bn) {
    std::vector<ge::NodePtr> nodes;
    for (const auto &item : mapping) {
      if (item.first->desc_name == kPatternEltwise0) {
        eltwise_0 = item.second[0];
      }
      if (item.first->desc_name == kPatternBnupdate) {
        bn = item.second[0];
      }
      for (const auto &node : item.second) {
        nodes.push_back(node);
      }
    }
    return nodes;
  }
/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status BnupdateEltwiseEltwiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                        vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do BNUpdateEltwiseEltwiseFusionPass!");
  for (auto &item : mapping) {
    auto elem_opdesc = find(item.first->types.begin(), item.first->types.end(), OP_PATTERN_ELEMWISE);
    if (elem_opdesc == item.first->types.end()) {
      continue;
    }
    for (auto &node : item.second) {
      if (node->GetOpDesc()->GetAllOutputsDesc().size() != kNodeOutputSize &&
          node->GetOpDesc()->GetAllInputsDesc().size() != kNodeInputSize) {
        OP_LOGD(kFusedOpType.c_str(), "The number of node[%s] output is [%zu], "
                "which is not equal to two, no need to do fusion.",
                node->GetName().c_str(), node->GetOpDesc()->GetAllOutputsDesc().size());
        return SUCCESS;
      }
    }
  }

  ge::NodePtr bn;
  ge::NodePtr eltwise0;
  fusion_nodes = GetMatchedNodes(mapping, eltwise0, bn);
  for (auto &item : mapping) {
    auto opdesc = find(item.first->types.begin(), item.first->types.end(), TBE_PATTERN_OUTPUT_NODE);
    if (opdesc != item.first->types.end()) {
      for (auto &node : item.second) {
        auto node_ptr = find(fusion_nodes.begin(), fusion_nodes.end(), node);
        if (node_ptr != fusion_nodes.end()) {
          fusion_nodes.erase(node_ptr);
        }
      }
    }
  }

  ge::NodePtr second_bn = nullptr;
  for (auto input_node : eltwise0->GetInDataNodes()) {
    if (input_node == bn) {
      continue;
    }
    if (input_node->GetType() == "BNTrainingUpdate") {
      second_bn = input_node;
    }
  }
  if (second_bn != nullptr) {
    fusion_nodes.emplace_back(second_bn);
  }

  OP_LOGD(kFusedOpType.c_str(), "End to do BNUpdateEltwiseEltwiseFusionPass!");
  return SUCCESS;
}
}  // namespace fe
