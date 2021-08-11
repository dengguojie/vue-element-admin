/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "bnupdate_eltwise_fusion_pass.h"
#include <string>
#include <vector>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;
namespace {
static const char kPatternBnupdate[] = "bnupdate";
static const char kPatternOutput0[] = "OUTPUT0";
static const char kPatternOutput1[] = "OUTPUT1";
static const char kPatternOutput2[] = "OUTPUT2";
static const char kPatternOutput3[] = "OUTPUT3";
static const char kPatternOutput4[] = "OUTPUT4";
static const char kPatternOutput5[] = "OUTPUT5";
static const char kPatternEltwise[] = "eltwise";
static const string kFusedOpType = "FusedOp";
}

/*
 * @brief:  define BNUpadte and ElemWise input op fusion pattern
 *
 *   BNReduce + ElemWise
 *
 * fusion node:  BNReduce, ElemWise
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> BnupdateEltwiseFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;
  string pass_name = "BNUpdateEltwiseFusionPass";

  int max_count = 6;
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(pass_name, TBE_FUSION_OP_NUM_MAX + max_count);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(kFusedOpType.c_str(), "New an object failed."), return patterns);
  OP_LOGD(kFusedOpType.c_str(), "Start to define %s pass pattern.", pass_name.c_str());

  pattern->AddOpDesc(kPatternBnupdate, {OP_PATTERN_BNUPDATE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternEltwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput0, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput1, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput2, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput3, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput4, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOutput5, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternBnupdate})
      .SetOutputs(kPatternBnupdate, {kPatternEltwise, kPatternOutput0, kPatternOutput1, kPatternOutput2,
                                     kPatternOutput3, kPatternOutput4, kPatternOutput5},
                  TBE_OUTPUT_BRANCH_MULTI);

  patterns.push_back(pattern);
  OP_LOGD(kFusedOpType.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status BnupdateEltwiseFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                    vector<ge::NodePtr> &fusion_nodes) {
  OP_LOGD(kFusedOpType.c_str(), "Begin to do BNUpdateEltwiseFusionPass!");

  fusion_nodes = GetMatchedNodes(mapping);
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

  OP_LOGD(kFusedOpType.c_str(), "End to do BNUpdateEltwiseFusionPass!");
  return SUCCESS;
}
}  // namespace fe
