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

#include "tbe_antiquant_maxpooling_fusion_pass.h"
#include <algorithm>
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;
namespace {
static const string PATTERN_ANTIQUANT = "antiquant";
static const string PATTERN_MAXPOOLING = "maxpooling";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_STRIDEDWRITE = "strided_write";
static const string OP_TYPE_POOLING = "Pool2d";
static const string OP_TYPE_MAXPOOL = "MaxPool";
static const string OP_TYPE_QUANT = "quant";
}

vector<BufferFusionPattern*> AntiquantMaxpoolingFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pattern_name = "TbeAntiquantMaxpoolingFusionPass";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_ANTIQUANT, {OP_PATTERN_ANTIQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_MAXPOOLING, {OP_TYPE_POOLING, OP_TYPE_MAXPOOL}, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_QUANT, {OP_TYPE_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_STRIDEDWRITE, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({PATTERN_ANTIQUANT})
          .SetOutputs(PATTERN_ANTIQUANT, {PATTERN_MAXPOOLING})
          .SetOutputs(PATTERN_MAXPOOLING, {PATTERN_QUANT})
          .SetOutputs(PATTERN_QUANT, {PATTERN_STRIDEDWRITE});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name.c_str());

  string pattern_name1 = "TbeAntiquantMaxpoolingQuantFusionPass";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pattern_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name1.c_str());
  // define pattern rules
  pattern1->AddOpDesc(PATTERN_ANTIQUANT, {OP_PATTERN_ANTIQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_MAXPOOLING, {OP_TYPE_POOLING, OP_TYPE_MAXPOOL}, TBE_PATTERN_NUM_DEFAULT,
                     TBE_PATTERN_NUM_DEFAULT)
          .AddOpDesc(PATTERN_STRIDEDWRITE, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
          .SetHead({PATTERN_ANTIQUANT})
          .SetOutputs(PATTERN_ANTIQUANT, {PATTERN_MAXPOOLING})
          .SetOutputs(PATTERN_MAXPOOLING, {PATTERN_STRIDEDWRITE});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name1.c_str());

  return patterns;
}

REGISTER_BUFFER_FUSION_PASS("TbeAntiquantMaxpoolingFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            AntiquantMaxpoolingFusionPass);
}  // namespace fe
