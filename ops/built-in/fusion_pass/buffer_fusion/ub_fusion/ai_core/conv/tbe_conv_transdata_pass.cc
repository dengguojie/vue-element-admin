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

#include "tbe_conv_transdata_pass.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "common/util/platform_info.h"

namespace fe {
using std::vector;
static const string PATTERN_CONV = "convolution";
static const string PATTERN_QUANT = "quant";
static const string PATTERN_TRANSDATA = "TransData";

vector<BufferFusionPattern*> ConvTransdataFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pattern_name = "TbeConvTransdataFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, 
                     TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_TRANSDATA, {"TransData"}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, 
                     TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE, true)
          .SetHead({PATTERN_CONV})
          .SetOutputs(PATTERN_CONV, {PATTERN_TRANSDATA});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name.c_str());
  return patterns;
}

Status ConvTransdataFusionPass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do ConvTransdataFusionPass!");

  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(PATTERN_CONV, mapping);
  vector<ge::NodePtr> transdata_nodes = GetMatchedNodesByDescName(PATTERN_TRANSDATA, mapping);

  if (conv_nodes.size() != 1 || transdata_nodes.size() != 1) {
    OP_LOGD(fused_op_type_.c_str(), "conv_nodes or transdata_nodes size is not 1.");
    return SUCCESS;
  }

  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, opti_compilation_info) != SUCCESS) {
    OP_LOGW("Fail to get platform info.");
  }

  OP_LOGD(fused_op_type_.c_str(), "Get opti_compilation_info.soc_version[%s]", opti_compilation_info.soc_version.c_str());
  if (opti_compilation_info.soc_version != "SD3403" && opti_compilation_info.soc_version != "Hi3796CV300CS") {
    return SUCCESS;
  }

  ge::NodePtr conv_node = conv_nodes[0];
  ge::NodePtr transdata_node = transdata_nodes[0];
  if (transdata_node->GetType() != "TransData") {
    OP_LOGD(fused_op_type_.c_str(), "transdata_node node type[%s] is not TransData", transdata_node->GetType().c_str());
    return SUCCESS;
  }

  std::string transdata_src_format;
  std::string transdata_dst_format;
  ge::AttrUtils::GetStr(transdata_node->GetOpDesc(), "src_format", transdata_src_format);
  ge::AttrUtils::GetStr(transdata_node->GetOpDesc(), "dst_format", transdata_dst_format);

  if (transdata_src_format != "NC1HWC0" || transdata_dst_format != "NCHW") {
    OP_LOGD(fused_op_type_.c_str(), "src format[%s] and dst format[%s] not support",
            transdata_src_format.c_str(), transdata_src_format.c_str());
    return SUCCESS;
  }

  auto outputDesc = transdata_node->GetOpDesc()->GetOutputDesc(0);
  vector<int64_t> output_shape = outputDesc.GetShape().GetDims();
  auto outputFormat = outputDesc.GetFormat();
  int64_t output_channel = 0;
  if (outputFormat == FORMAT_NCHW && output_shape.size() == 4) {
    output_channel = output_shape[1];
  } else {
    OP_LOGD(fused_op_type_.c_str(), "transdata node output format is not NCHW");
    return SUCCESS;
  }

  if (output_channel != 1) {
    OP_LOGD(fused_op_type_.c_str(), "transdata node output channel[%d] not equal 1.",
            output_channel);
    return SUCCESS;
  }

  fusion_nodes = GetMatchedNodes(mapping);
  OP_LOGD(fused_op_type_.c_str(), "End to do ConvTransdataFusionPass!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConvTransdataFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, ConvTransdataFusionPass);
}  // namespace fe
