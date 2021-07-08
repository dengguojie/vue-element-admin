/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file tbe_aipp_conv_relu_maxpooling_fusion_pass.cpp
 * \brief tbe aipp and convolution and relu and maxpooling ops fusion pattern
 */
#include "tbe_aipp_conv_relu_maxpooling_fusion_pass.h"
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include "tbe_aipp_fusion_rule.h"
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "aicore_util_attr_define.h"

namespace fe {
using std::vector;

namespace {
static const char kPatternAipp[] = "aipp";
static const char kPatternConv[] = "convolution";
static const char kPatternEltwise[] = "eltwise";
static const char kPatternMaxpool[] = "maxpool";

static const char kOpTypeMaxPool[] = "MaxPool";
static const char kOpTypePooling[] = "Pooling";
static const char kOpTypeLeakyRelu[] = "LeakyRelu";
static const char kOpTypeRelu[] = "Relu";
static const char kPads[] = "pads";
static const char kStrides[] = "strides";
static const char kKsize[] = "ksize";
static const char kWindow[] = "window";
static const char kStride[] = "stride";
static const char kMode[] = "mode";
static const char kNSlope[] = "negative_slope";
static int64_t width = 0;

}  // namespace

/*
 * @brief:  define conv and relu and max_pooling input op fusion pattern
 *   AIPP(optional)-->Convolution-->ElemWise(optional)-->MaxPool/Pooling
 * @return TbeFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeAippConvReluMaxpoolingFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pass_name = "TbeAippConvReluMaxpoolingFusionPass1";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pass_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pass_name.c_str());
  pattern->AddOpDesc(kPatternAipp, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternEltwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMaxpool, {kOpTypeMaxPool, OP_PATTERN_POOL2D}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternAipp, kPatternConv})
      .SetOutputs(kPatternAipp, {kPatternConv})
      .SetOutputs(kPatternConv, {kPatternEltwise}, TBE_OUTPUT_BRANCH_SINGLE, true)
      .SetOutputs(kPatternEltwise, {kPatternMaxpool});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pass_name.c_str());
  return patterns;
}

/*
 * @brief: aipp, conv and maxpool fusion checking.
 * @return bool: fusion status ok or not.
 */
bool TbeAippConvReluMaxpoolingFusionPass::CheckConvNodeValidation(const ge::NodePtr& conv_node) {
  FUSION_PASS_CHECK(conv_node->GetOpDesc() == nullptr,
                    OP_LOGD(fused_op_type_.c_str(), "get desc failed"),
                    return false);
  FUSION_PASS_CHECK(conv_node->GetOpDesc()->GetInputDesc(1).GetFormat() != ge::FORMAT_FRACTAL_Z_C04,
                    OP_LOGD(fused_op_type_.c_str(), "The format of node[%s]'s second input is not FORMAT_FRACTAL_Z_C04",
                            conv_node->GetName().c_str()),
                    return false);
  ge::Format first_format = conv_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  std::vector<int64_t> first_dims = conv_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  ge::Format second_format = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> second_dims = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  FUSION_PASS_CHECK(first_format != ge::FORMAT_NCHW && first_format != ge::FORMAT_NHWC &&
                    second_format != ge::FORMAT_NCHW && second_format != ge::FORMAT_NHWC &&
                    second_format != ge::FORMAT_HWCN,
                    OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s format is [%d] and [%d], can not fusion.",
                            conv_node->GetName().c_str(), first_format, second_format),
                    return false);
  FUSION_PASS_CHECK(first_dims.size() != 4,
                    OP_LOGD(fused_op_type_.c_str(),
                            "node[%s]'s first input shape size is [%zu] not 4,"
                            "can not fusion.",
                            conv_node->GetName().c_str(), first_dims.size()),
                    return false);
  FUSION_PASS_CHECK(second_dims.size() != 4,
                    OP_LOGD(fused_op_type_.c_str(),
                            "node[%s]'s second input shape size is [%zu] not 4,"
                            "can not fusion.",
                            conv_node->GetName().c_str(), second_dims.size()),
                    return false);
  vector<int64_t> strides;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), kStrides, strides),
      OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s strides attr not success.", conv_node->GetName().c_str()),
      return false);
  FUSION_PASS_CHECK(strides.size() != 4,
                    OP_LOGD(fused_op_type_.c_str(), "node[%s]'s attr strides size [%zu] is not 4, can not fusion.",
                            conv_node->GetName().c_str(), strides.size()),
                    return false);
  if (first_format == ge::FORMAT_NCHW) {
    FUSION_PASS_CHECK(first_dims[1] > 4 || first_dims[3] > width,
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s first input shape is more than [N, 4, N, " + std::to_string(width) +"],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
    FUSION_PASS_CHECK((strides[2] != 2 || strides[3] != 2) && (strides[2] != 1 || strides[3] != 1),
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s attr strides is not [N, N, 2, 2] or [N, N, 1, 1],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
  } else {
    FUSION_PASS_CHECK(first_dims[3] > 4 || first_dims[2] > width,
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s first input shape is more than [N, 4, N, " + std::to_string(width) +"],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
    FUSION_PASS_CHECK((strides[1] != 2 || strides[2] != 2) && (strides[1] != 1 || strides[2] != 1),
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s attr strides is not [N, 2, 2, N] or [N, 1, 1, N],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
  }
  if (second_format == ge::FORMAT_NCHW) {
    FUSION_PASS_CHECK(second_dims[0] > 96,
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s second input shape is more than [96, N, N, N],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
    FUSION_PASS_CHECK((second_dims[2] != 3 || second_dims[3] != 3) && (second_dims[2] != 5 || second_dims[3] != 5) &&
                      (second_dims[2] != 7 || second_dims[3] != 7),
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s second input shape is not [N, N, 3, 3] or [N, N, 5, 5] or [N, N, 7, 7],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
  } else if (second_format == ge::FORMAT_NHWC) {
    FUSION_PASS_CHECK(second_dims[0] > 64,
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s second input shape is more than [64, N, N, N],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
    FUSION_PASS_CHECK((second_dims[1] != 3 || second_dims[2] != 3) && (second_dims[1] != 5 || second_dims[2] != 5) &&
                      (second_dims[1] != 7 || second_dims[2] != 7),
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s second input shape is not [N, 3, 3, N] or [N, 5, 5, N] or [N, 7, 7, N],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
  } else {
    FUSION_PASS_CHECK(second_dims[3] > 64,
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s second input shape is more than [N, N, N, 64],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
    FUSION_PASS_CHECK((second_dims[0] != 3 || second_dims[1] != 3) && (second_dims[0] != 5 || second_dims[1] != 5) &&
                      (second_dims[0] != 7 || second_dims[1] != 7),
                      OP_LOGD(fused_op_type_.c_str(),
                              "node[%s]'s second input shape is not [3, 3, N, N] or [5, 5, N, N] or [7, 7, N, N],"
                              "can not fusion.",
                              conv_node->GetName().c_str()),
                      return false);
  }
  return true;
}

/*
 * @brief: aipp, conv and maxpool fusion checking.
 * @return bool: fusion status ok or not.
 */
bool TbeAippConvReluMaxpoolingFusionPass::CheckMaxpoolNodeValidation(const ge::NodePtr& max_pool_node) {
  int64_t windowSize = 0;
  std::vector<std::string> support_type = {kOpTypePooling, kOpTypeMaxPool};
  bool is_support = false;
  FUSION_PASS_CHECK((max_pool_node == nullptr || max_pool_node->GetOpDesc() == nullptr),
                    OP_LOGD(fused_op_type_.c_str(), "get desc failed"),
                    return false);
  std::string op_type = max_pool_node->GetOpDesc()->GetType();
  for (auto type : support_type) {
    if (type == op_type) {
      is_support = true;
      break;
    }
  }
  FUSION_PASS_CHECK(!is_support, OP_LOGD(fused_op_type_.c_str(), "op type [%s] is not supported.", op_type.c_str()),
                    return false);
  if (op_type == kOpTypePooling) {
    vector<int64_t> strides;
    vector<int64_t> standard_strides = {2, 2};
    FUSION_PASS_CHECK(
        !ge::AttrUtils::GetListInt(max_pool_node->GetOpDesc(), kStride, strides),
        OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s strides attr not success.", max_pool_node->GetName().c_str()),
        return false);
    FUSION_PASS_CHECK(strides != standard_strides,
                      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s attr stride is not [2, 2], can not fusion.",
                              max_pool_node->GetName().c_str()),
                      return false);

    vector<int64_t> window;
    vector<int64_t> standard_window = {3, 3};
    vector<int64_t> standard_window_2_2 = {2, 2};
    FUSION_PASS_CHECK(
        !ge::AttrUtils::GetListInt(max_pool_node->GetOpDesc(), kWindow, window),
        OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s window attr not success.", max_pool_node->GetName().c_str()),
        return false);
    FUSION_PASS_CHECK((window != standard_window) && (window != standard_window_2_2),
                      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s attr window is not [3, 3] and not [2, 2], can not fusion.",
                              max_pool_node->GetName().c_str()),
                      return false);
    windowSize = window[0];
    int64_t mode;
    FUSION_PASS_CHECK(
        !ge::AttrUtils::GetInt(max_pool_node->GetOpDesc(), kMode, mode),
        OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s mode attr not success.", max_pool_node->GetName().c_str()),
        return false);
    FUSION_PASS_CHECK(mode != 0,
                      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s attr mode is [%ld] not 0, can not fusion.",
                              max_pool_node->GetName().c_str(), mode),
                      return false);
  }
  if (op_type == kOpTypeMaxPool) {
    vector<int64_t> strides;
    FUSION_PASS_CHECK(
        !ge::AttrUtils::GetListInt(max_pool_node->GetOpDesc(), kStrides, strides),
        OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s strides attr not success.", max_pool_node->GetName().c_str()),
        return false);
    FUSION_PASS_CHECK(strides.size() != 4,
                      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s attr strides is [%zu] not 4, can not fusion.",
                              max_pool_node->GetName().c_str(), strides.size()),
                      return false);

    vector<int64_t> ksizes;
    FUSION_PASS_CHECK(
        !ge::AttrUtils::GetListInt(max_pool_node->GetOpDesc(), kKsize, ksizes),
        OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s ksize attr not success.", max_pool_node->GetName().c_str()),
        return false);
    FUSION_PASS_CHECK(ksizes.size() != 4,
                      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s Attr strides is [%zu] not 4, can not fusion.",
                              max_pool_node->GetName().c_str(), ksizes.size()),
                      return false);
    ge::Format format = max_pool_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
    if (format == ge::FORMAT_NCHW) {
      FUSION_PASS_CHECK(strides[2] != 2 || strides[3] != 2,
                        OP_LOGD(fused_op_type_.c_str(),
                                "node[%s]'s attr strides is not [N, N, 2, 2],"
                                "can not fusion.",
                                max_pool_node->GetName().c_str()),
                        return false);
      FUSION_PASS_CHECK((ksizes[2] != 3 || ksizes[3] != 3) && (ksizes[2] != 2 || ksizes[3] != 2),
                        OP_LOGD(fused_op_type_.c_str(), "node[%s]'s Attr ksize is not [N, N, 3, 3]"
                                " or not [N, N, 2, 2], can not fusion.", max_pool_node->GetName().c_str()),
                        return false);
      windowSize = ksizes[2];
    } else if (format == ge::FORMAT_NHWC) {
      FUSION_PASS_CHECK(strides[1] != 2 || strides[2] != 2,
                        OP_LOGD(fused_op_type_.c_str(),
                                "node[%s]'s attr strides is not [N, 2, 2, N],"
                                "can not fusion.",
                                max_pool_node->GetName().c_str()),
                        return false);
      FUSION_PASS_CHECK((ksizes[1] != 3 || ksizes[2] != 3) && (ksizes[1] != 2 || ksizes[2] != 2),
                        OP_LOGD(fused_op_type_.c_str(), "node[%s]'s Attr ksize is not [N, 3, 3, N] "
                                "or not [N, 2, 2, N], can not fusion.", max_pool_node->GetName().c_str()),
                        return false);
      windowSize = ksizes[2];
    } else {
      OP_LOGD(fused_op_type_.c_str(), "node[%s]'s format is [%d], can not fusion.", max_pool_node->GetName().c_str(),
              format);
      return false;
    }
  }
  if (windowSize == 2) {
    width = 1000;
  }
  else {
    width = 800;
  }
  return true;
}

void TbeAippConvReluMaxpoolingFusionPass::PoolingValidationAndFormatSet(const ge::NodePtr& aipp_node,
                                                                        const ge::NodePtr& conv_node,
                                                                        const ge::NodePtr& max_pool_node) {
  OP_LOGD(fused_op_type_.c_str(), "checking pooling input witdh of TbeAippConvReluMaxpoolingFusionPass");
  vector<int64_t> first_output_dims_aipp(5);
  vector<int64_t> first_input_dims_conv(5);
  vector<int64_t> first_input_dims_pool(5);

  first_output_dims_aipp = aipp_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
  first_input_dims_conv = conv_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  first_input_dims_pool = max_pool_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
  bool invalid = aipp_node->GetOpDesc() == nullptr || conv_node->GetOpDesc() == nullptr ||
                 max_pool_node->GetOpDesc() == nullptr;
  if (invalid) {
    OP_LOGD(fused_op_type_.c_str(), "get desc failed");
    return;
  }

  OP_LOGD(fused_op_type_.c_str(), "pooling input witdh of TbeAippConvReluMaxpoolingFusionPass is [%d]", first_input_dims_pool[3]);

  if (first_input_dims_pool[3] % 16 != 0) { // input width of pooling node
    aipp_node->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NC1HWC0);
    conv_node->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_NC1HWC0);

    first_output_dims_aipp[4] = 16; // C0 = 16 for FORMAT_NC1HWC0
    first_input_dims_conv[4] = 16;

    aipp_node->GetOpDesc()->MutableOutputDesc(0)->SetShape(ge::GeShape(first_output_dims_aipp));
    conv_node->GetOpDesc()->MutableInputDesc(0)->SetShape(ge::GeShape(first_input_dims_conv));

    ge::AttrUtils::SetBool(aipp_node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
    ge::AttrUtils::SetBool(conv_node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
    ge::ComputeGraphPtr graph_ptr = conv_node->GetOwnerComputeGraph();
    (void)ge::AttrUtils::SetBool(graph_ptr, NEED_RE_PRECOMPILE, true);

    OP_LOGD(fused_op_type_.c_str(), "Node[%s]'s output format has been changed to [%d].",
            aipp_node->GetName().c_str(), aipp_node->GetOpDesc()->GetOutputDesc(0).GetFormat());
    OP_LOGD(fused_op_type_.c_str(), "Node[%s]'s input format of fmap has been changed to [%d].",
            conv_node->GetName().c_str(), conv_node->GetOpDesc()->GetInputDesc(0).GetFormat());

    first_output_dims_aipp = aipp_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
    first_input_dims_conv = conv_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();

    OP_LOGD(fused_op_type_.c_str(), "Node[%s]'s output shape C0 has been changed to [%d].",
            aipp_node->GetName().c_str(), first_output_dims_aipp[4]);
    OP_LOGD(fused_op_type_.c_str(), "Node[%s]'s input shape C0 of fmap has been changed to [%d].",
            conv_node->GetName().c_str(), first_input_dims_conv[4]);
  }
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeAippConvReluMaxpoolingFusionPass::GetFusionNodes(const BufferFusionMapping& mapping,
                                                           vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do TbeAippConvReluMaxpoolingFusionPass!");
  vector<ge::NodePtr> aipp_nodes = GetMatchedNodesByDescName(kPatternAipp, mapping);
  vector<ge::NodePtr> conv_nodes = GetMatchedNodesByDescName(kPatternConv, mapping);
  vector<ge::NodePtr> elemwise_nodes = GetMatchedNodesByDescName(kPatternEltwise, mapping);
  vector<ge::NodePtr> max_pool_nodes = GetMatchedNodesByDescName(kPatternMaxpool, mapping);

  if (!elemwise_nodes.empty()) {
    for (const auto& elemwise_node : elemwise_nodes) {
      FUSION_PASS_CHECK((elemwise_node->GetType() != kOpTypeLeakyRelu) && (elemwise_node->GetType() != kOpTypeRelu),
                        OP_LOGD(fused_op_type_.c_str(), "Node[%s]'s opType is [%s], no need to do UB-fusion.",
                                elemwise_node->GetName().c_str(), elemwise_node->GetType().c_str()),
                        return SUCCESS);
      float nslope;
      if (elemwise_node->GetType() == kOpTypeLeakyRelu) {
        FUSION_PASS_CHECK(!ge::AttrUtils::GetFloat(elemwise_node->GetOpDesc(), kNSlope, nslope),
                        OP_LOGD(fused_op_type_.c_str(), "Get node[%s]'s negative_slope attr not success.",
                                elemwise_node->GetName().c_str()),
                        return SUCCESS);
        FUSION_PASS_CHECK(fabs(nslope - 0) > 1e-6,
                        OP_LOGD(fused_op_type_.c_str(), "node[%s]'s attr negative_slope is [%ld] not 0."
                                " not satisfied with fusion condition.",
                                elemwise_node->GetName().c_str(), nslope),
                        return SUCCESS);
      }
    }
  }
  for (const auto& max_pool_node : max_pool_nodes) {
    if (!CheckMaxpoolNodeValidation(max_pool_node)) {
      OP_LOGD(fused_op_type_.c_str(), "Node[%s] not satisfied with fusion condition.", max_pool_node->GetName().c_str());
      return SUCCESS;
    }
  }

  for (const auto& conv_node : conv_nodes) {
    if (!CheckConvNodeValidation(conv_node)) {
      OP_LOGD(fused_op_type_.c_str(), "Node[%s] not satisfied with fusion condition.", conv_node->GetName().c_str());
      return SUCCESS;
    }
    if (!aipp_nodes.empty() && !TbeAippFusionRule::CheckAippConvStridehValidation(conv_node)) {
      OP_LOGD(fused_op_type_.c_str(),
              "The case is the strideh optim. "
              "Node[%s] not satisfied with fusion condition.",
              conv_node->GetName().c_str());
      return SUCCESS;
    }
  }

  // if fmap witdh of pooling cannot be divided by 16, change format to NC1HWC0
  if (!aipp_nodes.empty() and !conv_nodes.empty() and !max_pool_nodes.empty()) {
    PoolingValidationAndFormatSet(aipp_nodes[0], conv_nodes[0], max_pool_nodes[0]);
  }

  fusion_nodes = GetMatchedNodes(mapping);
  TbeAippFusionRule::SetSplitInfo(conv_nodes, fusion_nodes, true);
  OP_LOGD(fused_op_type_.c_str(), "End to do TbeAippConvReluMaxpoolingFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeAippConvReluMaxpoolingFusion", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeAippConvReluMaxpoolingFusionPass);
}  // namespace fe
