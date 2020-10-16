/**
 * @file tbe_conv_bnreduce_fusion_pass.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief tbe aipp convolution relue pooling ops fusion pattern
 *
 * @version 1.0
 *
 */

#include "tbe_aipp_conv_relu_maxpooling_fusion_pass.h"
#include "tbe_aipp_fusion_rule.h"
#include <algorithm>
#include <string>
#include <vector>
#include <math.h>
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {
using std::vector;

namespace {
static const char PATTERN_AIPP[] = "aipp";
static const char PATTERN_CONV[] = "convolution";
static const char PATTERN_ELTWISE[] = "eltwise";
static const char PATTERN_MAXPOOL[] = "maxpool";
static const char PATTERN_OTHER_INPUT[] = "otherInput";
static const char PATTERN_OTHER_INPUT1[] = "otherInput1";

static const char OP_TYPE_MAXPOOL[] = "MaxPool";
static const char OP_TYPE_POOLING[] = "Pooling";
static const char OP_TYPE_LEAKY_RELU[] = "LeakyRelu";
static const char OP_TYPE_RELU[] = "Relu";

static const string PADS = "pads";
static const string STRIDES = "strides";
static const string KSIZE = "ksize";
static const string WINDOW = "window";
static const string STRIDE = "stride";
static const string MODE = "mode";
static const string DATA_FORMAT = "data_format";
static const string FORMAT_NC1HWC0 = "NC1HWC0";

}

/*
 * @brief:  define conv and relu and max_pooling input op fusion pattern
 *
 *   AIPP-->Convolution-->ElemWise(optional)-->MaxPool/Pooling

 * @return TbeFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *>
TbeAippConvReluMaxpoolingFusionPass::DefinePatterns() {
  vector<BufferFusionPattern *> patterns;

  string passName = "TbeAippConvReluMaxpoolingFusionPass1";
  BufferFusionPattern *pattern =
      new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern
      ->AddOpDesc(PATTERN_AIPP, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_MAXPOOL, {OP_PATTERN_MAXPOOL, OP_PATTERN_POOL2D},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_AIPP})
      .SetOutputs(PATTERN_AIPP, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_CONV})
      .SetOutputs(PATTERN_CONV, {PATTERN_ELTWISE})
      .SetOutputs(PATTERN_ELTWISE, {PATTERN_MAXPOOL});
  patterns.push_back(pattern);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

  string passName2 = "TbeAippConvReluMaxpoolingFusionPass2";
  BufferFusionPattern *pattern2 =
      new (std::nothrow) BufferFusionPattern(passName2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName2.c_str());
  pattern2
      ->AddOpDesc(PATTERN_AIPP, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_ELTWISE, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT,
                 TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_MAXPOOL, {OP_PATTERN_MAXPOOL, OP_PATTERN_POOL2D},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_AIPP})
      .SetOutputs(PATTERN_AIPP, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_CONV})
      .SetOutputs(PATTERN_CONV, {PATTERN_ELTWISE})
      .SetOutputs(PATTERN_ELTWISE, {PATTERN_MAXPOOL});
  patterns.push_back(pattern2);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName2.c_str());

  string passName3 = "TbeAippConvMaxpoolingFusionPass1";
  BufferFusionPattern *pattern3 = new (std::nothrow) BufferFusionPattern(passName3);
  FUSION_PASS_CHECK((pattern3 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName3.c_str());
  pattern3
      ->AddOpDesc(PATTERN_AIPP, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_MAXPOOL, {OP_PATTERN_MAXPOOL, OP_PATTERN_POOL2D},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_AIPP})
      .SetOutputs(PATTERN_AIPP, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_CONV})
      .SetOutputs(PATTERN_CONV, {PATTERN_MAXPOOL});
  patterns.push_back(pattern3);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName3.c_str());

  string passName4 = "TbeAippConvMaxpoolingFusionPass2";
  BufferFusionPattern *pattern4 = new (std::nothrow) BufferFusionPattern(passName4);
  FUSION_PASS_CHECK((pattern4 == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
           return patterns);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName4.c_str());
  pattern4
      ->AddOpDesc(PATTERN_AIPP, {OP_PATTERN_AIPP}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_OTHER_INPUT1, {TBE_PATTERN_INPUT_NODE},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT,
                  TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PATTERN_MAXPOOL, {OP_PATTERN_MAXPOOL, OP_PATTERN_POOL2D},
                 TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PATTERN_AIPP})
      .SetOutputs(PATTERN_AIPP, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_CONV})
      .SetOutputs(PATTERN_OTHER_INPUT1, {PATTERN_CONV})
      .SetOutputs(PATTERN_CONV, {PATTERN_MAXPOOL});
  patterns.push_back(pattern4);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName4.c_str());

  return patterns;
}
/***************************************************************
check! aipp, conv and maxpool fusion checking.
***************************************************************/
bool TbeAippConvReluMaxpoolingFusionPass::CheckConvPoolNodeValidation(ge::NodePtr convNode) {
  FUSION_PASS_CHECK(convNode->GetOpDesc()->GetInputDesc(1).GetFormat() != ge::FORMAT_FRACTAL_Z_C04,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "The format of node[%s]'s second input is not FORMAT_FRACTAL_Z_C04",
                   convNode->GetName().c_str()),
           return false);
  ge::Format firstFormat =
          convNode->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  std::vector<int64_t> firstDims =
          convNode->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  ge::Format secondFormat =
          convNode->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> secondDims =
          convNode->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  FUSION_PASS_CHECK(firstFormat != ge::FORMAT_NCHW && firstFormat != ge::FORMAT_NHWC &&
           secondFormat != ge::FORMAT_NCHW && secondFormat != ge::FORMAT_NHWC &&
           secondFormat != ge::FORMAT_HWCN,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s format is [%d] and [%d], can not fusion.",
                   convNode->GetName().c_str(), firstFormat, secondFormat),
           return false);
  FUSION_PASS_CHECK(firstDims.size() != 4,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s first input shape size is [%zu] not 4,"
                   "can not fusion.",
                   convNode->GetName().c_str(), firstDims.size()),
           return false);
  FUSION_PASS_CHECK(secondDims.size() != 4,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s second input shape size is [%zu] not 4,"
                   "can not fusion.",
                   convNode->GetName().c_str(), secondDims.size()),
           return false);
  vector<int64_t> strides;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(convNode->GetOpDesc(), STRIDES,
                                      strides),
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s strides attr not success.",
                   convNode->GetName().c_str()), return false);
  FUSION_PASS_CHECK(strides.size() != 4,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr strides size [%zu] not 4, can not fusion.",
                   convNode->GetName().c_str(), strides.size()),
           return false);
  if (firstFormat == ge::FORMAT_NCHW) {
    FUSION_PASS_CHECK(firstDims[1] > 4 || firstDims[3] > 1024,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s first input shape is more than [N, 4, N, 1024],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
    FUSION_PASS_CHECK((strides[2] != 2 || strides[3] != 2) && (strides[2] != 1 || strides[3] != 1),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr strides is not [N, N, 2, 2] or [N, N, 1, 1],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
  } else {
    FUSION_PASS_CHECK(firstDims[3] > 4 || firstDims[2] > 1024,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s first input shape is more than [N, N, 1024, 4],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
    FUSION_PASS_CHECK((strides[1] != 2 || strides[2] != 2) && (strides[1] != 1 || strides[2] != 1),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr strides is not [N, 2, 2, N],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
  }
  if (secondFormat == ge::FORMAT_NCHW) {
    FUSION_PASS_CHECK(secondDims[0] > 64,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s second input shape is more than [64, N, N, N],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
    FUSION_PASS_CHECK((secondDims[2] != 3 || secondDims[3] != 3) &&
             (secondDims[2] != 5 || secondDims[3] != 5) &&
             (secondDims[2] != 7 || secondDims[3] != 7),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s second input shape is not [N, 2, 2, N],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
  } else if (secondFormat == ge::FORMAT_NHWC) {
    FUSION_PASS_CHECK(secondDims[0] > 64,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s second input shape is more than [64, N, N, N],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
    FUSION_PASS_CHECK((secondDims[1] != 3 || secondDims[2] != 3) &&
             (secondDims[1] != 5 || secondDims[2] != 5) &&
             (secondDims[1] != 7 || secondDims[2] != 7),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s second input shape is not [N, 2, 2, N],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
  } else {
    FUSION_PASS_CHECK(secondDims[3] > 64,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s second input shape is more than [N, N, N, 64],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
    FUSION_PASS_CHECK((secondDims[0] != 3 || secondDims[1] != 3) &&
             (secondDims[0] != 5 || secondDims[1] != 5) &&
             (secondDims[0] != 7 || secondDims[1] != 7),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s second input shape is not [N, 2, 2, N],"
                     "can not fusion.",
                     convNode->GetName().c_str()),
             return false);
  }
  return true;
}
/***************************************************************
check! aipp, conv and maxpool fusion checking.
***************************************************************/
bool TbeAippConvReluMaxpoolingFusionPass::CheckMaxpoolNodeValidation(ge::NodePtr maxPoolNode) {
  if (maxPoolNode->GetOpDesc()->GetType() == OP_TYPE_POOLING) {
    vector<int64_t> strides;
    vector<int64_t> standardStrides = {2, 2};
    FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(maxPoolNode->GetOpDesc(), STRIDE,
                                        strides),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s strides attr not success.",
                     maxPoolNode->GetName().c_str()), return false);
    FUSION_PASS_CHECK(strides.size() != 2,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr strides size is [%zu] not 4,"
                     "can not fusion.",
                     maxPoolNode->GetName().c_str(), strides.size()),
             return false);
    FUSION_PASS_CHECK(strides != standardStrides,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr stride is not [2, 2], can not fusion.",
                     maxPoolNode->GetName().c_str()),
             return false);

    vector<int64_t> window;
    vector<int64_t> standardWindow = {3, 3};
    FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(maxPoolNode->GetOpDesc(), WINDOW,
                                        window),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s window attr not success.",
                     maxPoolNode->GetName().c_str()), return false);
    FUSION_PASS_CHECK(window.size() != 2,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr window size is [%zu] not 4,"
                     "can not fusion.",
                     maxPoolNode->GetName().c_str(), window.size()),
             return false);
    FUSION_PASS_CHECK(window != standardWindow,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr window is not [3, 3], can not fusion.",
                     maxPoolNode->GetName().c_str()),
             return false);

    int64_t mode;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetInt(maxPoolNode->GetOpDesc(), MODE, mode),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s mode attr not success.",
                     maxPoolNode->GetName().c_str()), return false);
    FUSION_PASS_CHECK(mode != 0,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr mode is [%ld] not 0, can not fusion.",
                     maxPoolNode->GetName().c_str(), mode),
             return false);
  }
  if (maxPoolNode->GetOpDesc()->GetType() == OP_TYPE_MAXPOOL) {
    vector<int64_t> strides;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(maxPoolNode->GetOpDesc(), STRIDES,
                                        strides),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s strides attr not success.",
                     maxPoolNode->GetName().c_str()), return false);
    FUSION_PASS_CHECK(strides.size() != 4,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr strides is [%zu] not 4, can not fusion.",
                     maxPoolNode->GetName().c_str(), strides.size()),
             return false);

    vector<int64_t> ksizes;
    FUSION_PASS_CHECK(!ge::AttrUtils::GetListInt(maxPoolNode->GetOpDesc(), KSIZE,
                                        ksizes),
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Get node[%s]'s ksize attr not success.",
                     maxPoolNode->GetName().c_str()), return false);
    FUSION_PASS_CHECK(ksizes.size() != 4,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s Attr strides is [%zu] not 4, can not fusion.",
                     maxPoolNode->GetName().c_str(), ksizes.size()),
             return false);

    ge::Format format =
          maxPoolNode->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
    if (format == ge::FORMAT_NCHW) {
      FUSION_PASS_CHECK(strides[2] != 2 || strides[3] != 2,
                OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr strides is not [N, N, 2, 2],"
                        "can not fusion.",
                        maxPoolNode->GetName().c_str()),
                return false);
      FUSION_PASS_CHECK(ksizes[2] != 3 || ksizes[3] != 3,
                OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s Attr ksize is not [N, N, 3, 3],"
                        "can not fusion.",
                        maxPoolNode->GetName().c_str()),
                return false);
    } else if (format == ge::FORMAT_NHWC) {
      FUSION_PASS_CHECK(strides[1] != 2 || strides[2] != 2,
                OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s attr strides is not [N, 2, 2, N],"
                        "can not fusion.",
                        maxPoolNode->GetName().c_str()),
                return false);
      FUSION_PASS_CHECK(ksizes[1] != 3 || ksizes[2] != 3,
                OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s Attr ksize is not [N, 3, 3, N],"
                        "can not fusion.",
                        maxPoolNode->GetName().c_str()),
                return false);
    } else {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "node[%s]'s format is [%d], can not fusion.",
              maxPoolNode->GetName().c_str(), format);
      return false;
    }
  }
  return true;
}
/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeAippConvReluMaxpoolingFusionPass::GetFusionNodes(
    const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do TbeConvReluMaxpoolingFusionPass!");
  vector<ge::NodePtr> convNodes = GetMatchedNodesByDescName(PATTERN_CONV, mapping);
  vector<ge::NodePtr> elemwiseNodes = GetMatchedNodesByDescName(PATTERN_ELTWISE, mapping);
  vector<ge::NodePtr> maxPoolNodes = GetMatchedNodesByDescName(PATTERN_MAXPOOL, mapping);

  if (!elemwiseNodes.empty()) {
    for (auto elemwiseNode : elemwiseNodes) {
      FUSION_PASS_CHECK((elemwiseNode->GetType() != OP_TYPE_LEAKY_RELU) &&
               (elemwiseNode->GetType() != OP_TYPE_RELU),
               OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s opType is [%s], no need to do UB-fusion.",
                       elemwiseNode->GetName().c_str(),
                       elemwiseNode->GetType().c_str()),
               return SUCCESS);
    }
  }
  for (auto convNode : convNodes) {
    if (!CheckConvPoolNodeValidation(convNode)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] not satisfied with fusion condition.",
              convNode->GetName().c_str());
      return SUCCESS;
    }
    if (!TbeAippFusionRule::CheckAippConvStridehValidation(convNode)){
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The case is the strideh optim. "
          "Node[%s] not satisfied with fusion condition.",
              convNode->GetName().c_str());
      return SUCCESS;
    }
  }
  for (auto maxPoolNode : maxPoolNodes) {
    if (!CheckMaxpoolNodeValidation(maxPoolNode)) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s] not satisfied with fusion condition.",
              maxPoolNode->GetName().c_str());
      return SUCCESS;
    }
  }

  fusionNodes = GetMatchedNodes(mapping);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "End to do TbeAippConvReluMaxpoolingFusionPass!");
  return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeAippConvReluMaxpoolingFusion",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeAippConvReluMaxpoolingFusionPass);
} // namespace fe
