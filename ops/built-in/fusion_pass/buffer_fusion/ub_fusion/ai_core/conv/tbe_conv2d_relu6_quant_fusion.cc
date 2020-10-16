#include <string>
#include <vector>
#include "tbe_conv2d_relu6_quant_fusion.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"

namespace fe {

static const char PATTERN_CONV[] =  "convolution";
static const char PATTERN_RELU[] = "relu6";
static const char PATTERN_QUANT[] = "quant";
/*
 * @brief:  define convolution relu6 quant op fusion pattern
 *
 *    Convolution-->relu6-->quant
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern *> TbeConv2dRelu6QuantFusion::DefinePatterns() {

    vector<BufferFusionPattern *> patterns;
    string passName = "TbeConv2dRelu6QuantFusion";
    BufferFusionPattern *pattern =
        new (std::nothrow) BufferFusionPattern(passName);
    FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(FUSED_OP_TYPE.c_str(), "new an object failed."),
             return patterns);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Start to define %s pass pattern.", passName.c_str());
    pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_RELU, {OP_PATTERN_ELEMWISE},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .AddOpDesc(PATTERN_QUANT, {OP_PATTERN_QUANT},
                       TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
            .SetHead({PATTERN_CONV})
            .SetOutputs(PATTERN_CONV, {PATTERN_RELU})
            .SetOutputs(PATTERN_RELU, {PATTERN_QUANT});
    patterns.push_back(pattern);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to define %s pass pattern.", passName.c_str());

    return patterns;
}

Status TbeConv2dRelu6QuantFusion::GetFusionNodes(
    const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Begin to do Conv2DRelu6Quant!");
    fusionNodes= GetMatchedNodes(mapping);
    // the outputData can't be fused
    for (auto &item : mapping) {
        auto opdesc = find(item.first->types.begin(),
                           item.first->types.end(),
                           TBE_PATTERN_OUTPUT_NODE);

        if (opdesc != item.first->types.end()) {
            for (auto &node : item.second) {
                auto nodePtr = find(fusionNodes.begin(), fusionNodes.end(),
                                    node);
                fusionNodes.erase(nodePtr);
            }
        }
    }
    for (auto &item : mapping) {
        // judge Relu6 node
        if (item.first->desc_name == PATTERN_RELU) {
            for (auto &node : item.second) {
                if (node->GetType() != "Relu6") {
                    fusionNodes.clear();
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "relu6 is not vaild, only support relu6");
                    return SUCCESS;
                }
            }
        }
        if (item.first->desc_name == PATTERN_QUANT) {
            for (auto &node : item.second) {
                // this pass only support quant op
                if (node->GetType() != "AscendQuant") {
                    fusionNodes.clear();
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "we only support op type : AscendQuant, "
                            "not support %s.",
                            node->GetType().c_str());
                    return SUCCESS;
                }
            }
        }
    }
    OP_LOGD(FUSED_OP_TYPE.c_str(), "End to Conv2DRelu6Quant!");
    return SUCCESS;
}
REGISTER_BUFFER_FUSION_PASS("TbeConv2dRelu6QuantFusion",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            TbeConv2dRelu6QuantFusion);
}
