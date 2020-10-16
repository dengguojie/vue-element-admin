/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief avg_pool fusion pass
 *
 */
#ifndef AVG_POOL_MATRIX_FUSION_H
#define AVG_POOL_MATRIX_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe
{
    class AvgPoolFusionPass: public PatternFusionBasePass
    {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
        Status Fusion(ge::ComputeGraph &graph,
                      Mapping &mapping,
                      vector<ge::NodePtr> &fusionNodes) override;
    private:
        Status AddCoffe(ge::ComputeGraph &graph, ge::NodePtr &mulNode, string &padding, vector<int64_t> &dimInfo,
                                           vector<int64_t> ksize, vector<int64_t> stride);
        ge::NodePtr AddMul(ge::ComputeGraph &graph, ge::NodePtr &avgPoolNode,
                ge::Format &inputOriginFormat);
        Status Calc4DWeightAvgPool(const std::vector<int64_t> &filterDims4D,
                                   const int64_t &kernelDataCount,
                                   const int8_t *filterInt8Data,
                                   std::unique_ptr<int32_t[]> &weightInt8Temp);
        Status DoBiasOptimizeAvgpool(ge::ComputeGraph &graph, ge::NodePtr poolingNode,
                                     vector<ge::NodePtr> &fusionNodes);
        Status GetWeightOfConvAvgpool(const std::string &opName,
                const int8_t *filterInt8Data, const std::vector<int64_t> &filterDims,
                std::unique_ptr<int32_t[]> &weightInt8OutParam);
        Status RemoveDequantAndquant(ge::ComputeGraph &graph, ge::NodePtr &avgpoolNode,
                                     ge::NodePtr &quantNode, ge::NodePtr &dequantNode, ge::NodePtr &hostcpuNode);
        Status UpdateDequantConst(ge::ComputeGraph &graph, ge::NodePtr &const_node, float &area_factor);
        const string FUSED_OP_TYPE = "AvgPool";
    };
}  // namespace fe

#endif  // AVG_POOL_MATRIX_FUSION_H