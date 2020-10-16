/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief Pooling fusion pass
 *
 */
#ifndef POOLING_MATRIX_FUSION_H
#define POOLING_MATRIX_FUSION_H

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe
{
    class PoolingFusionPass: public PatternFusionBasePass
    {
    protected:
        vector<FusionPattern*> DefinePatterns() override;
        Status Fusion(ge::ComputeGraph &graph,
                      Mapping &mapping,
                      vector<ge::NodePtr> &fusionNodes) override;
private:
  Status AddAntiquant(ge::ComputeGraph &graph, ge::NodePtr &poolingNode);
  Status Calc4DWeight(const std::vector<int64_t> &filterDims4D,
                      const int64_t &kernelDataCount,
                      const int8_t *filterInt8Data,
                      std::unique_ptr<int32_t[]> &weightInt8Temp);
  Status DoBiasOptimize(ge::ComputeGraph &graph, ge::NodePtr poolingNode,
                        vector<ge::NodePtr> &fusionNodes);
  Status GetWeightOfConv(const std::string &opName,
                         const int8_t *filterInt8Data,
                         const std::vector<int64_t> &filterDims,
                         std::unique_ptr<int32_t[]> &weightInt8OutParam);
  bool IsMeanValueAllEqual(vector<int64_t> input, vector<int64_t> window,
                           vector<int64_t> stride, vector<int64_t> pad,
                           int64_t ceil_mode);
  Status RemoveDequantAndAddAntiquant(ge::ComputeGraph &graph,
                                      ge::NodePtr &poolingNode);
  Status RemoveDequant(ge::ComputeGraph &graph, ge::NodePtr &poolingNode);

  const string FUSED_OP_TYPE = "Pooling";
};

}  // namespace fe

#endif  // POOLING_MATRIX_FUSION_H