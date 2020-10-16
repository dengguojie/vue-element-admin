#ifndef FE_AVG_POOL_1D_FUSION_H
#define FE_AVG_POOL_1D_FUSION_H

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe
{
  class AvgPool1DFusionPass: public PatternFusionBasePass
  {
  protected:
    vector<FusionPattern*> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;

  private:
      Status AvgValueTableGen(vector<int64_t> dimInfo, int64_t kernelSize,
              int64_t strideSize, vector<int64_t> padding,
              bool ceilMode, bool countIncludePad,
              ge::Format dataFormat, ge::DataType inputType,
              vector<int64_t> &assitDimInfo, uint16_t *output);
      Status AvgValueTableGenFp32(vector<int64_t> dimInfo, int64_t kernelSize,
              int64_t strideSize, vector<int64_t> padding,
              bool ceilMode, bool countIncludePad,
              ge::Format dataFormat, ge::DataType inputType,
              vector<int64_t> &assitDimInfo, float *output);

      const string FUSED_OP_TYPE = "AvgPool1DD";
  };

}  // namespace fe

#endif  // FE_AVG_POOL_1D_FUSION_H