/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file batch_matmul_elementwise_ub_fusion.h
 * \brief batch_matmul and all elementwise ops fusion pattern
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_BATCH_MATMUL_ELEMENTWISE_UB_FUSION_H
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_BATCH_MATMUL_ELEMENTWISE_UB_FUSION_H

#include <vector>

#include "common/lxfusion_json_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
struct OffsetIndex {
  size_t offset;
  std::vector<uint32_t> ignore_input_indices;
};

class TbeBatchMatmulElementWiseFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeBatchMatmulElementWiseFusionPass() {}

  ~TbeBatchMatmulElementWiseFusionPass() override {}

 protected:
  vector<BufferFusionPattern *> DefinePatterns() override;
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

 private:
  const string FUSED_OP_TYPE = "FusedOp";
  void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
  Status CheckPattern1(const BufferFusionMapping &mapping) const;
  Status CheckPattern2(const BufferFusionMapping &mapping) const;
  static std::vector<uint32_t> GetIgnoreInputIndices(const ge::NodePtr &node_ptr_curr,
                                                     const std::vector<ge::NodePtr> &fusion_nodes);

  static std::vector<AxisSplitMap> IntersectSplitMap(const std::vector<AxisSplitMap> &map1,
                                                     const std::vector<AxisSplitMap> &map2,
                                                     const struct OffsetIndex &offset_index);

  static size_t GetRealIdx(size_t ori_idx, const struct OffsetIndex &offset_index);

  static bool IntersectSplitMapWithElemwise(ge::NodePtr &nodes, const vector<AxisSplitMap> &split_maps_prev,
                                            vector<AxisSplitMap> *ptr_split_maps_intersect,
                                            size_t *index_already_provide_split_info,
                                            const std::vector<ge::NodePtr> &fusion_nodes);
  static AxisSplitMap GenFusionSplitMap(const AxisSplitMap &map1, const vector<InputSplitInfo> &inputs_map2,
                                        const struct OffsetIndex &offset_index);
  static void TraverseMaps2(const AxisSplitMap &map1, const OutputSplitInfoPtr output_ptr_map1,
                            const std::vector<AxisSplitMap> &maps2, const struct OffsetIndex &offset_index,
                            vector<AxisSplitMap> *intersect_maps);
};
}  // namespace fe

#endif // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TBE_MATMUL_ELEMWISE_H
