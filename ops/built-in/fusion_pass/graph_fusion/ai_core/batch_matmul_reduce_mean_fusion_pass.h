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
 * \file batch_matmul_reduce_mean_fusion_pass.h
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_REDUCE_MEAN_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_REDUCE_MEAN_FUSION_PASS_H

#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
struct PadParams {
  int op_pre_peer_idx;
  vector<int64_t> shape;
  vector<vector<int64_t> > paddings;
};

struct SliceParams {
  vector<int64_t> shape;
  vector<int64_t> offsets;
  vector<int64_t> size;
};

class BatchMatMulReduceMeanFusionPass : public PatternFusionBasePass {
protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

private:
  ge::NodePtr batch_matmul_node = nullptr;
  ge::NodePtr add_node = nullptr;
  ge::NodePtr relu_node = nullptr;
  ge::NodePtr reduce_mean_node = nullptr;
  ge::NodePtr slice_node = nullptr;
  int64_t n_dim = 0;
  int64_t n_dim_aligned = 0;

  Status GetNodes(const Mapping &mapping);
  Status CheckNodeShape(const ge::NodePtr &node) const;
  Status CheckStaticShape() const;
  Status CheckAligned();
  Status CheckReduceMean() const;
  Status InsertBatchMatMulPadD(ge::ComputeGraph *graph);
  Status InsertAddPadD(ge::ComputeGraph *graph) const;
  Status InsertReduceMeanSliceD(ge::ComputeGraph *graph);
  Status CreatePadDNode(ge::ComputeGraph *graph, ge::NodePtr *pad_node,
                        const ge::NodePtr &op_node, const PadParams &pad_params) const;
  Status CreateSliceDNode(ge::ComputeGraph *graph, const ge::NodePtr &op_node,
                          const SliceParams &slice_params);
  Status UpdateAllShape(ge::NodePtr *cur_node, const ge::NodePtr &end_node) const;
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_REDUCE_MEAN_FUSION_PASS_H
